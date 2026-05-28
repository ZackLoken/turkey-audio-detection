"""Single-stage frame-level SED training (ConvNeXt-BirdSet, gradual unfreezing).

Pipeline: build a per-clip table -> site/year split -> FrameSedDataset ->
phase loop that progressively unfreezes backbone groups (scaling LR/batch per
phase, with early stopping). Loss is per-frame focal (or BCE). Checkpoint selects
on the validation frame-level F1 (event-level eval is a separate stage).
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from turkey_audio_detection import __version__ as _PKG_VERSION
from turkey_audio_detection.config import SedTrainConfig
from turkey_audio_detection.dataset import CLASS_INDEX, N_CLASSES
from turkey_audio_detection.layout import model_dir
from turkey_audio_detection.manifest import build_stage_manifest, write_manifest
from turkey_audio_detection.models.backbone import DEFAULT_CHECKPOINT
from turkey_audio_detection.models.frame_sed import FrameSed
from turkey_audio_detection.sed_augment import SpecAugment
from turkey_audio_detection.sed_data import SedMelParams, FrameSedDataset
from turkey_audio_detection.sites import attach_site, load_site_map
from turkey_audio_detection.unfreeze import apply_phase, default_schedule
from turkey_audio_detection.training_labels import build_training_table

_INDEX_TO_CLASS = {v: k for k, v in CLASS_INDEX.items()}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FocalLoss(nn.Module):
    """Per-frame binary focal loss for multi-label (Tom/Hen) targets."""

    def __init__(self, gamma: float = 2.0, pos_weight: float = 1.0) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.pos_weight = float(pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw = torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pw)
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        return ((1.0 - p_t) ** self.gamma * bce).mean()


def site_split(table: pd.DataFrame, cfg: SedTrainConfig) -> pd.DataFrame:
    """Assign train/val/test by SITE so no site appears in two splits (leave-site-out)."""
    out = table.copy()
    if "site_id" not in out.columns:
        out["site_id"] = out.get("aru_id", pd.Series([""] * len(out), index=out.index)).astype(str)
    sites = sorted(out["site_id"].astype(str).unique())
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(sites)
    n = len(sites)
    n_val = int(round(n * cfg.val_fraction))
    n_test = int(round(n * cfg.test_fraction))
    val_sites = set(sites[:n_val])
    test_sites = set(sites[n_val : n_val + n_test])

    def _assign(site_id: object) -> str:
        s = str(site_id)
        if s in val_sites:
            return "val"
        if s in test_sites:
            return "test"
        return "train"

    out["split"] = out["site_id"].map(_assign)
    return out


def _build_optimizer(model: FrameSed, lr: float, backbone_mult: float, weight_decay: float):
    backbone_params = [
        p for _n, m in model.backbone.stage_groups() for p in m.parameters() if p.requires_grad
    ]
    head_params = [p for p in model.temporal.parameters() if p.requires_grad]
    head_params += [p for p in model.classifier.parameters() if p.requires_grad]
    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": lr})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr * backbone_mult})
    if not groups:
        return None
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


def _binary_prf(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _batch_mixup(log_mel: torch.Tensor, target: torch.Tensor, alpha: float, p: SedMelParams, rng):
    """Linear-power mixup at batch level (de-normalize -> dB -> blend -> re-normalize)."""
    if alpha <= 0 or log_mel.shape[0] < 2:
        return log_mel, target
    lam = float(rng.beta(alpha, alpha))
    perm = torch.randperm(log_mel.shape[0], device=log_mel.device)
    db = log_mel * p.norm_std + p.norm_mean
    lin = torch.pow(10.0, db / 10.0)
    mixed_lin = lam * lin + (1.0 - lam) * lin[perm]
    mixed_db = 10.0 * torch.log10(torch.clamp(mixed_lin, min=p.amin))
    mixed = (mixed_db - p.norm_mean) / p.norm_std
    return mixed, torch.maximum(target, target[perm])


def _run_epoch(model, loader, criterion, optimizer, device, mel, mixup_alpha, rng, train: bool):
    model.train(train)
    losses: list[float] = []
    yt = [[] for _ in range(N_CLASSES)]
    yp = [[] for _ in range(N_CLASSES)]
    torch.set_grad_enabled(train)
    for log_mel, target, _weak, _ids in loader:
        log_mel = log_mel.to(device)
        target = target.to(device)
        if train and mixup_alpha > 0:
            log_mel, target = _batch_mixup(log_mel, target, mixup_alpha, mel, rng)
        logits = model(log_mel)
        target_ds = F.adaptive_max_pool1d(target, logits.shape[-1])
        loss = criterion(logits, target_ds)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        losses.append(float(loss.item()))
        if not train:
            pred = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
            tgt = (target_ds > 0.5).int().cpu().numpy()
            for c in range(N_CLASSES):
                yp[c].append(pred[:, c, :].reshape(-1))
                yt[c].append(tgt[:, c, :].reshape(-1))
    torch.set_grad_enabled(True)
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    metrics: dict[str, float] = {}
    if not train and any(len(x) for x in yt):
        f1s = []
        for c in range(N_CLASSES):
            name = _INDEX_TO_CLASS[c].lower()
            p_, r_, f_ = _binary_prf(np.concatenate(yt[c]), np.concatenate(yp[c]))
            metrics[f"precision_{name}"] = p_
            metrics[f"recall_{name}"] = r_
            metrics[f"f1_{name}"] = f_
            f1s.append(f_)
        metrics["avg_f1"] = float(np.mean(f1s)) if f1s else 0.0
    return mean_loss, metrics


def train_sed_from_table(table: pd.DataFrame, cfg: SedTrainConfig, project_root: Path) -> dict:
    """Core trainer given a pre-built per-clip table (testable without the pipeline)."""
    _seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    site_map = load_site_map(Path(project_root) / cfg.site_map_path)
    table = attach_site(table, site_map)
    table = site_split(table, cfg)
    train_df = table[table["split"] == "train"].reset_index(drop=True)
    val_df = table[table["split"] == "val"].reset_index(drop=True)
    if train_df.empty:
        raise RuntimeError("Training split is empty after site/year split.")

    mel = SedMelParams()
    aug = [SpecAugment(rng=np.random.default_rng(cfg.seed + 1))] if cfg.specaugment_enabled else []
    train_ds = FrameSedDataset(train_df, cfg.clip_duration_s, mel, augmentations=aug)
    val_ds = FrameSedDataset(val_df, cfg.clip_duration_s, mel) if not val_df.empty else None

    model = FrameSed(
        n_classes=N_CLASSES, n_stages=cfg.n_stages, temporal=cfg.temporal,
        hidden_size=cfg.hidden_size, n_layers=cfg.n_layers, dropout=cfg.dropout,
        pretrained=cfg.pretrained,
    ).to(device)

    criterion: nn.Module = (
        FocalLoss(gamma=cfg.focal_gamma, pos_weight=cfg.pos_weight)
        if cfg.loss == "focal"
        else nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.pos_weight, device=device))
    )
    schedule = default_schedule(
        n_backbone_groups=len(model.backbone.stage_groups()),
        base_lr=cfg.base_lr, base_batch=cfg.base_batch_size,
    )
    mixup_rng = np.random.default_rng(cfg.seed + 2)

    out_dir = model_dir(Path(project_root), cfg.model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.pt"

    best_score = -float("inf")
    best_state = None
    best_meta: dict = {}
    rows: list[dict] = []

    for phase_idx, phase in enumerate(schedule.phases):
        apply_phase(model, schedule, phase_idx)
        optimizer = _build_optimizer(model, phase.lr, cfg.backbone_lr_mult, cfg.weight_decay)
        if optimizer is None:
            continue
        train_loader = DataLoader(
            train_ds, batch_size=phase.batch_size, shuffle=True,
            num_workers=cfg.num_workers, drop_last=False,
        )
        val_loader = (
            DataLoader(val_ds, batch_size=phase.batch_size, num_workers=cfg.num_workers)
            if val_ds is not None else None
        )
        patience = 0
        for epoch in range(1, phase.max_epochs + 1):
            train_loss, _ = _run_epoch(model, train_loader, criterion, optimizer, device, mel, cfg.mixup_alpha, mixup_rng, train=True)
            if val_loader is not None:
                val_loss, metrics = _run_epoch(model, val_loader, criterion, None, device, mel, 0.0, mixup_rng, train=False)
                score = metrics.get("avg_f1", 0.0)
            else:
                val_loss, metrics, score = float("nan"), {}, -train_loss
            rows.append({"phase": phase_idx, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics})

            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                best_meta = {"phase": phase_idx, "epoch": epoch, "score": score}
                patience = 0
            else:
                patience += 1
                if patience >= cfg.early_stop_patience:
                    break

    if best_state is None:
        best_state = model.state_dict()
    torch.save(
        {
            "model_state": best_state,
            "config": cfg.model_dump(mode="json"),
            "mel": {k: getattr(mel, k) for k in ("sample_rate", "n_fft", "hop_length", "n_mels", "fmin", "fmax", "top_db", "norm_mean", "norm_std")},
            "n_stages": cfg.n_stages,
            "time_downsample": model.backbone.time_downsample,
            "backbone_checkpoint": DEFAULT_CHECKPOINT,
            "backbone_config": model.backbone.convnext_config,
            "thresholds": {name: 0.5 for name in CLASS_INDEX},
            "best": best_meta,
            "package_version": _PKG_VERSION,
        },
        ckpt_path,
    )

    pd.DataFrame(rows).to_csv(out_dir / "train_metrics.csv", index=False)
    cols = [c for c in ["item_id", "aru_id", "site_id", "recording_datetime", "split"] if c in table.columns]
    table[cols].to_csv(out_dir / "splits.csv", index=False)
    (out_dir / "config.json").write_text(json.dumps(cfg.model_dump(mode="json"), indent=2), encoding="utf-8")

    manifest = build_stage_manifest(
        run_id=cfg.model_id, stage="train", project_root=Path(project_root),
        config_snapshot={"sed_train": cfg.model_dump(mode="json")},
        stage_outputs={"checkpoint": str(ckpt_path), "metrics_csv": str(out_dir / "train_metrics.csv")},
        status="completed", input_file_count=int(len(table)),
    )
    write_manifest(out_dir / "training_manifest.json", manifest)

    return {
        "model_id": cfg.model_id,
        "best_score": best_score,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int((table["split"] == "test").sum()),
    }


def train_sed(cfg: SedTrainConfig, project_root: Path) -> dict:
    """Build the training table from labeled runs, then train."""
    table = build_training_table(Path(project_root), cfg.run_ids, include_non_consensus=cfg.include_non_consensus)
    if table.empty:
        raise RuntimeError(f"No labeled clips found in runs {cfg.run_ids}.")
    return train_sed_from_table(table, cfg, project_root)

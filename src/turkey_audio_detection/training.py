"""Region-level SED training loop."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from turkey_audio_detection import __version__ as _PACKAGE_VERSION
from turkey_audio_detection.augment import BackgroundMix, Mixup, SpecAugment
from turkey_audio_detection.config import TrainConfig
from turkey_audio_detection.dataset import (
    MelParams,
    TurkeyClipDataset,
    load_log_mel,
)
from turkey_audio_detection.layout import model_dir
from turkey_audio_detection.manifest import build_stage_manifest, write_manifest
from turkey_audio_detection.models import CnnSed
from turkey_audio_detection.training_labels import build_training_table


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_precision_tom: float
    val_recall_tom: float
    val_f1_tom: float
    val_precision_hen: float
    val_recall_hen: float
    val_f1_hen: float


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stratified_split(table: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    """Assign train/val/test split per row using (aru_id, recording_date) groups.

    Same (aru, date) always lands in the same split → prevents same-recording leakage.
    """
    out = table.copy()
    if "aru_id" not in out.columns:
        out["aru_id"] = ""
    if "recording_datetime" not in out.columns:
        out["recording_datetime"] = ""
    out["__date"] = (
        pd.to_datetime(out["recording_datetime"], errors="coerce").dt.date.astype("string").fillna("unknown")
    )
    groups = sorted(out.groupby(["aru_id", "__date"]).groups.keys())
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(groups)
    n = len(groups)
    n_val = int(round(n * cfg.val_fraction))
    n_test = int(round(n * cfg.test_fraction))
    val = set(groups[:n_val])
    test = set(groups[n_val : n_val + n_test])

    def _split_for(row) -> str:
        key = (row["aru_id"], row["__date"])
        if key in val:
            return "val"
        if key in test:
            return "test"
        return "train"

    out["split"] = out.apply(_split_for, axis=1)
    out.drop(columns="__date", inplace=True)
    return out


def _mask_to_clip_label(mask: torch.Tensor) -> torch.Tensor:
    """(B, C, M, T) → (B, C) clip-level booleans via max-pool."""
    return (mask.amax(dim=(2, 3)) > 0.5).float()


def _logits_to_clip_score(logits: torch.Tensor) -> torch.Tensor:
    """(B, C, M, T) → (B, C) clip-level scores via global max + sigmoid."""
    pooled, _ = logits.flatten(2).max(dim=-1)
    return torch.sigmoid(pooled)


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Precision, recall, F1 for a single binary class (1d arrays of 0/1)."""
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _evaluate(model: CnnSed, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    with torch.no_grad():
        for log_mel, mask, _weak, _ids in loader:
            log_mel = log_mel.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            logits = model(log_mel)
            loss = criterion(logits, mask)
            losses.append(loss.item())
            scores = _logits_to_clip_score(logits)
            y_pred_all.append((scores > 0.5).int().cpu().numpy())
            y_true_all.append(_mask_to_clip_label(mask).int().cpu().numpy())
    if not losses:
        return float("nan"), np.zeros((0, 2), dtype=int), np.zeros((0, 2), dtype=int)
    return float(np.mean(losses)), np.concatenate(y_true_all), np.concatenate(y_pred_all)


def _sample_background_mel(
    background_table: pd.DataFrame,
    clip_duration_s: float,
    mel: MelParams,
) -> np.ndarray | None:
    if background_table.empty:
        return None
    row = background_table.sample(1).iloc[0]
    try:
        log_mel, _ = load_log_mel(str(row["clip_path"]), clip_duration_s, mel)
        return log_mel
    except Exception:
        return None


def train(cfg: TrainConfig, project_root: Path) -> dict:
    """Train a CnnSed model. Returns a dict summarizing the run."""
    _seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    table = build_training_table(project_root, cfg.run_ids, include_non_consensus=cfg.include_non_consensus)
    if table.empty:
        raise RuntimeError(
            f"No labeled clips found in runs {cfg.run_ids}. Aggregator returned an empty table."
        )
    table = _stratified_split(table, cfg)
    train_df = table[table["split"] == "train"].reset_index(drop=True)
    val_df = table[table["split"] == "val"].reset_index(drop=True)
    if train_df.empty:
        raise RuntimeError("Training split is empty after stratified group split.")

    mel = MelParams()
    background_df = train_df[(train_df["tom_present"] == 0) & (train_df["hen_present"] == 0)]

    spec_aug = SpecAugment(rng=np.random.default_rng(cfg.seed + 1))
    mixup = Mixup(alpha=cfg.mixup_alpha, rng=np.random.default_rng(cfg.seed + 2))
    bgmix = BackgroundMix(rng=np.random.default_rng(cfg.seed + 3))

    train_augmentations: list = []
    if cfg.background_mix_enabled:
        train_augmentations.append(bgmix)
    if cfg.mixup_alpha > 0:
        train_augmentations.append(mixup)
    if cfg.specaugment_enabled:
        train_augmentations.append(spec_aug)

    train_ds = TurkeyClipDataset(
        train_df,
        clip_duration_s=cfg.clip_duration_s,
        mel=mel,
        augmentations=train_augmentations,
    )
    val_ds = TurkeyClipDataset(val_df, clip_duration_s=cfg.clip_duration_s, mel=mel)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    model = CnnSed(n_classes=2, pretrained=cfg.pretrained).to(device)
    pos_weight = torch.tensor([cfg.pos_weight, cfg.pos_weight], device=device).view(1, 2, 1, 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.GradScaler(enabled=device.type == "cuda")

    out_dir = model_dir(project_root, cfg.model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[EpochMetrics] = []
    best_f1 = -1.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_losses: list[float] = []

        # Build a queue of partner indices for mixup
        partner_idx = list(range(len(train_ds)))
        random.shuffle(partner_idx)

        for batch_idx, (log_mel, mask, weak, _ids) in enumerate(train_loader):
            # Prepare augmentation context for the next iteration's items
            if cfg.background_mix_enabled:
                bg = _sample_background_mel(background_df, cfg.clip_duration_s, mel)
                if bg is not None:
                    bgmix.set_background(bg)
            if cfg.mixup_alpha > 0 and partner_idx:
                p_idx = partner_idx.pop()
                try:
                    p_mel, p_mask, p_weak, _ = train_ds[p_idx]
                    mixup.set_partner((p_mel.numpy(), p_mask.numpy(), p_weak.numpy()))
                except Exception:
                    mixup.set_partner(None)

            log_mel = log_mel.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(  # type: ignore[attr-defined]
                device_type=device.type, enabled=device.type == "cuda"
            ):
                logits = model(log_mel)
                loss = criterion(logits, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(loss.item())

        scheduler.step()

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        val_loss, y_true, y_pred = _evaluate(model, val_loader, criterion, device)
        if y_true.size > 0:
            p_tom, r_tom, f1_tom = _classification_metrics(y_true[:, 0], y_pred[:, 0])
            p_hen, r_hen, f1_hen = _classification_metrics(y_true[:, 1], y_pred[:, 1])
        else:
            p_tom = r_tom = f1_tom = p_hen = r_hen = f1_hen = 0.0

        metrics_rows.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_precision_tom=p_tom,
                val_recall_tom=r_tom,
                val_f1_tom=f1_tom,
                val_precision_hen=p_hen,
                val_recall_hen=r_hen,
                val_f1_hen=f1_hen,
            )
        )

        avg_f1 = (f1_tom + f1_hen) / 2.0
        print(
            f"epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"tom_f1={f1_tom:.3f} | hen_f1={f1_hen:.3f}"
        )

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg.model_dump(mode="json"),
                    "epoch": epoch,
                    "best_avg_f1": best_f1,
                    "package_version": _PACKAGE_VERSION,
                },
                out_dir / "checkpoint.pt",
            )

    pd.DataFrame([asdict(m) for m in metrics_rows]).to_csv(out_dir / "train_metrics.csv", index=False)
    table[["item_id", "aru_id", "recording_datetime", "split"]].to_csv(out_dir / "splits.csv", index=False)
    (out_dir / "config.json").write_text(json.dumps(cfg.model_dump(mode="json"), indent=2), encoding="utf-8")

    manifest = build_stage_manifest(
        run_id=cfg.model_id,
        stage="train",
        project_root=project_root,
        config_snapshot={"train": cfg.model_dump(mode="json")},
        stage_outputs={
            "checkpoint": str(out_dir / "checkpoint.pt"),
            "metrics_csv": str(out_dir / "train_metrics.csv"),
            "splits_csv": str(out_dir / "splits.csv"),
        },
        status="completed",
        input_file_count=int(len(table)),
    )
    write_manifest(out_dir / "training_manifest.json", manifest)

    return {
        "model_id": cfg.model_id,
        "best_avg_f1": best_f1,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int((table["split"] == "test").sum()),
    }

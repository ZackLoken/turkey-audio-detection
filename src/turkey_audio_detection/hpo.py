"""Optuna hyperparameter search for the frame-level SED model.

Each trial samples model/training hyperparameters, trains a FrameSed via the
gradual-unfreeze trainer, and scores the validation split with event-level F1.
The study is resumable when a SQLite `storage` path is given.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import optuna
import torch

from turkey_audio_detection.config import SedTrainConfig
from turkey_audio_detection.dataset import CLASS_INDEX
from turkey_audio_detection.evaluation import evaluate_table
from turkey_audio_detection.layout import model_dir
from turkey_audio_detection.sed_inference import load_sed_model
from turkey_audio_detection.sed_training import site_year_split, train_sed_from_table
from turkey_audio_detection.sites import attach_site, load_site_map


def suggest_config(trial: optuna.Trial, base: SedTrainConfig) -> SedTrainConfig:
    """Sample a SedTrainConfig from `base`, overriding the tuned fields."""
    overrides = dict(
        temporal=trial.suggest_categorical("temporal", ["bigru", "tcn"]),
        hidden_size=trial.suggest_categorical("hidden_size", [128, 256, 512]),
        n_layers=trial.suggest_int("n_layers", 1, 3),
        dropout=trial.suggest_float("dropout", 0.0, 0.5),
        focal_gamma=trial.suggest_float("focal_gamma", 0.0, 3.0),
        pos_weight=trial.suggest_float("pos_weight", 1.0, 20.0, log=True),
        base_lr=trial.suggest_float("base_lr", 1e-4, 3e-3, log=True),
        n_stages=trial.suggest_int("n_stages", 1, 3),
        backbone_lr_mult=trial.suggest_float("backbone_lr_mult", 0.05, 0.5, log=True),
    )
    return base.model_copy(update=overrides)


def _val_split(table, cfg: SedTrainConfig, project_root: Path):
    site_map = load_site_map(Path(project_root) / cfg.site_map_path)
    t = attach_site(table, site_map)
    t = site_year_split(t, cfg)
    return t[t["split"] == "val"].reset_index(drop=True)


def objective(trial, table, project_root: Path, base_cfg: SedTrainConfig, iou: float = 0.3) -> float:
    cfg = suggest_config(trial, base_cfg)
    cfg = cfg.model_copy(update={"model_id": f"hpo_trial_{trial.number}"})
    train_sed_from_table(table, cfg, project_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_sed_model(model_dir(Path(project_root), cfg.model_id) / "checkpoint.pt", device)
    val_df = _val_split(table, cfg, project_root)
    if val_df.empty:
        return 0.0
    result = evaluate_table(model, val_df, payload, device, iou_thresholds=(iou,), clip_duration_s=cfg.clip_duration_s)
    f1s = [result["event"][iou][name]["f1"] for name in CLASS_INDEX]
    return float(np.mean(f1s)) if f1s else 0.0


def run_hpo(
    table,
    project_root: Path,
    base_cfg: SedTrainConfig,
    n_trials: int = 20,
    storage: str | None = None,
    study_name: str = "sed_hpo",
    iou: float = 0.3,
) -> optuna.Study:
    """Run (or resume) an Optuna study; returns the study (best params on study.best_params)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage_url = f"sqlite:///{Path(storage).as_posix()}" if storage else None
    study = optuna.create_study(
        direction="maximize", study_name=study_name, storage=storage_url, load_if_exists=True,
    )
    study.optimize(lambda t: objective(t, table, project_root, base_cfg, iou), n_trials=n_trials)
    return study

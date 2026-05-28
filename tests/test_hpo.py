"""Tests for the Optuna HPO scaffolding (hpo.py)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import soundfile as sf
from optuna.trial import FixedTrial

from turkey_audio_detection.config import SedTrainConfig
from turkey_audio_detection.hpo import run_hpo, suggest_config


def test_suggest_config_maps_params_and_keeps_base() -> None:
    trial = FixedTrial({
        "temporal": "tcn", "hidden_size": 128, "n_layers": 2, "dropout": 0.1,
        "focal_gamma": 1.5, "pos_weight": 5.0, "base_lr": 1e-3, "n_stages": 2,
        "backbone_lr_mult": 0.1,
    })
    cfg = suggest_config(trial, SedTrainConfig(pretrained=False, num_workers=0))
    assert cfg.temporal == "tcn"
    assert cfg.hidden_size == 128
    assert cfg.n_stages == 2
    assert cfg.pretrained is False  # base field carried through
    assert cfg.num_workers == 0


def test_run_hpo_smoke(tmp_path, monkeypatch) -> None:
    from turkey_audio_detection import sed_training
    from turkey_audio_detection.unfreeze import Phase, UnfreezeSchedule

    monkeypatch.setattr(
        sed_training, "default_schedule",
        lambda **kw: UnfreezeSchedule([Phase(n_trainable_top_groups=2, lr=1e-3, batch_size=4, max_epochs=1)]),
    )

    sr = 32000
    rng = np.random.default_rng(0)
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    rows = []
    for i in range(4):
        wav = (0.1 * rng.standard_normal(sr * 3)).astype(np.float32)
        cp = clips_dir / f"c{i}.wav"
        sf.write(str(cp), wav, sr)
        regions = json.dumps([{"label": "Tom", "start_s": 1.0, "end_s": 2.0}]) if i % 2 == 0 else "[]"
        rows.append({
            "item_id": f"it{i}", "aru_id": f"ARU_{i:02d}",
            "recording_datetime": "2026-04-01 06:00:00", "clip_path": str(cp),
            "regions_json": regions, "tom_present": 1 if i % 2 == 0 else 0, "hen_present": 0,
        })
    table = pd.DataFrame(rows)

    base = SedTrainConfig(
        pretrained=False, num_workers=0, val_fraction=0.5, test_fraction=0.0,
        specaugment_enabled=False, early_stop_patience=1, site_map_path="nope.csv",
        seed=0, clip_duration_s=3.0,
    )
    study = run_hpo(table, tmp_path, base, n_trials=2, storage=str(tmp_path / "study.db"))

    assert len(study.trials) == 2
    assert isinstance(study.best_value, float)
    assert 0.0 <= study.best_value <= 1.0
    assert (tmp_path / "study.db").exists()

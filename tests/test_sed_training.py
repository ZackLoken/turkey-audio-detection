"""End-to-end smoke test for the SED trainer (synthetic clips, pretrained=False)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from turkey_audio_detection import sed_training
from turkey_audio_detection.config import SedTrainConfig
from turkey_audio_detection.layout import model_dir
from turkey_audio_detection.unfreeze import Phase, UnfreezeSchedule


def test_train_sed_from_table_smoke(tmp_path, monkeypatch) -> None:
    # 1-phase, 1-epoch, head-only schedule so the smoke test is fast.
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
        positive = i % 2 == 0
        regions = (
            json.dumps([{"start_s": 1.0, "end_s": 2.0, "freq_min_hz": 300, "freq_max_hz": 1500, "label": "Tom"}])
            if positive else "[]"
        )
        rows.append({
            "item_id": f"it{i}",
            "aru_id": f"ARU_{i:02d}",  # 4 distinct ARUs -> 4 sites (no site_map -> fallback)
            "recording_datetime": "2026-04-01 06:00:00",
            "clip_path": str(cp),
            "regions_json": regions,
            "tom_present": 1 if positive else 0,
            "hen_present": 0,
        })
    table = pd.DataFrame(rows)

    cfg = SedTrainConfig(
        model_id="m_test", clip_duration_s=3.0, pretrained=False,
        n_stages=2, hidden_size=16, n_layers=1,
        val_fraction=0.25, test_fraction=0.25, num_workers=0,
        specaugment_enabled=False, early_stop_patience=1, site_map_path="nope.csv", seed=0,
    )

    res = sed_training.train_sed_from_table(table, cfg, tmp_path)
    assert res["model_id"] == "m_test"
    assert res["n_train"] + res["n_val"] + res["n_test"] == 4
    assert res["n_train"] >= 1

    ckpt_path = model_dir(tmp_path, "m_test") / "checkpoint.pt"
    assert ckpt_path.exists()
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    for key in ("model_state", "config", "mel", "n_stages", "time_downsample", "backbone_checkpoint", "backbone_config", "thresholds", "package_version"):
        assert key in payload
    assert payload["n_stages"] == 2
    assert payload["time_downsample"] == 8
    assert set(payload["thresholds"]) == {"Tom", "Hen"}

"""Test the no-code app's analysis wrapper (UI itself is not unit-tested)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from turkey_audio_detection.app_infer import analyze_recording_file
from turkey_audio_detection.config import SedInferConfig
from turkey_audio_detection.layout import model_dir
from turkey_audio_detection.models.frame_sed import FrameSed
from turkey_audio_detection.sed_data import SedMelParams


def test_analyze_recording_file(tmp_path) -> None:
    mel = SedMelParams()
    # save a (random-weight) checkpoint where the trainer would put it
    out_dir = model_dir(tmp_path, "m_app")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = FrameSed(n_classes=2, n_stages=2, hidden_size=16, n_layers=1, pretrained=False)
    torch.save(
        {
            "config": {"n_stages": 2, "temporal": "bigru", "hidden_size": 16, "n_layers": 1, "dropout": 0.2},
            "model_state": model.state_dict(),
            "time_downsample": 8,
            "thresholds": {"Tom": 0.5, "Hen": 0.5},
            "backbone_config": model.backbone.convnext_config,
        },
        out_dir / "checkpoint.pt",
    )

    rng = np.random.default_rng(0)
    wav = (0.1 * rng.standard_normal(mel.sample_rate * 5)).astype(np.float32)
    audio_path = tmp_path / "2MA_20260401_060000.wav"
    sf.write(str(audio_path), wav, mel.sample_rate)
    windows = pd.DataFrame([{"start_time_s": 1.0, "end_time_s": 1.5}])

    cfg = SedInferConfig(model_id="m_app", thresholds={"Tom": 0.0, "Hen": 0.0}, site_map_path="nope.csv")
    events, counts = analyze_recording_file("m_app", audio_path, windows, tmp_path, cfg=cfg)

    assert not events.empty
    assert {"event_id", "start_time_s", "end_time_s", "sex", "score"}.issubset(events.columns)
    assert {"site_id", "date", "sex", "n_events", "present"}.issubset(counts.columns)

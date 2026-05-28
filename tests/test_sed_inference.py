"""Tests for candidate-gated frame-level SED inference (sed_inference.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from turkey_audio_detection.config import SedInferConfig
from turkey_audio_detection.models.frame_sed import FrameSed
from turkey_audio_detection.sed_data import LogMelExtractor, SedMelParams
from turkey_audio_detection.sed_inference import (
    aggregate_counts,
    frames_to_events,
    infer_recording,
    load_sed_model,
    stitch_windows,
)

HOP_S = 0.08  # 320 * 8 / 32000


def test_frames_to_events_basic() -> None:
    prob = np.zeros(100, dtype=np.float32)
    prob[10:20] = 0.9
    events = frames_to_events(prob, threshold=0.5, min_duration_s=0.1, merge_gap_s=0.2, hop_s=HOP_S)
    assert len(events) == 1
    assert abs(events[0]["start_s"] - 0.8) < 1e-6
    assert abs(events[0]["end_s"] - 1.6) < 1e-6
    assert abs(events[0]["score"] - 0.9) < 1e-4


def test_frames_to_events_merges_within_gap() -> None:
    prob = np.zeros(100, dtype=np.float32)
    prob[10:20] = 0.9
    prob[22:30] = 0.9  # gap of 2 frames = 0.16 s <= merge_gap 0.2 s
    events = frames_to_events(prob, threshold=0.5, min_duration_s=0.1, merge_gap_s=0.2, hop_s=HOP_S)
    assert len(events) == 1
    assert abs(events[0]["start_s"] - 0.8) < 1e-6
    assert abs(events[0]["end_s"] - 2.4) < 1e-6


def test_frames_to_events_min_duration_filter() -> None:
    prob = np.zeros(100, dtype=np.float32)
    prob[10:12] = 0.9  # 2 frames = 0.16 s
    events = frames_to_events(prob, threshold=0.5, min_duration_s=0.3, merge_gap_s=0.0, hop_s=HOP_S)
    assert events == []


def test_stitch_windows_averages_overlap() -> None:
    a = np.full((2, 5), 0.4, dtype=np.float32)
    b = np.full((2, 5), 0.8, dtype=np.float32)
    timeline = stitch_windows([(0, a), (3, b)], total_frames=8)
    assert timeline.shape == (2, 8)
    assert np.allclose(timeline[:, 0], 0.4)
    assert np.allclose(timeline[:, 3], 0.6)  # (0.4 + 0.8) / 2
    assert np.allclose(timeline[:, 7], 0.8)


def test_aggregate_counts() -> None:
    events = pd.DataFrame([
        {"source_audio_path": "data/ARU_01/x/2MA_20260401_060000.wav", "aru_id": "ARU_01", "sex": "Tom"},
        {"source_audio_path": "data/ARU_01/x/2MA_20260401_060000.wav", "aru_id": "ARU_01", "sex": "Tom"},
        {"source_audio_path": "data/ARU_02/x/2MB_20260401_060000.wav", "aru_id": "ARU_02", "sex": "Hen"},
    ])
    agg = aggregate_counts(events, {"ARU_01": "S1", "ARU_02": "S1"})
    tom = agg[(agg.site_id == "S1") & (agg.sex == "Tom")]
    assert int(tom["n_events"].iloc[0]) == 2
    assert int(tom["present"].iloc[0]) == 1


def test_load_sed_model(tmp_path) -> None:
    model = FrameSed(n_classes=2, n_stages=2, hidden_size=16, n_layers=1, pretrained=False)
    payload = {
        "config": {"n_stages": 2, "temporal": "bigru", "hidden_size": 16, "n_layers": 1, "dropout": 0.2},
        "model_state": model.state_dict(),
        "time_downsample": 8,
        "thresholds": {"Tom": 0.5, "Hen": 0.5},
    }
    p = tmp_path / "checkpoint.pt"
    torch.save(payload, p)
    loaded, pl = load_sed_model(p, torch.device("cpu"))
    with torch.no_grad():
        out = loaded(torch.randn(1, 128, 301))
    assert out.shape[1] == 2
    assert pl["time_downsample"] == 8


def test_load_sed_model_reconstructs_nondefault_architecture(tmp_path) -> None:
    # Regression: a checkpoint trained with a non-default backbone config must reload
    # via its saved backbone_config (not the pretrained=False default), else
    # load_state_dict mismatches. Mirrors the real pretrained=True (Base) case.
    from transformers import ConvNextConfig

    cfg = ConvNextConfig(num_channels=1, hidden_sizes=[64, 128, 256, 512], depths=[1, 1, 1, 1])
    model = FrameSed(n_classes=2, n_stages=2, hidden_size=16, n_layers=1, pretrained=False, config_dict=cfg.to_dict())
    payload = {
        "config": {"n_stages": 2, "temporal": "bigru", "hidden_size": 16, "n_layers": 1, "dropout": 0.2},
        "model_state": model.state_dict(),
        "time_downsample": 8,
        "thresholds": {"Tom": 0.5, "Hen": 0.5},
        "backbone_config": model.backbone.convnext_config,
    }
    p = tmp_path / "checkpoint.pt"
    torch.save(payload, p)
    loaded, _pl = load_sed_model(p, torch.device("cpu"))
    assert loaded.backbone.out_channels == 128  # hidden_sizes[1]; default would be 192
    with torch.no_grad():
        out = loaded(torch.randn(1, 128, 301))
    assert out.shape[1] == 2


def test_infer_recording_end_to_end(tmp_path) -> None:
    mel = SedMelParams()
    rng = np.random.default_rng(0)
    wav = (0.1 * rng.standard_normal(mel.sample_rate * 6)).astype(np.float32)  # 6 s recording
    audio_path = tmp_path / "2MA_20260401_060000.wav"
    sf.write(str(audio_path), wav, mel.sample_rate)

    windows = pd.DataFrame([
        {"start_time_s": 1.0, "end_time_s": 1.5},
        {"start_time_s": 3.5, "end_time_s": 4.0},
    ])
    model = FrameSed(n_classes=2, n_stages=2, hidden_size=16, n_layers=1, pretrained=False).eval()
    payload = {"time_downsample": 8, "thresholds": {"Tom": 0.0, "Hen": 0.0}}  # thr 0 -> guaranteed events
    cfg = SedInferConfig(model_id="m", candidate_window_duration_s=3.0, min_event_duration_s=0.1, merge_gap_s=0.2)
    extractor = LogMelExtractor(mel)

    events = infer_recording(audio_path, windows, model, mel, extractor, payload, cfg, torch.device("cpu"), "inf_1")
    assert not events.empty
    for col in ("event_id", "source_audio_path", "start_time_s", "end_time_s", "sex", "score"):
        assert col in events.columns
    assert (events["start_time_s"] >= 0).all()
    assert (events["end_time_s"] <= 6.01).all()
    assert set(events["sex"]).issubset({"Tom", "Hen"})

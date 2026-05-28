"""Tests for the frame-level SED data pipeline (sed_data.py)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import soundfile as sf

from turkey_audio_detection.dataset import CLASS_INDEX, N_CLASSES, parse_regions
from turkey_audio_detection.sed_data import (
    FrameSedDataset,
    LogMelExtractor,
    SedMelParams,
    downsample_targets,
    normalize_log_mel,
    regions_to_frame_targets,
)

P = SedMelParams()


def test_parse_regions_valid_and_invalid() -> None:
    parsed = parse_regions(json.dumps([{"start_s": 1.0, "end_s": 2.0, "label": "Tom"}]))
    assert parsed == [{"start_s": 1.0, "end_s": 2.0, "label": "Tom"}]
    assert parse_regions(None) == []
    assert parse_regions("") == []
    assert parse_regions("nan") == []
    assert parse_regions("not json") == []
    assert parse_regions(float("nan")) == []
    assert parse_regions(json.dumps({"not": "a list"})) == []


def test_regions_to_frame_targets_single_tom() -> None:
    region = {"start_s": 1.0, "end_s": 2.0, "freq_min_hz": 300.0, "freq_max_hz": 1500.0, "label": "Tom"}
    target = regions_to_frame_targets([region], n_frames=300, p=P)
    assert target.shape == (N_CLASSES, 300)
    # 1.0 s -> frame 100, 2.0 s -> frame 200 (32000/320 = 100 frames/s).
    tom = CLASS_INDEX["Tom"]
    hen = CLASS_INDEX["Hen"]
    assert target[tom, 100:200].sum() == 100.0
    assert target[tom, :100].sum() == 0.0
    assert target[tom, 200:].sum() == 0.0
    assert target[hen].sum() == 0.0


def test_regions_to_frame_targets_empty_is_all_zero() -> None:
    target = regions_to_frame_targets([], n_frames=120, p=P)
    assert target.shape == (N_CLASSES, 120)
    assert target.sum() == 0.0


def test_regions_to_frame_targets_ignores_unknown_label() -> None:
    region = {"start_s": 0.5, "end_s": 1.0, "label": "Coyote"}
    target = regions_to_frame_targets([region], n_frames=120, p=P)
    assert target.sum() == 0.0


def test_downsample_targets_preserves_localized_presence() -> None:
    target = np.zeros((N_CLASSES, 300), dtype=np.float32)
    target[CLASS_INDEX["Tom"], 100:200] = 1.0
    pooled = downsample_targets(target, t_out=75)
    assert pooled.shape == (N_CLASSES, 75)
    # frames 100:200 at /4 stride map to output frames 25:50.
    assert pooled[CLASS_INDEX["Tom"], 25:50].sum() == 25.0
    assert pooled[CLASS_INDEX["Tom"], :25].sum() == 0.0
    assert pooled[CLASS_INDEX["Tom"], 50:].sum() == 0.0
    assert pooled[CLASS_INDEX["Hen"]].sum() == 0.0


def test_downsample_targets_noop_when_lengths_match() -> None:
    target = np.zeros((N_CLASSES, 75), dtype=np.float32)
    out = downsample_targets(target, t_out=75)
    assert out.shape == (N_CLASSES, 75)


def test_log_mel_extractor_shape_and_topdb() -> None:
    rng = np.random.default_rng(0)
    wav = rng.standard_normal(P.sample_rate * 3).astype(np.float32)  # 3 s
    import torch

    mel = LogMelExtractor(P)(torch.from_numpy(wav)).numpy()
    assert mel.shape[0] == P.n_mels
    assert mel.shape[1] == 301  # 96000 // 320 + 1
    assert np.isfinite(mel).all()
    # PowerToDB top_db clamp => dynamic range bounded by top_db.
    assert (mel.max() - mel.min()) <= P.top_db + 1e-3


def test_normalize_log_mel() -> None:
    db = np.full((4, 4), P.norm_mean + P.norm_std, dtype=np.float32)
    out = normalize_log_mel(db, P)
    assert np.allclose(out, 1.0, atol=1e-5)


def test_frame_sed_dataset_getitem(tmp_path) -> None:
    rng = np.random.default_rng(1)
    wav = (0.1 * rng.standard_normal(P.sample_rate * 3)).astype(np.float32)
    clip_path = tmp_path / "clip.wav"
    sf.write(str(clip_path), wav, P.sample_rate)

    regions = [{"start_s": 1.0, "end_s": 2.0, "freq_min_hz": 300.0, "freq_max_hz": 1500.0, "label": "Tom"}]
    table = pd.DataFrame(
        [{
            "item_id": "it_1",
            "clip_path": str(clip_path),
            "regions_json": json.dumps(regions),
            "tom_present": 1,
            "hen_present": 0,
        }]
    )
    ds = FrameSedDataset(table, clip_duration_s=3.0, mel=P)
    log_mel, target, weak, item_id = ds[0]

    assert log_mel.shape == (P.n_mels, 301)
    assert target.shape == (N_CLASSES, 301)
    assert tuple(weak.tolist()) == (1.0, 0.0)
    assert item_id == "it_1"
    assert target[CLASS_INDEX["Tom"]].sum() > 0
    assert target[CLASS_INDEX["Hen"]].sum() == 0

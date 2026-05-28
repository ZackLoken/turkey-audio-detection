"""Tests for event-level + segment-level evaluation (evaluation.py)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from turkey_audio_detection.evaluation import (
    evaluate_table,
    match_events,
    prf,
    regions_to_events,
    segment_f1,
    time_iou,
)
from turkey_audio_detection.models.frame_sed import FrameSed
from turkey_audio_detection.sed_data import SedMelParams


def test_time_iou() -> None:
    assert abs(time_iou(0, 10, 5, 15) - (5 / 15)) < 1e-6
    assert time_iou(0, 10, 0, 10) == 1.0
    assert time_iou(0, 5, 10, 15) == 0.0


def test_match_events_tp_fp_fn() -> None:
    gt = [(0.0, 10.0), (20.0, 30.0)]
    pred = [(1.0, 9.0), (50.0, 60.0)]
    m = match_events(gt, pred, iou_threshold=0.3)
    assert m == {"tp": 1, "fp": 1, "fn": 1}


def test_match_events_below_threshold_is_no_match() -> None:
    m = match_events([(0.0, 10.0)], [(9.0, 19.0)], iou_threshold=0.5)  # iou ~ 1/19
    assert m == {"tp": 0, "fp": 1, "fn": 1}


def test_prf() -> None:
    assert prf(1, 1, 1) == (0.5, 0.5, 0.5)
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)


def test_segment_f1() -> None:
    assert segment_f1([(0.0, 2.0)], [(0.0, 2.0)], duration_s=4.0, seg_s=1.0) == 1.0
    assert segment_f1([(0.0, 2.0)], [], duration_s=4.0, seg_s=1.0) == 0.0


def test_regions_to_events() -> None:
    regions = [
        {"label": "Tom", "start_s": 1.0, "end_s": 2.0},
        {"label": "Hen", "start_s": 0.5, "end_s": 0.8},
        {"label": "Tom", "start_s": 3.0, "end_s": 3.0},  # zero-length -> dropped
    ]
    assert regions_to_events(regions, "Tom") == [(1.0, 2.0)]
    assert regions_to_events(regions, "Hen") == [(0.5, 0.8)]


def test_evaluate_table_structure(tmp_path) -> None:
    mel = SedMelParams()
    rng = np.random.default_rng(0)
    rows = []
    for i in range(2):
        wav = (0.1 * rng.standard_normal(mel.sample_rate * 3)).astype(np.float32)
        cp = tmp_path / f"c{i}.wav"
        sf.write(str(cp), wav, mel.sample_rate)
        regions = json.dumps([{"label": "Tom", "start_s": 1.0, "end_s": 2.0}])
        rows.append({"item_id": f"it{i}", "clip_path": str(cp), "regions_json": regions})
    table = pd.DataFrame(rows)

    model = FrameSed(n_classes=2, n_stages=2, hidden_size=16, n_layers=1, pretrained=False)
    payload = {"time_downsample": 8, "thresholds": {"Tom": 0.0, "Hen": 0.0}}
    result = evaluate_table(
        model, table, payload, torch.device("cpu"), mel=mel,
        iou_thresholds=(0.1, 0.3), seg_s=1.0,
    )
    assert set(result["event"].keys()) == {0.1, 0.3}
    for iou in (0.1, 0.3):
        assert set(result["event"][iou].keys()) == {"Tom", "Hen"}
        for cls in ("Tom", "Hen"):
            for k in ("precision", "recall", "f1"):
                assert 0.0 <= result["event"][iou][cls][k] <= 1.0
    assert set(result["segment_f1"].keys()) == {"Tom", "Hen"}

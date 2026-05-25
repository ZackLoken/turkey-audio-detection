"""Tests for the inference postprocessing helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turkey_audio_detection.config import InferConfig
from turkey_audio_detection.dataset import MelParams
from turkey_audio_detection.inference import (
    _components_to_events,
    _merge_adjacent,
    _window_indices,
    make_event_id,
)


P = MelParams()


def _cfg(**overrides) -> InferConfig:
    base = dict(
        model_id="test",
        audio_glob="",
        score_threshold=0.5,
        min_event_duration_s=0.0,
        merge_gap_s=0.0,
    )
    base.update(overrides)
    return InferConfig(**base)


def test_window_indices_short_audio() -> None:
    starts = _window_indices(n_samples=1000, window_n=2000, stride_n=500)
    assert starts == [0]


def test_window_indices_covers_tail() -> None:
    starts = _window_indices(n_samples=10_000, window_n=3000, stride_n=1500)
    # Every sample should be within at least one window.
    covered = set()
    for s in starts:
        for x in range(s, min(10_000, s + 3000)):
            covered.add(x)
    assert covered == set(range(10_000))


def test_components_to_events_extracts_box() -> None:
    prob = np.zeros((P.n_mels, 300), dtype=np.float32)
    prob[20:40, 100:200] = 0.9
    events = _components_to_events(
        prob_class=prob,
        class_label="Tom",
        audio_path=Path("test.wav"),
        aru_id="ARU_01",
        cfg=_cfg(),
        mel_params=P,
        model_id="m1",
        inference_id="i1",
    )
    assert len(events) == 1
    evt = events[0]
    assert evt["label"] == "Tom"
    assert evt["start_time_s"] == pytest.approx(100 * P.hop_length / P.sample_rate, abs=0.01)
    assert evt["end_time_s"] == pytest.approx(200 * P.hop_length / P.sample_rate, abs=0.01)
    assert evt["score"] > 0.5


def test_components_to_events_filters_by_min_duration() -> None:
    prob = np.zeros((P.n_mels, 300), dtype=np.float32)
    prob[20:30, 100:105] = 0.9  # very short event
    events = _components_to_events(
        prob_class=prob,
        class_label="Tom",
        audio_path=Path("test.wav"),
        aru_id="",
        cfg=_cfg(min_event_duration_s=1.0),
        mel_params=P,
        model_id="m1",
        inference_id="i1",
    )
    assert events == []


def test_components_to_events_drops_below_threshold() -> None:
    prob = np.zeros((P.n_mels, 300), dtype=np.float32)
    prob[20:40, 100:200] = 0.3  # below default threshold 0.5
    events = _components_to_events(
        prob_class=prob,
        class_label="Tom",
        audio_path=Path("test.wav"),
        aru_id="",
        cfg=_cfg(),
        mel_params=P,
        model_id="m1",
        inference_id="i1",
    )
    assert events == []


def test_merge_adjacent_combines_close_events() -> None:
    events = [
        {"event_id": "a", "source_audio_path": "x", "label": "Tom",
         "start_time_s": 1.0, "end_time_s": 1.5,
         "freq_min_hz": 300, "freq_max_hz": 1500, "score": 0.8,
         "aru_id": "", "model_id": "", "model_version": "", "inference_id": ""},
        {"event_id": "b", "source_audio_path": "x", "label": "Tom",
         "start_time_s": 1.6, "end_time_s": 2.0,
         "freq_min_hz": 400, "freq_max_hz": 1800, "score": 0.7,
         "aru_id": "", "model_id": "", "model_version": "", "inference_id": ""},
    ]
    merged = _merge_adjacent(events, merge_gap_s=0.2)
    assert len(merged) == 1
    m = merged[0]
    assert m["start_time_s"] == 1.0
    assert m["end_time_s"] == 2.0
    assert m["freq_min_hz"] == 300
    assert m["freq_max_hz"] == 1800
    assert m["score"] == 0.8


def test_merge_adjacent_keeps_distinct_classes_separate() -> None:
    events = [
        {"event_id": "a", "source_audio_path": "x", "label": "Tom",
         "start_time_s": 1.0, "end_time_s": 1.5,
         "freq_min_hz": 300, "freq_max_hz": 1500, "score": 0.8,
         "aru_id": "", "model_id": "", "model_version": "", "inference_id": ""},
        {"event_id": "b", "source_audio_path": "x", "label": "Hen",
         "start_time_s": 1.5, "end_time_s": 2.0,
         "freq_min_hz": 600, "freq_max_hz": 2500, "score": 0.7,
         "aru_id": "", "model_id": "", "model_version": "", "inference_id": ""},
    ]
    merged = _merge_adjacent(events, merge_gap_s=1.0)
    assert len(merged) == 2


def test_make_event_id_deterministic() -> None:
    a = make_event_id("C:/audio/a.wav", 1.234, 2.567, "Tom")
    b = make_event_id("C:/audio/a.wav", 1.234, 2.567, "Tom")
    c = make_event_id("C:/audio/a.wav", 1.234, 2.567, "Hen")
    assert a == b
    assert a != c
    assert a.startswith("evt_")

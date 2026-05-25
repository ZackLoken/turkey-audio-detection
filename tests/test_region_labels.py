"""Tests for the canvas-region helpers and the new label-CSV schema."""

import json
import math

import pytest

from turkey_audio_detection.app import (
    CANVAS_FMAX_HZ,
    CANVAS_FMIN_HZ,
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    STROKE_COLOR_HEN,
    STROKE_COLOR_TOM,
    _parse_regions,
    derive_presence,
    hz_to_pixel_y,
    pixel_y_to_hz,
    rect_to_region,
    region_to_rect,
)


def test_pixel_y_to_hz_endpoints() -> None:
    # py=0 is the top of the canvas → high frequency end.
    assert pixel_y_to_hz(0, CANVAS_HEIGHT, CANVAS_FMIN_HZ, CANVAS_FMAX_HZ) == pytest.approx(CANVAS_FMAX_HZ)
    # py=canvas_h is the bottom → low frequency end.
    assert pixel_y_to_hz(CANVAS_HEIGHT, CANVAS_HEIGHT, CANVAS_FMIN_HZ, CANVAS_FMAX_HZ) == pytest.approx(
        CANVAS_FMIN_HZ
    )


def test_pixel_y_to_hz_clips_out_of_range() -> None:
    # Out-of-bounds py values are clamped so we never get NaNs into regions_json.
    high = pixel_y_to_hz(-10, CANVAS_HEIGHT, CANVAS_FMIN_HZ, CANVAS_FMAX_HZ)
    low = pixel_y_to_hz(CANVAS_HEIGHT + 10, CANVAS_HEIGHT, CANVAS_FMIN_HZ, CANVAS_FMAX_HZ)
    assert high == pytest.approx(CANVAS_FMAX_HZ)
    assert low == pytest.approx(CANVAS_FMIN_HZ)


def test_hz_to_pixel_y_round_trip() -> None:
    for hz in (300.0, 1200.0, 3500.0, 5500.0):
        py = hz_to_pixel_y(hz, CANVAS_HEIGHT, CANVAS_FMIN_HZ, CANVAS_FMAX_HZ)
        back = pixel_y_to_hz(py, CANVAS_HEIGHT, CANVAS_FMIN_HZ, CANVAS_FMAX_HZ)
        assert back == pytest.approx(hz, rel=1e-6)


def test_rect_to_region_basic_tom() -> None:
    obj = {
        "type": "rect",
        "left": CANVAS_WIDTH * 0.25,  # 0.75 s into a 3 s clip
        "top": 50.0,
        "width": CANVAS_WIDTH * 0.25,  # spans 0.75 s
        "height": 100.0,
        "stroke": STROKE_COLOR_TOM,
    }
    region = rect_to_region(
        obj, CANVAS_WIDTH, CANVAS_HEIGHT, 3.0, CANVAS_FMIN_HZ, CANVAS_FMAX_HZ, snap_freq=False
    )
    assert region is not None
    assert region["label"] == "Tom"
    assert region["start_s"] == pytest.approx(0.75, abs=1e-3)
    assert region["end_s"] == pytest.approx(1.5, abs=1e-3)
    assert CANVAS_FMIN_HZ < region["freq_min_hz"] < region["freq_max_hz"] < CANVAS_FMAX_HZ


def test_rect_to_region_hen_stroke_resolves_to_hen() -> None:
    obj = {
        "type": "rect",
        "left": 0.0,
        "top": 0.0,
        "width": 100.0,
        "height": 100.0,
        "stroke": STROKE_COLOR_HEN,
    }
    region = rect_to_region(
        obj, CANVAS_WIDTH, CANVAS_HEIGHT, 3.0, CANVAS_FMIN_HZ, CANVAS_FMAX_HZ, snap_freq=False
    )
    assert region is not None
    assert region["label"] == "Hen"


def test_rect_to_region_snap_freq_clamps_to_canvas_band() -> None:
    obj = {
        "type": "rect",
        "left": 0.0,
        "top": 100.0,
        "width": 200.0,
        "height": 50.0,
        "stroke": STROKE_COLOR_TOM,
    }
    region = rect_to_region(
        obj, CANVAS_WIDTH, CANVAS_HEIGHT, 3.0, CANVAS_FMIN_HZ, CANVAS_FMAX_HZ, snap_freq=True
    )
    assert region is not None
    assert region["freq_min_hz"] == pytest.approx(CANVAS_FMIN_HZ)
    assert region["freq_max_hz"] == pytest.approx(CANVAS_FMAX_HZ)


def test_rect_to_region_zero_size_drops() -> None:
    for w, h in [(0, 100), (100, 0), (0, 0)]:
        obj = {
            "type": "rect",
            "left": 100.0,
            "top": 100.0,
            "width": float(w),
            "height": float(h),
            "stroke": STROKE_COLOR_TOM,
        }
        assert rect_to_region(
            obj, CANVAS_WIDTH, CANVAS_HEIGHT, 3.0, CANVAS_FMIN_HZ, CANVAS_FMAX_HZ, snap_freq=False
        ) is None


def test_rect_to_region_clips_time_to_clip_duration() -> None:
    # Rect drawn past the right edge of the canvas should clip to clip_duration_s.
    obj = {
        "type": "rect",
        "left": CANVAS_WIDTH * 0.9,
        "top": 50.0,
        "width": CANVAS_WIDTH * 0.5,  # extends well past canvas
        "height": 100.0,
        "stroke": STROKE_COLOR_TOM,
    }
    region = rect_to_region(
        obj, CANVAS_WIDTH, CANVAS_HEIGHT, 3.0, CANVAS_FMIN_HZ, CANVAS_FMAX_HZ, snap_freq=False
    )
    assert region is not None
    assert region["end_s"] == pytest.approx(3.0, abs=1e-3)


def test_region_round_trip_through_rect() -> None:
    original = {
        "start_s": 0.5,
        "end_s": 1.75,
        "freq_min_hz": 300.0,
        "freq_max_hz": 2500.0,
        "label": "Tom",
    }
    rect = region_to_rect(original, CANVAS_WIDTH, CANVAS_HEIGHT, 3.0, CANVAS_FMIN_HZ, CANVAS_FMAX_HZ)
    region = rect_to_region(
        rect, CANVAS_WIDTH, CANVAS_HEIGHT, 3.0, CANVAS_FMIN_HZ, CANVAS_FMAX_HZ, snap_freq=False
    )
    assert region is not None
    assert region["start_s"] == pytest.approx(original["start_s"], abs=1e-2)
    assert region["end_s"] == pytest.approx(original["end_s"], abs=1e-2)
    assert region["freq_min_hz"] == pytest.approx(original["freq_min_hz"], rel=1e-3)
    assert region["freq_max_hz"] == pytest.approx(original["freq_max_hz"], rel=1e-3)
    assert region["label"] == "Tom"


def test_derive_presence_combinations() -> None:
    assert derive_presence([]) == (0, 0)
    assert derive_presence([{"label": "Tom"}]) == (1, 0)
    assert derive_presence([{"label": "Hen"}]) == (0, 1)
    assert derive_presence([{"label": "Tom"}, {"label": "Hen"}]) == (1, 1)


def test_parse_regions_handles_empty_and_invalid() -> None:
    assert _parse_regions("") == []
    assert _parse_regions("nan") == []
    assert _parse_regions(float("nan")) == []
    assert _parse_regions(None) == []
    assert _parse_regions("not json") == []
    assert _parse_regions("[]") == []
    assert _parse_regions('[{"label":"Tom"}]') == [{"label": "Tom"}]


def test_regions_json_round_trip_via_json_module() -> None:
    regions = [
        {"start_s": 0.5, "end_s": 1.5, "freq_min_hz": 250.0, "freq_max_hz": 1500.0, "label": "Tom"},
        {"start_s": 2.0, "end_s": 2.7, "freq_min_hz": 600.0, "freq_max_hz": 2500.0, "label": "Hen"},
    ]
    serialized = json.dumps(regions, separators=(",", ":"))
    restored = _parse_regions(serialized)
    assert restored == regions

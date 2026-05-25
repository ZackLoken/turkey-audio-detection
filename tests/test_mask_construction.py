"""Tests for regions_to_mask + seconds/Hz conversion helpers in dataset.py."""

import numpy as np
import pytest

from turkey_audio_detection.dataset import (
    CLASS_INDEX,
    MelParams,
    N_CLASSES,
    hz_to_mel_bin,
    parse_regions,
    regions_to_mask,
    seconds_to_frame,
)


P = MelParams()


def test_seconds_to_frame_basic() -> None:
    # At PANNs SR/hop, 1 second = 32000/320 = 100 frames.
    assert seconds_to_frame(1.0, P) == 100
    assert seconds_to_frame(0.0, P) == 0
    assert seconds_to_frame(3.0, P) == 300


def test_hz_to_mel_bin_endpoints() -> None:
    # fmin maps to bin 0, fmax to the last bin.
    assert hz_to_mel_bin(P.fmin, P) == 0
    assert hz_to_mel_bin(P.fmax, P) == P.n_mels - 1


def test_hz_to_mel_bin_clips_out_of_range() -> None:
    # Below fmin → bin 0; above fmax → last bin.
    assert hz_to_mel_bin(0.0, P) == 0
    assert hz_to_mel_bin(50_000.0, P) == P.n_mels - 1


def test_regions_to_mask_empty() -> None:
    mask = regions_to_mask([], n_frames=300)
    assert mask.shape == (N_CLASSES, P.n_mels, 300)
    assert mask.sum() == 0.0


def test_regions_to_mask_single_tom() -> None:
    region = {
        "start_s": 1.0,
        "end_s": 2.0,
        "freq_min_hz": 300.0,
        "freq_max_hz": 1500.0,
        "label": "Tom",
    }
    mask = regions_to_mask([region], n_frames=300)
    assert mask.shape == (N_CLASSES, P.n_mels, 300)
    # Tom channel: must contain ones within the time slice.
    tom_idx = CLASS_INDEX["Tom"]
    assert mask[tom_idx, :, 100:200].sum() > 0
    # Hen channel must remain zero.
    hen_idx = CLASS_INDEX["Hen"]
    assert mask[hen_idx].sum() == 0.0


def test_regions_to_mask_overlapping_classes() -> None:
    regions = [
        {"start_s": 0.0, "end_s": 1.0, "freq_min_hz": 300, "freq_max_hz": 1500, "label": "Tom"},
        {"start_s": 0.5, "end_s": 1.5, "freq_min_hz": 600, "freq_max_hz": 2500, "label": "Hen"},
    ]
    mask = regions_to_mask(regions, n_frames=300)
    tom = CLASS_INDEX["Tom"]
    hen = CLASS_INDEX["Hen"]
    # Each class has its own region; both should be active in their respective channels.
    assert mask[tom].sum() > 0
    assert mask[hen].sum() > 0


def test_regions_to_mask_freq_full_band_snap() -> None:
    """Snap-to-full-band regions populate the full mel range."""
    region = {
        "start_s": 0.0,
        "end_s": 3.0,
        "freq_min_hz": P.fmin,
        "freq_max_hz": P.fmax,
        "label": "Tom",
    }
    mask = regions_to_mask([region], n_frames=300)
    tom = CLASS_INDEX["Tom"]
    # Every mel bin must have at least one positive frame.
    assert (mask[tom].sum(axis=1) > 0).all()


def test_regions_to_mask_invalid_dropped() -> None:
    regions = [
        {"start_s": 1.0, "end_s": 0.5, "freq_min_hz": 300, "freq_max_hz": 1500, "label": "Tom"},  # bad time
        {"start_s": 0.0, "end_s": 1.0, "freq_min_hz": 1500, "freq_max_hz": 1500, "label": "Tom"},  # zero band
        {"start_s": 0.0, "end_s": 1.0, "freq_min_hz": 300, "freq_max_hz": 1500, "label": "Unknown"},  # bad class
    ]
    mask = regions_to_mask(regions, n_frames=300)
    assert mask.sum() == 0.0


def test_parse_regions_round_trip() -> None:
    import json

    regions = [{"start_s": 1.0, "end_s": 2.0, "freq_min_hz": 300, "freq_max_hz": 1500, "label": "Tom"}]
    text = json.dumps(regions)
    assert parse_regions(text) == regions
    assert parse_regions("") == []
    assert parse_regions("nan") == []
    assert parse_regions(float("nan")) == []
    assert parse_regions(None) == []
    assert parse_regions("not json") == []

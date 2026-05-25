"""Sanity check for the axis-free canvas spectrogram render."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def _make_wav(path: Path, duration_s: float = 3.0, sr: int = 48000) -> None:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    # Mix of three tones so the mel spectrogram has actual content in the canvas band.
    data = (
        0.2 * np.sin(2 * np.pi * 400 * t)
        + 0.2 * np.sin(2 * np.pi * 1200 * t)
        + 0.2 * np.sin(2 * np.pi * 3500 * t)
    ).astype("float32")
    sf.write(str(path), data, sr)


def test_spectrogram_for_canvas_returns_exact_dimensions(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    from turkey_audio_detection.app import _spectrogram_for_canvas, CANVAS_HEIGHT, CANVAS_WIDTH

    wav = tmp_path / "test.wav"
    _make_wav(wav)

    img = _spectrogram_for_canvas(str(wav), CANVAS_WIDTH, CANVAS_HEIGHT)
    assert img is not None
    assert img.size == (CANVAS_WIDTH, CANVAS_HEIGHT)
    assert img.mode == "RGB"


def test_spectrogram_for_canvas_returns_none_for_missing_file(tmp_path: Path) -> None:
    from turkey_audio_detection.app import _spectrogram_for_canvas, CANVAS_HEIGHT, CANVAS_WIDTH

    img = _spectrogram_for_canvas(str(tmp_path / "does_not_exist.wav"), CANVAS_WIDTH, CANVAS_HEIGHT)
    assert img is None

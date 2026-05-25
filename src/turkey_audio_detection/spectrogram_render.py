"""Spectrogram rendering shared between the review app and the cache stage.

Pulled out of app.py so the cache CLI can render PNGs without pulling in the
streamlit / drawable-canvas import graph.
"""

from __future__ import annotations

import io
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image


# Spectrogram band shown in the reviewer canvas.
# 50 Hz – 14 kHz covers the full audible range birds use (Wild Turkey calls dominate
# 150–3000 Hz; songbirds + insects sit 4–10 kHz). Matches PANNs CNN14's training band,
# so downstream model training reuses these mel filters without re-extraction.
CANVAS_FMIN_HZ = 50.0
CANVAS_FMAX_HZ = 14000.0
CANVAS_SR = 48000
CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 320
N_MELS_CANVAS = 128
N_FFT_CANVAS = 2048
HOP_LENGTH_CANVAS = 512

# Data area inside the rendered figure, as fractions of width/height.
# Leaves room on the left for frequency axis labels and on the bottom for time
# axis labels. The drawable-canvas widget uses these fractions when converting
# pixel coordinates of drawn rectangles into (time, Hz) regions.
DATA_LEFT_FRAC = 0.055
DATA_BOTTOM_FRAC = 0.13
DATA_RIGHT_FRAC = 1.0
DATA_TOP_FRAC = 1.0

_BG = "#0e1117"  # matches Streamlit dark theme background


def data_area_bounds(canvas_w: int, canvas_h: int) -> tuple[float, float, float, float]:
    """Pixel bounds (left, top, right, bottom) of the spectrogram data area inside
    a (canvas_w, canvas_h) image. Anything outside this area is axis chrome."""
    return (
        canvas_w * DATA_LEFT_FRAC,
        canvas_h * (1.0 - DATA_TOP_FRAC),
        canvas_w * DATA_RIGHT_FRAC,
        canvas_h * (1.0 - DATA_BOTTOM_FRAC),
    )


def render_canvas_spectrogram(
    audio_path: str | Path,
    width_px: int = CANVAS_WIDTH,
    height_px: int = CANVAS_HEIGHT,
) -> Image.Image | None:
    """Mel spectrogram with visible time + frequency axes."""
    try:
        y, sr = librosa.load(str(audio_path), sr=CANVAS_SR, mono=True)
    except Exception:
        return None
    if y.size == 0:
        return None

    melspec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT_CANVAS,
        hop_length=HOP_LENGTH_CANVAS,
        n_mels=N_MELS_CANVAS,
        fmin=CANVAS_FMIN_HZ,
        fmax=CANVAS_FMAX_HZ,
    )
    db = librosa.power_to_db(melspec, ref=max(1e-6, float(melspec.max())))

    dpi = 100
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi, facecolor=_BG)
    ax = fig.add_axes(
        [
            DATA_LEFT_FRAC,
            DATA_BOTTOM_FRAC,
            DATA_RIGHT_FRAC - DATA_LEFT_FRAC,
            DATA_TOP_FRAC - DATA_BOTTOM_FRAC,
        ]
    )
    ax.set_facecolor(_BG)
    librosa.display.specshow(
        db,
        sr=sr,
        hop_length=HOP_LENGTH_CANVAS,
        x_axis="time",
        y_axis="mel",
        ax=ax,
        fmin=CANVAS_FMIN_HZ,
        fmax=CANVAS_FMAX_HZ,
    )
    ax.tick_params(colors="#cccccc", labelsize=9, length=3)
    ax.set_xlabel("Time (s)", color="#cccccc", fontsize=9, labelpad=2)
    ax.set_ylabel("Frequency (Hz)", color="#cccccc", fontsize=9, labelpad=2)
    for spine in ax.spines.values():
        spine.set_edgecolor("#666666")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", pad_inches=0, facecolor=_BG)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB").resize((width_px, height_px))


def save_canvas_spectrogram(
    audio_path: str | Path,
    output_path: Path,
    width_px: int = CANVAS_WIDTH,
    height_px: int = CANVAS_HEIGHT,
) -> bool:
    """Render and save the canvas-band spectrogram to a PNG file."""
    img = render_canvas_spectrogram(audio_path, width_px, height_px)
    if img is None:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG")
    return True

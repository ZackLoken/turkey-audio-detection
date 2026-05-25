"""PyTorch dataset + mask construction for region-level SED training.

The training target is a 2D mask of shape (n_classes, n_mels, n_frames) where
mask[c, m, t] = 1 if mel-bin m and frame t fall inside any region of class c.
Region coordinates in regions_json are in (seconds, Hz) and are converted to
(frame, mel-bin) indices using PANNs CNN14's mel-filterbank parameters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


# PANNs CNN14 expected input parameters
PANNS_SR = 32000
PANNS_N_FFT = 1024
PANNS_HOP = 320
PANNS_N_MELS = 64
PANNS_FMIN = 50.0
PANNS_FMAX = 14000.0

CLASS_INDEX = {"Tom": 0, "Hen": 1}
N_CLASSES = len(CLASS_INDEX)


@dataclass(frozen=True)
class MelParams:
    sample_rate: int = PANNS_SR
    n_fft: int = PANNS_N_FFT
    hop_length: int = PANNS_HOP
    n_mels: int = PANNS_N_MELS
    fmin: float = PANNS_FMIN
    fmax: float = PANNS_FMAX


def _mel_band_centers_hz(p: MelParams) -> np.ndarray:
    return librosa.mel_frequencies(n_mels=p.n_mels, fmin=p.fmin, fmax=p.fmax)


def hz_to_mel_bin(hz: float, p: MelParams, centers: np.ndarray | None = None) -> int:
    """Return the closest mel-bin index for the given frequency in Hz."""
    if centers is None:
        centers = _mel_band_centers_hz(p)
    hz_clipped = float(np.clip(hz, p.fmin, p.fmax))
    return int(np.argmin(np.abs(centers - hz_clipped)))


def seconds_to_frame(s: float, p: MelParams) -> int:
    """Convert seconds to a STFT-frame index (using PANNs' hop_length)."""
    return int(round(s * p.sample_rate / p.hop_length))


def regions_to_mask(
    regions: list[dict],
    n_frames: int,
    p: MelParams = MelParams(),
) -> np.ndarray:
    """Build a (n_classes, n_mels, n_frames) binary mask from a region list."""
    mask = np.zeros((N_CLASSES, p.n_mels, n_frames), dtype=np.float32)
    if not regions:
        return mask

    centers = _mel_band_centers_hz(p)
    for region in regions:
        label = str(region.get("label", ""))
        if label not in CLASS_INDEX:
            continue
        c = CLASS_INDEX[label]

        start_s = float(region.get("start_s", 0.0))
        end_s = float(region.get("end_s", 0.0))
        if end_s <= start_s:
            continue
        t0 = max(0, seconds_to_frame(start_s, p))
        t1 = min(n_frames, seconds_to_frame(end_s, p))
        if t1 <= t0:
            continue

        f_min = float(region.get("freq_min_hz", p.fmin))
        f_max = float(region.get("freq_max_hz", p.fmax))
        if f_max <= f_min:
            continue
        m0 = hz_to_mel_bin(f_min, p, centers)
        m1 = hz_to_mel_bin(f_max, p, centers)
        if m1 < m0:
            m0, m1 = m1, m0
        m1 = min(p.n_mels - 1, m1)
        m0 = max(0, m0)

        mask[c, m0 : m1 + 1, t0:t1] = 1.0

    return mask


def parse_regions(regions_json: object) -> list[dict]:
    if regions_json is None:
        return []
    if isinstance(regions_json, float) and np.isnan(regions_json):
        return []
    text = str(regions_json).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [r for r in parsed if isinstance(r, dict)]


def load_log_mel(
    audio_path: str | Path,
    target_duration_s: float,
    p: MelParams = MelParams(),
) -> tuple[np.ndarray, int]:
    """Load WAV, resample to PANNs SR, pad/truncate, return (log_mel, n_frames)."""
    y, sr = librosa.load(str(audio_path), sr=p.sample_rate, mono=True)
    n_target = int(round(target_duration_s * p.sample_rate))
    if y.size < n_target:
        y = np.concatenate([y, np.zeros(n_target - y.size, dtype=y.dtype)])
    else:
        y = y[:n_target]
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=p.sample_rate,
        n_fft=p.n_fft,
        hop_length=p.hop_length,
        n_mels=p.n_mels,
        fmin=p.fmin,
        fmax=p.fmax,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=1.0, top_db=80.0).astype(np.float32)
    n_frames = log_mel.shape[1]
    return log_mel, n_frames


class TurkeyClipDataset(Dataset):
    """One sample = one labeled clip. Returns (log_mel, mask_2d, weak_label, item_id).

    log_mel: (n_mels, n_frames) float32
    mask_2d: (n_classes, n_mels, n_frames) float32
    weak_label: (n_classes,) float32 with clip-level presence booleans
    item_id: str
    """

    def __init__(
        self,
        table: "pd.DataFrame",  # noqa: F821 — pandas typed by user at call site
        clip_duration_s: float = 3.0,
        mel: MelParams = MelParams(),
        augmentations: Iterable[Callable] = (),
    ) -> None:
        self.table = table.reset_index(drop=True)
        self.clip_duration_s = float(clip_duration_s)
        self.mel = mel
        self.augmentations = tuple(augmentations)

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int):
        row = self.table.iloc[idx]
        clip_path = str(row["clip_path"])
        log_mel, n_frames = load_log_mel(clip_path, self.clip_duration_s, self.mel)

        regions = parse_regions(row.get("regions_json", ""))
        mask = regions_to_mask(regions, n_frames=n_frames, p=self.mel)

        weak = np.array(
            [float(row.get("tom_present", 0) or 0), float(row.get("hen_present", 0) or 0)],
            dtype=np.float32,
        )

        for aug in self.augmentations:
            log_mel, mask, weak = aug(log_mel, mask, weak)

        return (
            torch.from_numpy(log_mel),
            torch.from_numpy(mask),
            torch.from_numpy(weak),
            str(row.get("item_id", "")),
        )

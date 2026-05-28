"""Frame-level SED data pipeline (single-stage ConvNeXt-BirdSet model).

One training sample = one labeled 3 s candidate clip ->
  (log_mel (n_mels, T), frame_target (n_classes, T), weak (n_classes,), item_id)

Targets are TIME-ONLY: each reviewer box (one call phrase = one event) is projected
onto the time axis; frequency extent is dropped. Mel features match
DBD-research-group/ConvNeXT-Base-BirdSet-XCL so the pretrained weights stay valid:
torchaudio Spectrogram(power=2) + MelScale(128) + PowerToDB(top_db=80), then a
per-sample normalize (mean=-4.268, std=4.569) applied AFTER augmentation.

This module is additive; the legacy 2D-mask path in `dataset.py` is retired later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from turkey_audio_detection.dataset import CLASS_INDEX, N_CLASSES, parse_regions


# Mel config of DBD-research-group/ConvNeXT-Base-BirdSet-XCL (must match to reuse weights).
SED_SR = 32000
SED_N_FFT = 1024
SED_HOP = 320              # 32000/320 = 100 frames/s -> 10 ms/frame at the mel
SED_N_MELS = 128
SED_FMIN = 0.0             # torchaudio MelScale defaults (BirdSet did not override)
SED_FMAX = 16000.0         # sample_rate / 2
SED_TOP_DB = 80.0
SED_NORM_MEAN = -4.268
SED_NORM_STD = 4.569
SED_AMIN = 1e-10


@dataclass(frozen=True)
class SedMelParams:
    sample_rate: int = SED_SR
    n_fft: int = SED_N_FFT
    hop_length: int = SED_HOP
    n_mels: int = SED_N_MELS
    fmin: float = SED_FMIN
    fmax: float = SED_FMAX
    top_db: float = SED_TOP_DB
    norm_mean: float = SED_NORM_MEAN
    norm_std: float = SED_NORM_STD
    amin: float = SED_AMIN


class LogMelExtractor(torch.nn.Module):
    """Waveform -> dB log-mel (n_mels, T), matched to BirdSet's PowerToDB.

    Normalization is intentionally NOT applied here (see `normalize_log_mel`) so
    augmentation can run in the dB domain before standardization.
    """

    def __init__(self, p: SedMelParams = SedMelParams()) -> None:
        super().__init__()
        self.p = p
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=p.n_fft, hop_length=p.hop_length, power=2.0
        )
        self.melscale = torchaudio.transforms.MelScale(
            n_mels=p.n_mels, sample_rate=p.sample_rate, n_stft=p.n_fft // 2 + 1,
            f_min=p.fmin, f_max=p.fmax,
        )

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:  # (..., samples) -> (..., n_mels, T)
        power = self.spec(waveform)
        mel = self.melscale(power)
        db = 10.0 * torch.log10(torch.clamp(mel, min=self.p.amin))
        db = torch.maximum(db, db.amax(dim=(-2, -1), keepdim=True) - self.p.top_db)
        return db


def normalize_log_mel(log_mel_db: np.ndarray, p: SedMelParams = SedMelParams()) -> np.ndarray:
    """Standardize dB log-mel with BirdSet's ESC-50 mean/std."""
    return ((log_mel_db - p.norm_mean) / p.norm_std).astype(np.float32)


def load_waveform(audio_path: str | Path, target_duration_s: float, sample_rate: int = SED_SR) -> np.ndarray:
    """Load mono WAV at `sample_rate`, pad/truncate to `target_duration_s`."""
    y, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    n = int(round(target_duration_s * sample_rate))
    if y.size < n:
        y = np.concatenate([y, np.zeros(n - y.size, dtype=y.dtype)])
    else:
        y = y[:n]
    return y.astype(np.float32)


def regions_to_frame_targets(
    regions: list[dict], n_frames: int, p: SedMelParams = SedMelParams()
) -> np.ndarray:
    """Project reviewer boxes onto a (N_CLASSES, n_frames) time-only binary target.

    Each box's [start_s, end_s] becomes 1 across the covered frames of its class.
    Frequency extent is ignored. Empty regions -> all-zero (hard negative).
    """
    target = np.zeros((N_CLASSES, n_frames), dtype=np.float32)
    for r in regions:
        label = str(r.get("label", ""))
        if label not in CLASS_INDEX:
            continue
        c = CLASS_INDEX[label]
        start_s = float(r.get("start_s", 0.0))
        end_s = float(r.get("end_s", 0.0))
        if end_s <= start_s:
            continue
        t0 = max(0, int(round(start_s * p.sample_rate / p.hop_length)))
        t1 = min(n_frames, int(round(end_s * p.sample_rate / p.hop_length)))
        if t1 <= t0:
            continue
        target[c, t0:t1] = 1.0
    return target


def downsample_targets(target: np.ndarray, t_out: int) -> np.ndarray:
    """Max-pool a (C, T) binary target along time to (C, t_out) to match model output.

    Max-pool preserves presence: a pooled frame is positive if any source frame is.
    """
    if target.shape[-1] == t_out:
        return target.astype(np.float32)
    t = torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32)).unsqueeze(0)  # (1,C,T)
    pooled = F.adaptive_max_pool1d(t, t_out)
    return pooled.squeeze(0).numpy()


class FrameSedDataset(Dataset):
    """Labeled clips -> (log_mel (n_mels,T), frame_target (C,T), weak (C,), item_id).

    log_mel is normalized dB (augmentations applied in the dB domain first).
    Augmentations are callables (log_mel_db, target, weak) -> (log_mel_db, target, weak).
    """

    def __init__(
        self,
        table: pd.DataFrame,
        clip_duration_s: float = 3.0,
        mel: SedMelParams = SedMelParams(),
        augmentations: Iterable[Callable] = (),
    ) -> None:
        self.table = table.reset_index(drop=True)
        self.clip_duration_s = float(clip_duration_s)
        self.p = mel
        self.extractor = LogMelExtractor(mel)
        self.augmentations = tuple(augmentations)

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int):
        row = self.table.iloc[idx]
        wav = load_waveform(str(row["clip_path"]), self.clip_duration_s, self.p.sample_rate)
        log_mel = self.extractor(torch.from_numpy(wav)).numpy()  # (n_mels, T) dB
        n_frames = log_mel.shape[1]

        regions = parse_regions(row.get("regions_json", ""))
        target = regions_to_frame_targets(regions, n_frames=n_frames, p=self.p)  # (C, T)
        weak = np.array(
            [float(row.get("tom_present", 0) or 0), float(row.get("hen_present", 0) or 0)],
            dtype=np.float32,
        )

        for aug in self.augmentations:
            log_mel, target, weak = aug(log_mel, target, weak)

        log_mel = normalize_log_mel(log_mel, self.p)
        return (
            torch.from_numpy(log_mel),
            torch.from_numpy(target.astype(np.float32)),
            torch.from_numpy(weak),
            str(row.get("item_id", "")),
        )

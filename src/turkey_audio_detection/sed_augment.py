"""Frame-level SED augmentations: operate on (log_mel_db, target(C,T), weak(C,)).

Each augmentation is a callable
  (log_mel_db (n_mels,T), target (C,T), weak (C,)) -> same shapes,
applied in the dB domain before normalization. Pure-functional + unit-testable.

Mixup and BackgroundMix blend in LINEAR POWER (the legacy `augment.py` Mixup
blended in dB, which is acoustically wrong; fixed here).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_AMIN = 1e-10


@dataclass
class SpecAugment:
    """Mask random time/frequency strips. Time masks also zero the (C,T) target."""

    time_mask_width: int = 30
    n_time_masks: int = 2
    freq_mask_width: int = 12
    n_freq_masks: int = 2
    fill_value: float | None = None  # None -> per-call minimum (treat masked as silence)
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def __call__(self, log_mel: np.ndarray, target: np.ndarray, weak: np.ndarray):
        n_mels, n_frames = log_mel.shape
        out = log_mel.copy()
        tgt = target.copy()
        fill = float(out.min()) if self.fill_value is None else float(self.fill_value)
        rng = self.rng

        for _ in range(self.n_time_masks):
            width = int(rng.integers(0, self.time_mask_width + 1))
            if width == 0 or width >= n_frames:
                continue
            start = int(rng.integers(0, n_frames - width))
            out[:, start : start + width] = fill
            tgt[:, start : start + width] = 0.0  # supervision removed where masked

        for _ in range(self.n_freq_masks):
            width = int(rng.integers(0, self.freq_mask_width + 1))
            if width == 0 or width >= n_mels:
                continue
            start = int(rng.integers(0, n_mels - width))
            out[start : start + width, :] = fill  # target has no frequency axis

        return out, tgt, weak


@dataclass
class Mixup:
    """Blend two samples in linear power. Provide the partner via set_partner()."""

    alpha: float = 0.4
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def __post_init__(self) -> None:
        self._partner: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def set_partner(self, partner: tuple[np.ndarray, np.ndarray, np.ndarray] | None) -> None:
        self._partner = partner

    def __call__(self, log_mel: np.ndarray, target: np.ndarray, weak: np.ndarray):
        if self._partner is None or self.alpha <= 0:
            return log_mel, target, weak
        p_mel, p_tgt, p_weak = self._partner
        if p_mel.shape != log_mel.shape or p_tgt.shape != target.shape:
            return log_mel, target, weak
        lam = float(self.rng.beta(self.alpha, self.alpha))
        a_lin = np.power(10.0, log_mel / 10.0)
        b_lin = np.power(10.0, p_mel / 10.0)
        out = 10.0 * np.log10(np.maximum(lam * a_lin + (1.0 - lam) * b_lin, _AMIN))
        # Targets/weak are unions so supervision stays a valid {0,1} presence label.
        return (
            out.astype(np.float32),
            np.maximum(target, p_tgt).astype(np.float32),
            np.maximum(weak, p_weak).astype(np.float32),
        )


@dataclass
class BackgroundMix:
    """Add a scaled background dB log-mel at a random SNR (linear-power domain)."""

    snr_db_range: tuple[float, float] = (5.0, 25.0)
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def __post_init__(self) -> None:
        self._background: np.ndarray | None = None

    def set_background(self, background_log_mel: np.ndarray | None) -> None:
        self._background = background_log_mel

    def __call__(self, log_mel: np.ndarray, target: np.ndarray, weak: np.ndarray):
        if self._background is None or self._background.shape != log_mel.shape:
            return log_mel, target, weak
        snr_db = float(self.rng.uniform(*self.snr_db_range))
        fg_lin = np.power(10.0, log_mel / 10.0)
        bg_lin = np.power(10.0, (self._background - snr_db) / 10.0)
        out = 10.0 * np.log10(np.maximum(fg_lin + bg_lin, _AMIN))
        return out.astype(np.float32), target, weak

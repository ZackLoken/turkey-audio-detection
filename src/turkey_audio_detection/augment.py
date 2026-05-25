"""Training-time augmentations for region-level SED.

Each augmentation is a callable (log_mel, mask, weak) -> (log_mel, mask, weak),
applied in sequence inside the dataset. Designed to be pure-functional and
unit-testable in isolation (no torch modules, no global state).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SpecAugment:
    """Mask random time and frequency strips on the log-mel; mirror on the target mask."""

    time_mask_width: int = 30
    n_time_masks: int = 2
    freq_mask_width: int = 12
    n_freq_masks: int = 2
    fill_value: float = -80.0  # roughly the floor of librosa log-mel in dB
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def __call__(self, log_mel: np.ndarray, mask: np.ndarray, weak: np.ndarray):
        n_mels, n_frames = log_mel.shape
        out = log_mel.copy()
        m = mask.copy()
        rng = self.rng

        for _ in range(self.n_time_masks):
            width = int(rng.integers(0, self.time_mask_width + 1))
            if width == 0 or width >= n_frames:
                continue
            start = int(rng.integers(0, n_frames - width))
            out[:, start : start + width] = self.fill_value
            m[:, :, start : start + width] = 0.0

        for _ in range(self.n_freq_masks):
            width = int(rng.integers(0, self.freq_mask_width + 1))
            if width == 0 or width >= n_mels:
                continue
            start = int(rng.integers(0, n_mels - width))
            out[start : start + width, :] = self.fill_value
            m[:, start : start + width, :] = 0.0

        return out, m, weak


@dataclass
class Mixup:
    """Linearly blend two samples. Call with the partner sample provided via .set_partner()."""

    alpha: float = 0.4
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def __post_init__(self) -> None:
        self._partner: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def set_partner(self, partner: tuple[np.ndarray, np.ndarray, np.ndarray] | None) -> None:
        self._partner = partner

    def __call__(self, log_mel: np.ndarray, mask: np.ndarray, weak: np.ndarray):
        if self._partner is None or self.alpha <= 0:
            return log_mel, mask, weak
        p_mel, p_mask, p_weak = self._partner
        if p_mel.shape != log_mel.shape:
            return log_mel, mask, weak
        lam = float(self.rng.beta(self.alpha, self.alpha))
        # Use max for masks/weak so the supervision remains a valid {0,1} target;
        # use a convex combination for the input spectrogram.
        out_mel = lam * log_mel + (1.0 - lam) * p_mel
        out_mask = np.maximum(mask, p_mask)
        out_weak = np.maximum(weak, p_weak)
        return out_mel.astype(np.float32), out_mask.astype(np.float32), out_weak.astype(np.float32)


@dataclass
class BackgroundMix:
    """Add a scaled background log-mel sample (no mask change)."""

    snr_db_range: tuple[float, float] = (5.0, 25.0)
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def __post_init__(self) -> None:
        self._background: np.ndarray | None = None

    def set_background(self, background_log_mel: np.ndarray) -> None:
        self._background = background_log_mel

    def __call__(self, log_mel: np.ndarray, mask: np.ndarray, weak: np.ndarray):
        if self._background is None or self._background.shape != log_mel.shape:
            return log_mel, mask, weak
        snr_db = float(self.rng.uniform(*self.snr_db_range))
        # Treat log-mel as already in dB: adding a noise sample at SNR_db below the
        # foreground means the background contributes at -snr_db relative to fg.
        # In dB: out = 10 * log10(10^(fg/10) + 10^(bg/10 - snr_db/10))
        fg_lin = np.power(10.0, log_mel / 10.0)
        bg_lin = np.power(10.0, (self._background - snr_db) / 10.0)
        out = 10.0 * np.log10(np.maximum(fg_lin + bg_lin, 1e-12))
        return out.astype(np.float32), mask, weak

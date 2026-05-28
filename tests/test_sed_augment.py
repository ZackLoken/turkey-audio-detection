"""Tests for frame-level SED augmentations (sed_augment.py)."""

from __future__ import annotations

import numpy as np

from turkey_audio_detection.sed_augment import BackgroundMix, Mixup, SpecAugment

N_MELS, N_FRAMES, N_CLASSES = 128, 300, 2


def _sample(seed: int = 0):
    rng = np.random.default_rng(seed)
    log_mel = rng.normal(-30.0, 10.0, size=(N_MELS, N_FRAMES)).astype(np.float32)
    target = np.zeros((N_CLASSES, N_FRAMES), dtype=np.float32)
    target[0, 100:200] = 1.0
    weak = np.array([1.0, 0.0], dtype=np.float32)
    return log_mel, target, weak


def test_specaugment_preserves_shapes_and_only_removes_target() -> None:
    log_mel, target, weak = _sample()
    aug = SpecAugment(rng=np.random.default_rng(3))
    out_mel, out_tgt, out_weak = aug(log_mel, target, weak)
    assert out_mel.shape == (N_MELS, N_FRAMES)
    assert out_tgt.shape == (N_CLASSES, N_FRAMES)
    assert set(np.unique(out_tgt)).issubset({0.0, 1.0})
    # time masking can only remove supervision, never add it
    assert out_tgt.sum() <= target.sum()
    assert np.isfinite(out_mel).all()


def test_mixup_linear_power_blend_and_union_targets() -> None:
    log_mel, target, weak = _sample(0)
    p_mel = np.full((N_MELS, N_FRAMES), -10.0, dtype=np.float32)
    p_tgt = np.zeros((N_CLASSES, N_FRAMES), dtype=np.float32)
    p_tgt[1, 50:150] = 1.0
    p_weak = np.array([0.0, 1.0], dtype=np.float32)

    mix = Mixup(alpha=0.4, rng=np.random.default_rng(7))
    mix.set_partner((p_mel, p_tgt, p_weak))
    out_mel, out_tgt, out_weak = mix(log_mel, target, weak)

    # Reproduce lam from an identically-seeded generator to verify the linear-power math.
    lam = float(np.random.default_rng(7).beta(0.4, 0.4))
    expected = 10.0 * np.log10(
        np.maximum(lam * 10 ** (log_mel / 10.0) + (1 - lam) * 10 ** (p_mel / 10.0), 1e-10)
    )
    assert np.allclose(out_mel, expected, atol=1e-4)
    # targets/weak are unions
    assert out_tgt[0, 100:200].sum() == 100.0
    assert out_tgt[1, 50:150].sum() == 100.0
    assert tuple(out_weak.tolist()) == (1.0, 1.0)


def test_mixup_noop_without_partner() -> None:
    log_mel, target, weak = _sample()
    mix = Mixup(alpha=0.4)
    out_mel, out_tgt, out_weak = mix(log_mel, target, weak)
    assert np.array_equal(out_mel, log_mel)
    assert np.array_equal(out_tgt, target)


def test_backgroundmix_adds_energy_and_keeps_target() -> None:
    log_mel, target, weak = _sample()
    bg = np.full((N_MELS, N_FRAMES), -20.0, dtype=np.float32)
    aug = BackgroundMix(snr_db_range=(10.0, 10.0), rng=np.random.default_rng(2))
    aug.set_background(bg)
    out_mel, out_tgt, out_weak = aug(log_mel, target, weak)
    assert out_mel.shape == (N_MELS, N_FRAMES)
    # adding background power can only raise the dB level
    assert (out_mel >= log_mel - 1e-4).all()
    assert np.array_equal(out_tgt, target)
    assert np.array_equal(out_weak, weak)

"""Tests for the training augmentations: SpecAugment, Mixup, BackgroundMix."""

import numpy as np
import pytest

from turkey_audio_detection.augment import BackgroundMix, Mixup, SpecAugment


N_MELS = 64
N_FRAMES = 300
N_CLASSES = 2


def _sample():
    rng = np.random.default_rng(0)
    log_mel = rng.normal(loc=-20.0, scale=5.0, size=(N_MELS, N_FRAMES)).astype(np.float32)
    mask = np.zeros((N_CLASSES, N_MELS, N_FRAMES), dtype=np.float32)
    mask[0, 20:40, 50:150] = 1.0  # Tom region
    weak = np.array([1.0, 0.0], dtype=np.float32)
    return log_mel, mask, weak


def test_specaugment_preserves_shape() -> None:
    log_mel, mask, weak = _sample()
    aug = SpecAugment(rng=np.random.default_rng(123))
    out_mel, out_mask, out_weak = aug(log_mel, mask, weak)
    assert out_mel.shape == log_mel.shape
    assert out_mask.shape == mask.shape
    assert out_weak.shape == weak.shape


def test_specaugment_zeros_out_mask_in_masked_strips() -> None:
    """Where the input log-mel is masked, the target mask is also zeroed to keep the
    supervision consistent with what the model can see."""
    log_mel, mask, weak = _sample()
    aug = SpecAugment(
        time_mask_width=N_FRAMES,
        n_time_masks=1,
        freq_mask_width=0,
        n_freq_masks=0,
        rng=np.random.default_rng(0),
    )
    out_mel, out_mask, _ = aug(log_mel, mask, weak)
    # Find the masked time band.
    masked_cols = (out_mel == aug.fill_value).all(axis=0)
    if masked_cols.any():
        # In those columns the mask must also be zero.
        assert (out_mask[:, :, masked_cols] == 0).all()


def test_mixup_blends_inputs_and_targets() -> None:
    log_mel, mask, weak = _sample()
    log_mel2, mask2, weak2 = _sample()
    # Build a partner with a different region so we can detect blending.
    mask2 = np.zeros_like(mask2)
    mask2[1, 40:60, 100:200] = 1.0
    weak2 = np.array([0.0, 1.0], dtype=np.float32)

    mixup = Mixup(alpha=0.4, rng=np.random.default_rng(0))
    mixup.set_partner((log_mel2, mask2, weak2))
    out_mel, out_mask, out_weak = mixup(log_mel, mask, weak)

    assert out_mel.shape == log_mel.shape
    # Both classes should be present in the mixed mask.
    assert out_mask[0].sum() > 0
    assert out_mask[1].sum() > 0
    # Weak label is union (max).
    assert np.allclose(out_weak, np.array([1.0, 1.0]))


def test_mixup_noop_without_partner() -> None:
    log_mel, mask, weak = _sample()
    mixup = Mixup(alpha=0.4)
    out_mel, out_mask, out_weak = mixup(log_mel, mask, weak)
    assert np.array_equal(out_mel, log_mel)
    assert np.array_equal(out_mask, mask)


def test_background_mix_modifies_log_mel_but_not_mask() -> None:
    log_mel, mask, weak = _sample()
    rng = np.random.default_rng(0)
    background = rng.normal(loc=-30.0, scale=5.0, size=log_mel.shape).astype(np.float32)

    bg = BackgroundMix(snr_db_range=(5.0, 5.0), rng=np.random.default_rng(0))
    bg.set_background(background)
    out_mel, out_mask, _ = bg(log_mel, mask, weak)

    assert out_mel.shape == log_mel.shape
    assert not np.array_equal(out_mel, log_mel)
    assert np.array_equal(out_mask, mask)


def test_background_mix_noop_without_background() -> None:
    log_mel, mask, weak = _sample()
    bg = BackgroundMix()
    out_mel, out_mask, _ = bg(log_mel, mask, weak)
    assert np.array_equal(out_mel, log_mel)

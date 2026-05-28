"""Tests for the FrameSed model + gradual-unfreeze hooks (offline, pretrained=False)."""

from __future__ import annotations

import numpy as np
import torch

from turkey_audio_detection.models.frame_sed import FrameSed
from turkey_audio_detection.sed_data import downsample_targets
from turkey_audio_detection.unfreeze import set_trainable


def _model(temporal: str = "bigru") -> FrameSed:
    # pretrained=False uses a small ConvNext config -> no network/download.
    return FrameSed(
        n_classes=2, n_stages=2, temporal=temporal,
        hidden_size=32, n_layers=1, pretrained=False,
    )


def test_forward_shapes_and_target_alignment() -> None:
    model = _model().eval()
    x = torch.randn(2, 128, 301)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape[0] == 2 and logits.shape[1] == 2
    t_prime = logits.shape[2]
    assert t_prime > 0
    # targets resample to the model's output stride
    tgt = np.zeros((2, 301), dtype=np.float32)
    tgt[0, 100:200] = 1.0
    ds = downsample_targets(tgt, t_prime)
    assert ds.shape == (2, t_prime)


def test_accepts_4d_input() -> None:
    model = _model().eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 1, 128, 301))
    assert logits.shape[0] == 1 and logits.shape[1] == 2


def test_backward_flows() -> None:
    model = _model()
    logits = model(torch.randn(2, 128, 301))
    logits.float().pow(2).mean().backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_tcn_head() -> None:
    model = _model(temporal="tcn").eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 128, 301))
    assert logits.shape[0] == 1 and logits.shape[1] == 2


def test_time_downsample_factor() -> None:
    model = _model()
    assert model.backbone.time_downsample == 8  # 4x stem * 2x (one stage after the first)


def test_gradual_unfreeze_groups() -> None:
    model = _model()
    groups = model.layer_groups()
    assert len(groups) == 5  # embeddings, stage0, stage1, temporal, classifier

    set_trainable(model, 2)  # head only (classifier + temporal)
    backbone_trainable = sum(
        p.numel() for _n, m in model.backbone.stage_groups() for p in m.parameters() if p.requires_grad
    )
    assert backbone_trainable == 0
    head_trainable = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
    assert head_trainable > 0

    set_trainable(model, len(groups))  # everything
    assert all(p.requires_grad for p in model.parameters())

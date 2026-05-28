"""Gradual-unfreezing schedule for FrameSed.

A schedule is a list of phases. Each phase unfreezes the top-`n_trainable_top_groups`
layer-groups (counting from the output/head side toward the input), and carries the
learning rate + batch size to use for that phase (the owner's preferred recipe:
progressively unfreeze deeper layers while scaling LR/batch, with early stopping).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn


@dataclass(frozen=True)
class Phase:
    n_trainable_top_groups: int  # groups unfrozen, counted from the output side
    lr: float
    batch_size: int
    max_epochs: int


@dataclass(frozen=True)
class UnfreezeSchedule:
    phases: list[Phase]

    def __len__(self) -> int:
        return len(self.phases)


def set_trainable(model: nn.Module, n_top_groups: int) -> int:
    """Unfreeze the top `n_top_groups` layer-groups; freeze the rest. Returns #trainable params."""
    groups = model.layer_groups()
    n = len(groups)
    k = max(0, min(int(n_top_groups), n))
    for i, (_name, module) in enumerate(groups):
        trainable = i >= (n - k)
        for p in module.parameters():
            p.requires_grad_(trainable)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def apply_phase(model: nn.Module, schedule: UnfreezeSchedule, phase_idx: int) -> Phase:
    phase = schedule.phases[phase_idx]
    set_trainable(model, phase.n_trainable_top_groups)
    return phase


def default_schedule(n_backbone_groups: int, base_lr: float = 1e-3, base_batch: int = 32) -> UnfreezeSchedule:
    """Head-first phases, then progressively unfreeze one backbone group at a time,
    halving LR and batch size each phase (scaled per the owner's recipe)."""
    phases = [Phase(n_trainable_top_groups=2, lr=base_lr, batch_size=base_batch, max_epochs=15)]
    lr, bs = base_lr, base_batch
    for g in range(1, n_backbone_groups + 1):
        lr = max(lr * 0.5, 1e-5)
        bs = max(bs // 2, 4)
        phases.append(Phase(n_trainable_top_groups=2 + g, lr=lr, batch_size=bs, max_epochs=12))
    return UnfreezeSchedule(phases=phases)

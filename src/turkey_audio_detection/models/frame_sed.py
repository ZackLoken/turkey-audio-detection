"""Single-stage frame-level SED model.

log-mel (B, n_mels, T) -> ConvNeXt-BirdSet frontend (mid-stage tap) -> pool
frequency -> temporal head (BiGRU default, TCN optional) -> per-frame, per-class
logits (B, n_classes, T'). T' is the frontend's downsampled time axis (~80 ms at
n_stages=2); training/inference resample targets to T' to match.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from turkey_audio_detection.models.backbone import DEFAULT_CHECKPOINT, ConvNextBackbone


class _TCN(nn.Module):
    """Dilated temporal conv stack (all-conv, export-friendly)."""

    def __init__(self, in_ch: int, hidden: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c = in_ch
        for i in range(max(1, n_layers)):
            d = 2 ** i
            layers += [
                nn.Conv1d(c, hidden, kernel_size=3, padding=d, dilation=d),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            c = hidden
        self.net = nn.Sequential(*layers)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T) -> (B, hidden, T)
        return self.net(x)


class FrameSed(nn.Module):
    def __init__(
        self,
        n_classes: int = 2,
        checkpoint: str = DEFAULT_CHECKPOINT,
        n_stages: int = 2,
        temporal: str = "bigru",
        hidden_size: int = 256,
        n_layers: int = 2,
        dropout: float = 0.2,
        pretrained: bool = True,
        config_dict: dict | None = None,
    ) -> None:
        super().__init__()
        self.backbone = ConvNextBackbone(
            checkpoint=checkpoint, n_stages=n_stages, pretrained=pretrained, config_dict=config_dict
        )
        c = self.backbone.out_channels
        self.temporal_kind = temporal
        if temporal == "bigru":
            self.temporal = nn.GRU(
                input_size=c, hidden_size=hidden_size, num_layers=n_layers,
                batch_first=True, bidirectional=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            head_in = hidden_size * 2
        elif temporal == "tcn":
            self.temporal = _TCN(c, hidden_size, n_layers, dropout)
            head_in = hidden_size
        else:
            raise ValueError(f"unknown temporal head: {temporal!r}")
        self.classifier = nn.Linear(head_in, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_mels, T) or (B, 1, n_mels, T) -> (B, n_classes, T')."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        feat = self.backbone(x)        # (B, C, F', T')
        feat = feat.mean(dim=2)        # collapse frequency -> (B, C, T')
        if self.temporal_kind == "bigru":
            seq = feat.transpose(1, 2)         # (B, T', C)
            seq, _ = self.temporal(seq)        # (B, T', 2*hidden)
            logits = self.classifier(seq)      # (B, T', n_classes)
        else:
            t = self.temporal(feat)            # (B, hidden, T')
            logits = self.classifier(t.transpose(1, 2))  # (B, T', n_classes)
        return logits.transpose(1, 2)          # (B, n_classes, T')

    def layer_groups(self) -> list[tuple[str, nn.Module]]:
        """Ordered (name, module) from input to output, for gradual unfreezing."""
        groups = list(self.backbone.stage_groups())
        groups.append(("temporal", self.temporal))
        groups.append(("classifier", self.classifier))
        return groups

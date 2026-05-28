"""BirdSet ConvNeXt frontend for frame-level SED.

Runs the BirdSet-pretrained ConvNeXt embeddings + the first `n_stages` stages and
returns a time-resolved feature map. Deeper stages are dropped to keep the model
light and the time resolution fine (tight onset/offset). With the BirdSet mel
(10 ms/frame), n_stages=2 gives ~80 ms/frame (8x downsample); n_stages=1 ~40 ms.
"""

from __future__ import annotations

import torch
import torch.nn as nn

DEFAULT_CHECKPOINT = "DBD-research-group/ConvNeXT-Base-BirdSet-XCL"


class ConvNextBackbone(nn.Module):
    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        n_stages: int = 2,
        pretrained: bool = True,
        config_dict: dict | None = None,
    ) -> None:
        super().__init__()
        from transformers import ConvNextConfig, ConvNextForImageClassification

        if pretrained:
            model = ConvNextForImageClassification.from_pretrained(
                checkpoint, num_channels=1, ignore_mismatched_sizes=True
            )
        elif config_dict is not None:
            # Rebuild the exact architecture used at training (random weights, no
            # download) so a fine-tuned state_dict loads cleanly at inference time.
            model = ConvNextForImageClassification(ConvNextConfig.from_dict(config_dict))
        else:
            model = ConvNextForImageClassification(ConvNextConfig(num_channels=1))

        cfg = model.config
        self.convnext_config = cfg.to_dict()  # persisted in checkpoints for faithful reload
        all_stages = list(model.convnext.encoder.stages)
        n_stages = max(1, min(int(n_stages), len(all_stages)))
        self.n_stages = n_stages
        self.embeddings = model.convnext.embeddings
        self.stages = nn.ModuleList(all_stages[:n_stages])  # drop deeper stages
        self.out_channels = int(cfg.hidden_sizes[n_stages - 1])
        # 4x patchify stem, then 2x at the start of each stage after the first.
        self.time_downsample = 4 * (2 ** (n_stages - 1))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """(B, 1, n_mels, T) -> (B, out_channels, F', T')."""
        x = self.embeddings(pixel_values)
        for stage in self.stages:
            x = stage(x)
        return x

    def stage_groups(self) -> list[tuple[str, nn.Module]]:
        """Ordered (name, module) from input to output, for gradual unfreezing."""
        groups: list[tuple[str, nn.Module]] = [("embeddings", self.embeddings)]
        for i, st in enumerate(self.stages):
            groups.append((f"stage{i}", st))
        return groups

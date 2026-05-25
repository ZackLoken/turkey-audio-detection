"""Region-level SED model: PANNs CNN14 encoder + U-Net decoder.

The encoder is borrowed from the panns_inference package (which vendors the
original PANNs Cnn14 PyTorch module). Pretrained weights are downloaded from
the upstream Zenodo release the first time the model is constructed.

Input to the model is a librosa-style log-mel spectrogram of shape
(B, n_mels, n_frames). The model bypasses PANNs' internal spectrogram extractor
so the same log-mel used for mask construction is used for inference. The
encoder's bn0 + conv blocks run as in PANNs; the decoder upsamples back to
(B, n_classes, n_mels, n_frames).
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import torch
import torch.nn as nn
import torch.nn.functional as F


PANNS_DATA_DIR = Path.home() / "panns_data"
PANNS_WEIGHTS_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth"
PANNS_WEIGHTS_FILENAME = "Cnn14_mAP=0.431.pth"


def _ensure_panns_data_dir() -> Path:
    """Create the directory + stub class-labels CSV that panns_inference's __init__ expects."""
    PANNS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = PANNS_DATA_DIR / "class_labels_indices.csv"
    if not csv_path.exists():
        csv_path.write_text("index,mid,display_name\n0,/m/09x0r,Speech\n", encoding="utf-8")
    return PANNS_DATA_DIR


def download_panns_weights(target_dir: Path | None = None) -> Path:
    """Download PANNs CNN14 pretrained weights to the target directory. Idempotent."""
    target_dir = target_dir or PANNS_DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / PANNS_WEIGHTS_FILENAME
    if out.exists() and out.stat().st_size > 0:
        return out
    print(f"Downloading PANNs CNN14 weights to {out} ...")
    urlretrieve(PANNS_WEIGHTS_URL, str(out))
    return out


class _UpBlock(nn.Module):
    """Single U-Net decoder block: upsample → concat skip → 2× conv-bn-relu."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class CnnSed(nn.Module):
    """PANNs CNN14 encoder + U-Net decoder → per-(mel, frame) 2-class logits."""

    def __init__(
        self,
        n_classes: int = 2,
        pretrained: bool = True,
        sample_rate: int = 32000,
        window_size: int = 1024,
        hop_size: int = 320,
        mel_bins: int = 64,
        fmin: float = 50.0,
        fmax: float = 14000.0,
    ) -> None:
        super().__init__()
        _ensure_panns_data_dir()
        # panns_inference's models.py expects to be imported as part of the package;
        # the stub class_labels_indices.csv keeps __init__ happy.
        from panns_inference.models import Cnn14  # noqa: E402

        self.cnn = Cnn14(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=527,  # placeholder; we ignore the classification head
        )
        if pretrained:
            self._load_pretrained()

        # Encoder channel widths come from PANNs CNN14 definition:
        # conv_block1: 64, conv_block2: 128, conv_block3: 256, conv_block4: 512,
        # conv_block5: 1024, conv_block6: 2048.
        self.dec5 = _UpBlock(in_channels=2048, skip_channels=1024, out_channels=1024)
        self.dec4 = _UpBlock(in_channels=1024, skip_channels=512, out_channels=512)
        self.dec3 = _UpBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.dec2 = _UpBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.dec1 = _UpBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.head = nn.Conv2d(64, n_classes, kernel_size=1)

    def _load_pretrained(self) -> None:
        weights_path = download_panns_weights()
        state = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        sd = state["model"] if "model" in state else state
        missing, unexpected = self.cnn.load_state_dict(sd, strict=False)
        # We don't use fc1 / fc_audioset so any "unexpected" hits there are fine.
        # Surface anything missing from the encoder for diagnosis.
        encoder_missing = [k for k in missing if k.startswith(("conv_block", "bn0"))]
        if encoder_missing:
            raise RuntimeError(f"PANNs encoder weights missing keys: {encoder_missing[:5]}...")

    def forward(self, log_mel: torch.Tensor) -> torch.Tensor:
        """log_mel: (B, n_mels, n_frames) → logits: (B, n_classes, n_mels, n_frames)."""
        # PANNs internal convention: (B, 1, n_frames, n_mels)
        x = log_mel.unsqueeze(1).transpose(2, 3)  # (B, 1, T, F)

        # bn0 normalizes per-mel; PANNs transposes to put F into the channel dim.
        x_t = x.transpose(1, 3)  # (B, F, T, 1)
        x_t = self.cnn.bn0(x_t)
        x = x_t.transpose(1, 3)  # (B, 1, T, F)

        x1 = self.cnn.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x2 = self.cnn.conv_block2(x1, pool_size=(2, 2), pool_type="avg")
        x3 = self.cnn.conv_block3(x2, pool_size=(2, 2), pool_type="avg")
        x4 = self.cnn.conv_block4(x3, pool_size=(2, 2), pool_type="avg")
        x5 = self.cnn.conv_block5(x4, pool_size=(2, 2), pool_type="avg")
        x6 = self.cnn.conv_block6(x5, pool_size=(1, 1), pool_type="avg")

        d = self.dec5(x6, x5)
        d = self.dec4(d, x4)
        d = self.dec3(d, x3)
        d = self.dec2(d, x2)
        d = self.dec1(d, x1)

        logits = self.head(d)  # (B, n_classes, T, F)

        # Resize to the input log-mel shape (encoder downsamples by ~32x then we
        # upsample 5 times; rounding can introduce off-by-one errors at the edges).
        if logits.shape[-2:] != (x.shape[-2], x.shape[-1]):
            logits = F.interpolate(
                logits, size=(x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False
            )

        return logits.transpose(2, 3)  # back to (B, n_classes, n_mels, n_frames)

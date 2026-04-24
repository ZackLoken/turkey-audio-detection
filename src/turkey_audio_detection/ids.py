"""Deterministic identity helpers for detections and review items."""

from __future__ import annotations

import hashlib
from pathlib import Path


def _digest(parts: list[str]) -> str:
    raw = "|".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def make_detection_id(
    source_audio_path: str,
    start_time_s: float,
    end_time_s: float,
    species_code: str,
) -> str:
    """Create a stable ID for one normalized BirdNET detection."""
    # Normalize path to forward-slash POSIX form so IDs are consistent
    # regardless of how the path was constructed on Windows.
    normalized_path = Path(source_audio_path).as_posix().lower()
    return "det_" + _digest(
        [
            normalized_path,
            f"{start_time_s:.3f}",
            f"{end_time_s:.3f}",
            species_code.strip().lower(),
        ]
    )


def make_item_id(detection_id: str, clip_start_s: float, clip_end_s: float) -> str:
    """Create a stable queue item ID for one review clip."""
    return "itm_" + _digest(
        [
            detection_id.strip().lower(),
            f"{clip_start_s:.3f}",
            f"{clip_end_s:.3f}",
        ]
    )

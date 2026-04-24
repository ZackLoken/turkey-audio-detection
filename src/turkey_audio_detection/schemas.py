"""Data contracts aligned to plan.md."""

from __future__ import annotations

from dataclasses import dataclass


VALID_LABELS = {"Tom", "Hen", "Background", "Skip"}


@dataclass(frozen=True)
class DetectionRow:
    detection_id: str
    project_root: str
    aru_id: str
    audio_path: str
    start_time_s: float
    end_time_s: float
    species_code: str
    species_common_name: str
    confidence: float
    birdnet_model_version: str
    source_filename: str
    source_row_index: int


@dataclass(frozen=True)
class QueueRow:
    item_id: str
    detection_id: str
    clip_path: str
    clip_start_s: float
    clip_end_s: float
    queue_order: int
    project_root: str
    aru_id: str
    source_audio_path: str


@dataclass(frozen=True)
class LabelRow:
    item_id: str
    detection_id: str
    reviewer_id: str
    reviewer_name: str
    label: str
    label_timestamp_utc: str
    session_id: str
    app_version: str

    def validate(self) -> None:
        if self.label not in VALID_LABELS:
            raise ValueError(f"Invalid label: {self.label}")

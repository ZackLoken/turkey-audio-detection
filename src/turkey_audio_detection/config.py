"""Typed configuration models for pipeline stages."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class IndexConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deployment_start: date = date(2026, 3, 1)
    timezone_name: str = "US/Eastern"
    latitude: float = 41.7
    longitude: float = -71.5
    prime_window_minutes_before: float = 90.0
    prime_window_minutes_after: float = 90.0
    wav_glob: str = "*.wav"


class BirdNetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_confidence: float = Field(default=0.1, ge=0.0, le=1.0)
    prime_window_only: bool = False
    latitude: float = 41.7
    longitude: float = -71.5


class ClipConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_duration_s: float = Field(default=3.0, gt=0.0, le=30.0)
    species_match_substring: str = "Wild Turkey"


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_ids: list[str] = Field(default_factory=list)
    model_id: str = ""  # filled by CLI with make_run_id("model") if empty
    clip_duration_s: float = Field(default=3.0, gt=0.0, le=30.0)
    epochs: int = Field(default=60, ge=1)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    include_non_consensus: bool = False
    val_fraction: float = Field(default=0.15, ge=0.0, le=0.5)
    test_fraction: float = Field(default=0.15, ge=0.0, le=0.5)
    num_workers: int = Field(default=2, ge=0)
    mixup_alpha: float = Field(default=0.4, ge=0.0, le=2.0)
    specaugment_enabled: bool = True
    background_mix_enabled: bool = True
    pos_weight: float = Field(default=10.0, gt=0.0)
    seed: int = Field(default=42, ge=0)
    pretrained: bool = True


class InferConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    audio_glob: str = "data/ARU_*/**/*.wav"
    inference_id: str = ""  # filled by CLI if empty
    window_duration_s: float = Field(default=3.0, gt=0.0, le=30.0)
    window_stride_s: float = Field(default=1.5, gt=0.0, le=30.0)
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_event_duration_s: float = Field(default=0.2, ge=0.0, le=10.0)
    merge_gap_s: float = Field(default=0.3, ge=0.0, le=10.0)
    batch_size: int = Field(default=16, ge=1)
    num_workers: int = Field(default=2, ge=0)

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


class SedTrainConfig(BaseModel):
    """Config for the single-stage ConvNeXt-BirdSet frame-level SED trainer."""

    model_config = ConfigDict(extra="forbid")

    run_ids: list[str] = Field(default_factory=list)
    model_id: str = ""  # filled by CLI with make_run_id("model") if empty
    clip_duration_s: float = Field(default=3.0, gt=0.0, le=30.0)
    include_non_consensus: bool = False

    # splits: site-grouped (data/site_map.csv) + optional year holdout
    site_map_path: str = "data/site_map.csv"
    val_fraction: float = Field(default=0.15, ge=0.0, le=0.5)
    test_fraction: float = Field(default=0.15, ge=0.0, le=0.5)
    holdout_years: list[int] = Field(default_factory=list)

    # model
    pretrained: bool = True
    n_stages: int = Field(default=2, ge=1, le=4)
    temporal: str = Field(default="bigru")  # "bigru" | "tcn"
    hidden_size: int = Field(default=256, ge=8)
    n_layers: int = Field(default=2, ge=1)
    dropout: float = Field(default=0.2, ge=0.0, le=0.9)

    # loss
    loss: str = Field(default="focal")  # "focal" | "bce"
    focal_gamma: float = Field(default=2.0, ge=0.0)
    pos_weight: float = Field(default=1.0, gt=0.0)

    # optimization (gradual-unfreeze schedule scales LR/batch per phase)
    base_lr: float = Field(default=1e-3, gt=0.0)
    base_batch_size: int = Field(default=32, ge=1)
    backbone_lr_mult: float = Field(default=0.1, gt=0.0, le=1.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    mixup_alpha: float = Field(default=0.0, ge=0.0, le=2.0)
    specaugment_enabled: bool = True
    early_stop_patience: int = Field(default=5, ge=1)

    num_workers: int = Field(default=2, ge=0)
    seed: int = Field(default=42, ge=0)


class SedInferConfig(BaseModel):
    """Config for whole-recording frame-level SED inference (the trained model runs
    directly over full recordings; BirdNET is not used at inference)."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    audio_glob: str = "data/ARU_*/**/*.wav"  # recordings to run the detector over
    inference_id: str = ""  # filled by CLI if empty
    window_duration_s: float = Field(default=3.0, gt=0.0, le=30.0)
    window_stride_s: float = Field(default=1.0, gt=0.0, le=30.0)
    min_event_duration_s: float = Field(default=0.1, ge=0.0, le=10.0)
    merge_gap_s: float = Field(default=0.2, ge=0.0, le=10.0)
    thresholds: dict[str, float] | None = None  # overrides checkpoint per-class thresholds
    batch_size: int = Field(default=16, ge=1)
    site_map_path: str = "data/site_map.csv"

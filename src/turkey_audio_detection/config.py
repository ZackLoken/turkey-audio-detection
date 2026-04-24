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

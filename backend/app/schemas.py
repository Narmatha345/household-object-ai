"""Pydantic schemas shared by the API layer and services."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DetectionSource = Literal["local", "openai"]


class DetectedObject(BaseModel):
    label: str
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="0-1 score; None when the source gives none"
    )


class DetectionResponse(BaseModel):
    source: DetectionSource
    objects: list[DetectedObject]
    fallback_used: bool
    fallback_reason: str | None = None
    description: str | None = None
    reliability_note: str | None = None
    # Low-confidence local detections, kept for transparency when falling back.
    local_candidates: list[DetectedObject] = Field(default_factory=list)
    confidence_threshold: float


class HealthResponse(BaseModel):
    status: Literal["ok"]
    local_model_loaded: bool
    openai_fallback_configured: bool


class StatsResponse(BaseModel):
    total_requests: int
    local_detections: int
    openai_fallbacks: int
    failed_requests: int
    fallback_rate: float = Field(description="openai_fallbacks / total_requests, 0-1")


class ErrorResponse(BaseModel):
    detail: str
    code: str

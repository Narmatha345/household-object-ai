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
    box: list[float] | None = Field(
        default=None, description="Bounding box [x1, y1, x2, y2] in image pixels; local detections only"
    )


class DetectionTiming(BaseModel):
    """Server-side durations in milliseconds, measured with time.perf_counter()."""

    local_inference_ms: float | None = Field(
        default=None, description="YOLO predict call (preprocess + inference + NMS); None if it failed"
    )
    local_processing_ms: float | None = Field(
        default=None, description="Household-label filtering, confidence gate and de-duplication"
    )
    openai_ms: float | None = Field(
        default=None, description="OpenAI fallback (image encoding + API call); None when not called"
    )
    total_ms: float = Field(
        description="From the /api/detect handler start (after upload receipt) to the response"
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
    # Optional so older clients and fixtures without timing stay valid.
    timing: DetectionTiming | None = None


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

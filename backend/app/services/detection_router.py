"""Core local-first routing: local YOLO -> confidence gate -> OpenAI fallback."""

from __future__ import annotations

import logging
import time

from PIL import Image

from app.errors import AppError, LocalInferenceError, ModelLoadError
from app.schemas import DetectionResponse, DetectionTiming
from app.services.local_detection import LocalDetectionResult, LocalDetectionService
from app.services.openai_fallback import FallbackDetector
from app.services.stats import StatsTracker

logger = logging.getLogger(__name__)


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def _log_timing(source: str, timing: DetectionTiming) -> None:
    logger.info(
        "Request timing: source=%s local_inference_ms=%s postprocess_ms=%s openai_ms=%s total_ms=%s",
        source, timing.local_inference_ms, timing.local_processing_ms, timing.openai_ms, timing.total_ms,
    )


class DetectionRouter:
    def __init__(
        self,
        local_service: LocalDetectionService,
        fallback: FallbackDetector,
        stats: StatsTracker,
    ) -> None:
        self._local = local_service
        self._fallback = fallback
        self._stats = stats

    def detect(self, image: Image.Image, request_started: float | None = None) -> DetectionResponse:
        """Route one image. `request_started` is a time.perf_counter() value from the caller,
        so total_ms can include upload decoding; defaults to now."""
        started = request_started if request_started is not None else time.perf_counter()
        threshold = self._local.confidence_threshold

        # 1. Always try the local model first.
        try:
            local_result = self._local.detect(image)
        except (ModelLoadError, LocalInferenceError) as exc:
            # Keep the app usable when the local model is broken, but make it visible.
            logger.warning("Local detection unavailable, using fallback: %s", exc.message)
            local_result = LocalDetectionResult(
                is_reliable=False, unreliable_reason="the local model was unavailable"
            )

        # 2. Reliable -> return immediately. OpenAI is NOT called.
        if local_result.is_reliable:
            logger.info(
                "Route: Local ML (OpenAI not called): %s",
                ", ".join(f"{o.label} {o.confidence:.2f}" for o in local_result.objects),
            )
            self._stats.record_local()
            timing = DetectionTiming(
                local_inference_ms=local_result.inference_ms,
                local_processing_ms=local_result.processing_ms,
                total_ms=_ms_since(started),
            )
            _log_timing("local", timing)
            return DetectionResponse(
                source="local",
                objects=local_result.objects,
                fallback_used=False,
                confidence_threshold=threshold,
                timing=timing,
            )

        # 3. Unreliable -> OpenAI fallback.
        hints = [c.label for c in local_result.candidates]
        logger.info("Route: OpenAI fallback because %s", local_result.unreliable_reason)
        openai_start = time.perf_counter()
        try:
            fallback_result = self._fallback.detect(image, local_hints=hints or None)
        except AppError:
            self._stats.record_failure()
            raise
        openai_ms = _ms_since(openai_start)

        self._stats.record_fallback()
        timing = DetectionTiming(
            local_inference_ms=local_result.inference_ms,
            local_processing_ms=local_result.processing_ms,
            openai_ms=openai_ms,
            total_ms=_ms_since(started),
        )
        _log_timing("openai", timing)
        return DetectionResponse(
            source="openai",
            objects=fallback_result.objects,
            fallback_used=True,
            fallback_reason=f"Used because {local_result.unreliable_reason}.",
            description=fallback_result.description,
            reliability_note=fallback_result.reliability_note,
            local_candidates=local_result.candidates,
            confidence_threshold=threshold,
            timing=timing,
        )

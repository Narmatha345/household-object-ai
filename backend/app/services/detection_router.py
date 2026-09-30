"""Core local-first routing: local YOLO -> confidence gate -> OpenAI fallback."""

from __future__ import annotations

import logging

from PIL import Image

from app.errors import AppError, LocalInferenceError, ModelLoadError
from app.schemas import DetectionResponse
from app.services.local_detection import LocalDetectionResult, LocalDetectionService
from app.services.openai_fallback import FallbackDetector
from app.services.stats import StatsTracker

logger = logging.getLogger(__name__)


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

    def detect(self, image: Image.Image) -> DetectionResponse:
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
            self._stats.record_local()
            return DetectionResponse(
                source="local",
                objects=local_result.objects,
                fallback_used=False,
                confidence_threshold=threshold,
            )

        # 3. Unreliable -> OpenAI fallback.
        hints = [c.label for c in local_result.candidates]
        try:
            fallback_result = self._fallback.detect(image, local_hints=hints or None)
        except AppError:
            self._stats.record_failure()
            raise

        self._stats.record_fallback()
        return DetectionResponse(
            source="openai",
            objects=fallback_result.objects,
            fallback_used=True,
            fallback_reason=f"Used because {local_result.unreliable_reason}.",
            description=fallback_result.description,
            reliability_note=fallback_result.reliability_note,
            local_candidates=local_result.candidates,
            confidence_threshold=threshold,
        )

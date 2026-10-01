"""Turns raw detector output into a local result with a reliability verdict."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from PIL import Image

from app.config import UNVALIDATED_LABELS
from app.ml.detector import ObjectDetector, RawDetection
from app.schemas import DetectedObject

logger = logging.getLogger(__name__)


@dataclass
class LocalDetectionResult:
    is_reliable: bool
    objects: list[DetectedObject] = field(default_factory=list)
    candidates: list[DetectedObject] = field(default_factory=list)
    unreliable_reason: str | None = None
    inference_ms: float | None = None
    processing_ms: float | None = None


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def _all_instances(detections: list[RawDetection]) -> list[DetectedObject]:
    """Every detection as its own object (no per-class merging), highest confidence first."""
    ranked = sorted(detections, key=lambda det: det.confidence, reverse=True)
    return [
        DetectedObject(
            label=det.label,
            confidence=round(det.confidence, 4),
            box=list(det.box) if det.box is not None else None,
        )
        for det in ranked
    ]


def _best_per_label(detections: list[RawDetection]) -> list[DetectedObject]:
    """Collapse multiple boxes of the same class to the highest-confidence one."""
    best: dict[str, float] = {}
    for det in detections:
        if det.confidence > best.get(det.label, -1.0):
            best[det.label] = det.confidence
    ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)
    return [DetectedObject(label=label, confidence=round(conf, 4)) for label, conf in ranked]


class LocalDetectionService:
    def __init__(
        self,
        detector: ObjectDetector,
        confidence_threshold: float,
        allowed_labels: tuple[str, ...] = (),
    ) -> None:
        self._detector = detector
        self.confidence_threshold = confidence_threshold
        self._allowed = {label.lower() for label in allowed_labels}

    @property
    def is_available(self) -> bool:
        return self._detector.is_loaded

    def detect(self, image: Image.Image) -> LocalDetectionResult:
        inference_start = time.perf_counter()
        raw = self._detector.detect(image)
        inference_ms = _elapsed_ms(inference_start)

        processing_start = time.perf_counter()
        result = self._evaluate(raw)
        result.inference_ms = inference_ms
        result.processing_ms = _elapsed_ms(processing_start)
        return result

    def _evaluate(self, raw: list[RawDetection]) -> LocalDetectionResult:
        if self._allowed:
            dropped = sorted({d.label for d in raw if d.label.lower() not in self._allowed})
            if dropped:
                logger.warning("Dropped labels outside the allowed classes: %s", ", ".join(dropped))
            raw = [d for d in raw if d.label.lower() in self._allowed]

        for label in sorted({d.label for d in raw if d.label.lower() in UNVALIDATED_LABELS}):
            logger.warning("'%s' was detected but this class had no training data; treat it as unvalidated.", label)

        confident = [d for d in raw if d.confidence >= self.confidence_threshold]
        weak = [d for d in raw if d.confidence < self.confidence_threshold]
        top = max(raw, key=lambda d: d.confidence, default=None)
        logger.info(
            "Local result: %d allowed detection(s), top=%s, threshold=%.2f -> %s",
            len(raw),
            f"{top.label} {top.confidence:.2f}" if top else "none",
            self.confidence_threshold,
            "reliable" if confident else "unreliable",
        )

        # Reliable iff the highest-confidence allowed detection reaches the threshold.
        # Every detection at or above the threshold is returned, including repeated classes.
        if confident:
            return LocalDetectionResult(is_reliable=True, objects=_all_instances(confident))

        reason = (
            "no household object was detected by the local model"
            if not raw
            else f"local detection confidence was below the {self.confidence_threshold:.0%} threshold"
        )
        return LocalDetectionResult(
            is_reliable=False,
            candidates=_best_per_label(weak),
            unreliable_reason=reason,
        )

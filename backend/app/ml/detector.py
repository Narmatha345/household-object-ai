"""Local object detector abstraction.

`ObjectDetector` is the contract the rest of the app depends on. The YOLO
implementation below can be replaced by any class with the same methods.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

from app.errors import LocalInferenceError, ModelLoadError
from app.ml.model_loader import load_yolo_model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawDetection:
    label: str
    confidence: float
    box: tuple[float, float, float, float] | None = None  # xyxy, in image pixels


class ObjectDetector(Protocol):
    @property
    def is_loaded(self) -> bool: ...

    def detect(self, image: Image.Image) -> list[RawDetection]: ...


class LocalObjectDetector:
    """YOLO-backed detector. Loads weights once and reuses them for every request."""

    # Low floor so the service layer can still see (and report) weak candidates.
    # The reliability threshold is applied later, from configuration.
    _MIN_CANDIDATE_CONFIDENCE = 0.15

    def __init__(self, model_path: Path, device: str | None = None, image_size: int = 480) -> None:
        self._model_path = model_path
        self._device = device
        self._image_size = image_size
        self._model = None
        # Ultralytics predictors are not guaranteed to be thread-safe.
        self._predict_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def class_names(self) -> dict[int, str]:
        """The model's own class mapping (model.names); empty until loaded."""
        return dict(self._model.names) if self._model is not None else {}

    def load(self) -> None:
        if self._model is None:
            self._model = load_yolo_model(self._model_path)
            logger.info(
                "Local model loaded from %s with %d classes (imgsz=%d): %s",
                self._model_path, len(self.class_names), self._image_size, self.class_names,
            )

    def detect(self, image: Image.Image) -> list[RawDetection]:
        if self._model is None:
            raise ModelLoadError("The local detection model is not loaded.")

        try:
            with self._predict_lock:
                results = self._model.predict(
                    source=image,
                    conf=self._MIN_CANDIDATE_CONFIDENCE,
                    imgsz=self._image_size,
                    device=self._device,
                    verbose=False,
                )
        except Exception as exc:
            logger.exception("Local YOLO inference failed")
            raise LocalInferenceError("Local detection failed for this image.") from exc

        speed = getattr(results[0], "speed", None) if results else None  # per-stage ms from Ultralytics
        if speed:
            logger.info(
                "PyTorch timing: preprocess_ms=%.1f inference_ms=%.1f postprocess_ms=%.1f",
                speed.get("preprocess") or 0, speed.get("inference") or 0, speed.get("postprocess") or 0,
            )

        # Labels come only from the model's own mapping; nothing is renamed or invented.
        names = self._model.names
        detections: list[RawDetection] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for cls_id, conf, xyxy in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()):
                x1, y1, x2, y2 = (round(float(v), 1) for v in xyxy)
                detection = RawDetection(
                    label=str(names[int(cls_id)]),
                    confidence=float(conf),
                    box=(x1, y1, x2, y2),
                )
                logger.debug("YOLO box: %s %.3f %s", detection.label, detection.confidence, detection.box)
                detections.append(detection)
        logger.info(
            "YOLO returned %d detection(s): %s",
            len(detections),
            ", ".join(f"{d.label} {d.confidence:.2f}" for d in detections) or "none",
        )
        return detections

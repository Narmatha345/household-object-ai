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


class ObjectDetector(Protocol):
    @property
    def is_loaded(self) -> bool: ...

    def detect(self, image: Image.Image) -> list[RawDetection]: ...


class LocalObjectDetector:
    """YOLO-backed detector. Loads weights once and reuses them for every request."""

    # Low floor so the service layer can still see (and report) weak candidates.
    # The reliability threshold is applied later, from configuration.
    _MIN_CANDIDATE_CONFIDENCE = 0.15

    def __init__(self, model_path: Path, device: str | None = None) -> None:
        self._model_path = model_path
        self._device = device
        self._model = None
        # Ultralytics predictors are not guaranteed to be thread-safe.
        self._predict_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is None:
            self._model = load_yolo_model(self._model_path)

    def detect(self, image: Image.Image) -> list[RawDetection]:
        if self._model is None:
            raise ModelLoadError("The local detection model is not loaded.")

        try:
            with self._predict_lock:
                results = self._model.predict(
                    source=image,
                    conf=self._MIN_CANDIDATE_CONFIDENCE,
                    device=self._device,
                    verbose=False,
                )
        except Exception as exc:
            logger.exception("Local YOLO inference failed")
            raise LocalInferenceError("Local detection failed for this image.") from exc

        detections: list[RawDetection] = []
        for result in results:
            names = result.names
            if result.boxes is None:
                continue
            for cls_id, conf in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
                detections.append(RawDetection(label=str(names[int(cls_id)]), confidence=float(conf)))
        return detections

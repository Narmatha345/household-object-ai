"""Loads YOLO weights once per process.

Swapping in a custom household model only requires pointing MODEL_PATH at the
new weights file (e.g. models/household-yolo26n.pt). No other code changes.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from app.errors import ModelLoadError

if TYPE_CHECKING:
    from ultralytics import YOLO

logger = logging.getLogger(__name__)

_cache: dict[Path, "YOLO"] = {}
_lock = threading.Lock()


def load_yolo_model(model_path: Path) -> "YOLO":
    """Return a cached YOLO model for `model_path`, loading it on first use.

    If the file does not exist and its name matches an official Ultralytics
    asset (e.g. yolo26n.pt), Ultralytics downloads it to that path.
    """
    model_path = model_path.resolve()
    with _lock:
        if model_path in _cache:
            return _cache[model_path]

        try:
            from ultralytics import YOLO

            model_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Loading local YOLO model from %s", model_path)
            model = YOLO(str(model_path))
        except Exception as exc:  # ultralytics raises a wide range of errors
            logger.exception("Failed to load YOLO model from %s", model_path)
            raise ModelLoadError(
                "The local detection model could not be loaded."
            ) from exc

        _cache[model_path] = model
        return model

"""Builds and loads the detection pipeline.

Shared by the web API (app.main) and the A2A agent (app.a2a.server) so both
use exactly the same model loading, confidence gate and fallback routing.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.config import Settings
from app.errors import ModelLoadError
from app.ml.detector import LocalObjectDetector, ObjectDetector
from app.ml.onnx_detector import OnnxObjectDetector
from app.services.local_detection import LocalDetectionService
from app.services.openai_fallback import FallbackDetector, OpenAIFallbackService

logger = logging.getLogger("household_object_ai")


def build_detector(settings: Settings, model_path: Path) -> LocalObjectDetector | OnnxObjectDetector:
    """ONNX Runtime for .onnx weights, Ultralytics/PyTorch for everything else."""
    if model_path.suffix.lower() == ".onnx":
        return OnnxObjectDetector(
            model_path, image_size=settings.inference_image_size, threads=settings.inference_threads
        )
    return LocalObjectDetector(model_path, image_size=settings.inference_image_size)


def build_fallback(settings: Settings) -> OpenAIFallbackService:
    return OpenAIFallbackService(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        timeout_seconds=settings.openai_timeout_seconds,
        image_max_side=settings.openai_image_max_side,
    )


def _load_with_fallback(
    detector: LocalObjectDetector | OnnxObjectDetector, settings: Settings
) -> LocalObjectDetector | OnnxObjectDetector:
    """Load the configured model; if that fails, try FALLBACK_MODEL_PATH (e.g. the .pt model)."""
    try:
        detector.load()
        return detector
    except ModelLoadError:
        if settings.fallback_model_path is None:
            raise
        logger.warning("Primary model failed to load; loading fallback model %s", settings.fallback_model_path)
        fallback_detector = build_detector(settings, settings.fallback_model_path)
        fallback_detector.load()
        return fallback_detector


def load_local_model(
    detector: ObjectDetector, local_service: LocalDetectionService, settings: Settings, fallback: FallbackDetector
) -> None:
    """Load the model once at startup instead of on every request.

    Test fakes (anything that is not a real detector) are left untouched.
    """
    if isinstance(detector, (LocalObjectDetector, OnnxObjectDetector)):
        try:
            active = _load_with_fallback(detector, settings)
            if active is not detector:
                local_service.use_detector(active)
            logger.info("Local model ready: %s", type(active).__name__)
            model_labels = set(active.class_names.values())
            allowed = set(settings.household_labels)
            if allowed and model_labels != allowed:
                logger.warning(
                    "Model classes do not match the allowed classes. Not in model: %s. Ignored model classes: %s",
                    sorted(allowed - model_labels) or "none",
                    sorted(model_labels - allowed) or "none",
                )
        except ModelLoadError:
            logger.error("Starting without a local model; requests will use the fallback.")
    if not fallback.is_configured:
        logger.warning("OPENAI_API_KEY not set; low-confidence images cannot fall back to OpenAI.")

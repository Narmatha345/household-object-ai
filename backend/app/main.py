"""FastAPI entry point.

Run with:  uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import Settings, get_settings
from app.errors import AppError, ModelLoadError
from app.ml.detector import LocalObjectDetector, ObjectDetector
from app.services.detection_router import DetectionRouter
from app.services.local_detection import LocalDetectionService
from app.services.openai_fallback import FallbackDetector, OpenAIFallbackService
from app.services.stats import StatsTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("household_object_ai")


def create_app(
    settings: Settings | None = None,
    detector: ObjectDetector | None = None,
    fallback: FallbackDetector | None = None,
) -> FastAPI:
    """Build the app. Tests inject fake `detector` / `fallback` implementations."""
    settings = settings or get_settings()
    detector = detector or LocalObjectDetector(settings.model_path)
    fallback = fallback or OpenAIFallbackService(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        timeout_seconds=settings.openai_timeout_seconds,
        image_max_side=settings.openai_image_max_side,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load the model once at startup instead of on every request.
        if isinstance(detector, LocalObjectDetector):
            try:
                detector.load()
                logger.info("Local model ready: %s", settings.model_path.name)
            except ModelLoadError:
                logger.error("Starting without a local model; requests will use the fallback.")
        if not fallback.is_configured:
            logger.warning("OPENAI_API_KEY not set; low-confidence images cannot fall back to OpenAI.")
        yield

    app = FastAPI(title="Household Object AI", version="0.1.0", lifespan=lifespan)

    stats = StatsTracker()
    local_service = LocalDetectionService(
        detector,
        confidence_threshold=settings.local_confidence_threshold,
        allowed_labels=settings.household_labels,
    )
    app.state.settings = settings
    app.state.stats = stats
    app.state.fallback = fallback
    app.state.local_service = local_service
    app.state.detection_router = DetectionRouter(local_service, fallback, stats)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message, "code": exc.code})

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"detail": "Please attach an image file in the 'file' field.", "code": "invalid_request"},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error")
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected server error occurred.", "code": "internal_error"},
        )

    app.include_router(router)
    return app


app = create_app()

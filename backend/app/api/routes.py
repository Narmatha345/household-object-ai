from __future__ import annotations

import time

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.schemas import DetectionResponse, ErrorResponse, HealthResponse, StatsResponse
from app.services.image_input import decode_image, validate_upload_metadata

router = APIRouter(prefix="/api")

_ERROR_RESPONSES = {
    code: {"model": ErrorResponse} for code in (400, 413, 415, 500, 502, 503)
}


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    state = request.app.state
    return HealthResponse(
        status="ok",
        local_model_loaded=state.local_service.is_available,
        openai_fallback_configured=state.fallback.is_configured,
    )


@router.post("/detect", response_model=DetectionResponse, responses=_ERROR_RESPONSES)
async def detect(request: Request, file: UploadFile = File(...)) -> DetectionResponse:
    # Starts once FastAPI has received the upload; network transfer time is excluded.
    started = time.perf_counter()
    state = request.app.state
    validate_upload_metadata(file.filename, file.content_type)

    # Read at most one byte past the limit so oversized uploads are rejected cheaply.
    data = await file.read(state.settings.max_upload_bytes + 1)
    image = decode_image(data, state.settings.max_upload_bytes)

    # YOLO inference and the OpenAI call are blocking; keep the event loop free.
    return await run_in_threadpool(state.detection_router.detect, image, started)


@router.get("/stats", response_model=StatsResponse)
def stats(request: Request) -> StatsResponse:
    return request.app.state.stats.snapshot()

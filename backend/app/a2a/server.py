"""A2A server: a separate ASGI app that exposes the detector as an A2A agent.

It runs on its own port (default 9100), apart from the web UI/API on 8000,
so only the A2A surface is ever exposed to the Tailnet.

Run with:  python -m app.a2a
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import add_a2a_routes_to_fastapi, create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from fastapi import FastAPI

from app.a2a.agent_card import JSONRPC_PATH, build_agent_card
from app.a2a.config import A2ASettings, get_a2a_settings
from app.a2a.executor import ExpectedRejectionFilter, ObjectRecognitionExecutor
from app.a2a.security import A2AGuardMiddleware
from app.bootstrap import build_detector, build_fallback, load_local_model
from app.config import Settings, get_settings
from app.ml.detector import ObjectDetector
from app.services.detection_router import DetectionRouter
from app.services.local_detection import LocalDetectionService
from app.services.openai_fallback import FallbackDetector
from app.services.stats import StatsTracker

logger = logging.getLogger("household_object_ai.a2a")

# Base64 makes inline image bytes ~4/3 larger; leave room for the JSON-RPC envelope.
_ENVELOPE_OVERHEAD_BYTES = 64 * 1024


def create_a2a_app(
    settings: Settings | None = None,
    a2a_settings: A2ASettings | None = None,
    detector: ObjectDetector | None = None,
    fallback: FallbackDetector | None = None,
) -> FastAPI:
    """Build the A2A app. Tests inject fake `detector` / `fallback` implementations."""
    settings = settings or get_settings()
    a2a_settings = a2a_settings or get_a2a_settings()
    a2a_settings.validate()
    sdk_task_logger = logging.getLogger("a2a.server.agent_execution.active_task")
    if not any(isinstance(f, ExpectedRejectionFilter) for f in sdk_task_logger.filters):
        sdk_task_logger.addFilter(ExpectedRejectionFilter())

    detector = detector or build_detector(settings, settings.model_path)
    fallback = fallback or build_fallback(settings)
    local_service = LocalDetectionService(
        detector,
        confidence_threshold=settings.local_confidence_threshold,
        allowed_labels=settings.household_labels,
    )
    stats = StatsTracker()
    detection_router = DetectionRouter(local_service, fallback, stats)
    agent_card = build_agent_card(a2a_settings, settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        load_local_model(detector, local_service, settings, fallback)
        logger.info(
            "A2A agent ready: card=%s/.well-known/agent-card.json rpc=%s%s auth=%s",
            a2a_settings.base_url, a2a_settings.base_url, JSONRPC_PATH,
            "bearer token" if a2a_settings.auth_token else "NONE (loopback only)",
        )
        yield

    app = FastAPI(title="Household Object Recognition A2A Agent", version=agent_card.version, lifespan=lifespan)
    app.state.stats = stats

    request_handler = DefaultRequestHandler(
        agent_executor=ObjectRecognitionExecutor(detection_router, settings.max_upload_bytes),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
        validate_input_modes=True,
    )
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(agent_card),
        # v0.3 compat lets agents built on the older A2A SDK (message/send) call this agent too.
        jsonrpc_routes=create_jsonrpc_routes(request_handler, rpc_url=JSONRPC_PATH, enable_v0_3_compat=True),
    )

    @app.get("/health", tags=["Operations"])
    def health() -> dict:
        """Liveness check for operators. Not part of the A2A protocol."""
        return {
            "status": "ok",
            "local_model_loaded": local_service.is_available,
            "openai_fallback_configured": fallback.is_configured,
            "auth_required": bool(a2a_settings.auth_token),
            "stats": stats.snapshot().model_dump(),
        }

    max_body = int(settings.max_upload_bytes * 4 / 3) + _ENVELOPE_OVERHEAD_BYTES
    app.add_middleware(A2AGuardMiddleware, auth_token=a2a_settings.auth_token, max_body_bytes=max_body)
    return app


def build_app() -> FastAPI:
    """uvicorn factory: `uvicorn app.a2a.server:build_app --factory`."""
    return create_a2a_app()

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.errors import FallbackError
from app.main import create_app
from app.ml.detector import RawDetection
from app.schemas import DetectedObject
from app.services.openai_fallback import FallbackResult


class FakeDetector:
    def __init__(self, detections: list[RawDetection] | None = None) -> None:
        self.detections = detections or []
        self.calls = 0

    @property
    def is_loaded(self) -> bool:
        return True

    def detect(self, image: Image.Image) -> list[RawDetection]:
        self.calls += 1
        return list(self.detections)


class FakeFallback:
    """Mock OpenAI fallback. Never touches the network."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0
        self.last_hints: list[str] | None = None

    @property
    def is_configured(self) -> bool:
        return True

    def detect(self, image: Image.Image, local_hints: list[str] | None = None) -> FallbackResult:
        self.calls += 1
        self.last_hints = local_hints
        if self.fail:
            raise FallbackError("The AI fallback service is currently unavailable. Please try again.")
        return FallbackResult(
            objects=[DetectedObject(label="ceiling fan", confidence=0.88)],
            description="A white ceiling fan mounted on a living room ceiling.",
            reliability_note="The fan blades and motor housing are clearly visible.",
        )


@pytest.fixture
def settings() -> Settings:
    return Settings(openai_api_key="test-key", local_confidence_threshold=0.70)


@pytest.fixture
def detector() -> FakeDetector:
    return FakeDetector()


@pytest.fixture
def fallback() -> FakeFallback:
    return FakeFallback()


@pytest.fixture
def client(settings: Settings, detector: FakeDetector, fallback: FakeFallback):
    app = create_app(settings=settings, detector=detector, fallback=fallback)
    with TestClient(app) as test_client:
        yield test_client


def make_image_bytes(fmt: str = "JPEG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color=(120, 90, 60)).save(buffer, format=fmt)
    return buffer.getvalue()

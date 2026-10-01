"""OpenAI fallback prompt and response handling, with a fake OpenAI client (no network)."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app
from app.services.openai_fallback import (
    PROMPT,
    FallbackResult,
    OpenAIFallbackService,
    _FallbackObject,
    _FallbackPayload,
)

from tests.conftest import FakeDetector, make_image_bytes


class FakeResponses:
    def __init__(self, payload: _FallbackPayload) -> None:
        self.payload = payload
        self.kwargs: dict = {}

    def parse(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(output_parsed=self.payload)


def _service(payload: _FallbackPayload) -> tuple[OpenAIFallbackService, FakeResponses]:
    service = OpenAIFallbackService(api_key="test-key", model="test-model")
    responses = FakeResponses(payload)
    service._client = SimpleNamespace(responses=responses)
    return service, responses


def _sent_prompt(responses: FakeResponses) -> str:
    return responses.kwargs["input"][0]["content"][0]["text"]


def test_prompt_forbids_guessing_and_allows_empty_result():
    for rule in (
        "Never guess or hallucinate an object",
        "Do not infer an object from colors, patterns",
        "return an empty objects list",
        "When uncertain,\nreturn no detection",
        "single-color",
        "Never invent an object just to give an answer",
    ):
        assert rule in PROMPT
    # The household-object task is still there.
    assert "household objects" in PROMPT


def test_blank_image_with_empty_openai_answer_returns_no_objects():
    service, responses = _service(
        _FallbackPayload(objects=[], description="No recognizable object is clearly visible.", reliability="Plain color.")
    )
    result = service.detect(Image.new("RGB", (320, 240), (120, 160, 200)))

    assert result.objects == []
    assert "no detection" in _sent_prompt(responses)
    assert responses.kwargs["input"][0]["content"][1]["image_url"].startswith("data:image/jpeg;base64,")


def test_genuine_household_answer_still_parses():
    service, _ = _service(
        _FallbackPayload(
            objects=[_FallbackObject(name=" Chair ", confidence=0.91234), _FallbackObject(name="  ", confidence=0.5)],
            description="A wooden chair.",
            reliability="Clearly visible.",
        )
    )
    result = service.detect(Image.new("RGB", (64, 64)))

    assert [(o.label, o.confidence, o.box) for o in result.objects] == [("chair", 0.9123, None)]
    assert result.description == "A wooden chair."
    assert result.reliability_note == "Clearly visible."


def test_local_hints_are_not_presented_as_answers_to_confirm():
    service, responses = _service(_FallbackPayload(objects=[], description="-", reliability="-"))
    service.detect(Image.new("RGB", (64, 64)), local_hints=["sofa"])

    prompt = _sent_prompt(responses)
    assert "sofa" in prompt
    assert "only if it is clearly visible" in prompt
    assert "Verify or correct" not in prompt


class EmptyFallback:
    is_configured = True

    def detect(self, image, local_hints=None) -> FallbackResult:
        return FallbackResult(objects=[], description="No recognizable object is clearly visible.", reliability_note="-")


def test_empty_fallback_keeps_api_response_structure():
    app = create_app(settings=Settings(openai_api_key="test-key"), detector=FakeDetector(), fallback=EmptyFallback())
    with TestClient(app) as client:
        response = client.post("/api/detect", files={"file": ("blank.jpg", make_image_bytes(), "image/jpeg")})
    body = response.json()

    assert response.status_code == 200
    assert set(body) == {
        "source", "objects", "fallback_used", "fallback_reason", "description",
        "reliability_note", "local_candidates", "confidence_threshold", "timing",
    }
    assert body["source"] == "openai"
    assert body["fallback_used"] is True
    assert body["objects"] == []

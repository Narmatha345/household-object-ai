"""A2A agent tests. The detector and OpenAI fallback are fakes; no model or network is used."""

from __future__ import annotations

import base64
import io
import json
import uuid

import httpx
import pytest
from a2a.client import ClientConfig, ClientFactory
from a2a.helpers import get_data_parts
from a2a.types import Message, Part, Role, SendMessageRequest, TaskState
from fastapi.testclient import TestClient
from PIL import Image

from app.a2a.config import A2ASettings
from app.a2a.server import create_a2a_app
from app.config import Settings
from app.errors import LocalInferenceError
from app.ml.detector import RawDetection

from .conftest import FakeDetector, FakeFallback, make_image_bytes

TOKEN = "test-token-0123456789abcdef"
AUTH = {"Authorization": f"Bearer {TOKEN}", "A2A-Version": "1.0"}
INVALID_PARAMS = -32602
CONTENT_TYPE_NOT_SUPPORTED = -32005


class NotConfiguredFallback(FakeFallback):
    @property
    def is_configured(self) -> bool:
        return False

    def detect(self, image, local_hints=None):
        from app.errors import FallbackNotConfiguredError

        raise FallbackNotConfiguredError("The local model was not confident and the OpenAI fallback is not configured.")


class CrashingDetector(FakeDetector):
    def __init__(self, exc: Exception) -> None:
        super().__init__()
        self.exc = exc

    def detect(self, image):
        self.calls += 1
        raise self.exc


def make_app(detector=None, fallback=None, settings=None, token=TOKEN, public_url="http://agent.example.ts.net:9100"):
    return create_a2a_app(
        settings=settings or Settings(openai_api_key="test-key", local_confidence_threshold=0.70),
        a2a_settings=A2ASettings(auth_token=token, public_url=public_url),
        detector=detector or FakeDetector(),
        fallback=fallback or FakeFallback(),
    )


@pytest.fixture
def detector() -> FakeDetector:
    return FakeDetector([RawDetection("chair", 0.91, (1.0, 2.0, 30.0, 40.0)), RawDetection("tv", 0.82)])


@pytest.fixture
def a2a(detector, fallback):
    with TestClient(make_app(detector, fallback)) as client:
        yield client


def image_part(data: bytes | None = None, media_type: str = "image/jpeg", filename: str = "photo.jpg") -> dict:
    data = make_image_bytes() if data is None else data
    return {"raw": base64.b64encode(data).decode(), "mediaType": media_type, "filename": filename}


def rpc(client, parts, method="SendMessage", headers=AUTH, configuration=None):
    params = {"message": {"messageId": str(uuid.uuid4()), "role": "ROLE_USER", "parts": parts}}
    if configuration:
        params["configuration"] = configuration
    body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    return client.post("/a2a", json=body, headers=headers)


def rpc_error(response) -> dict:
    assert response.status_code == 200, response.text
    body = response.json()
    assert "error" in body, body
    return body["error"]


# --------------------------------------------------------------------------- Agent Card

def test_agent_card_is_public_and_complete(a2a):
    response = a2a.get("/.well-known/agent-card.json")  # no token
    assert response.status_code == 200
    card = response.json()

    assert card["name"] == "Household Object Recognition Agent"
    assert card["version"]
    assert card["supportedInterfaces"] == [
        {"url": "http://agent.example.ts.net:9100/a2a", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"},
        {"url": "http://agent.example.ts.net:9100/a2a", "protocolBinding": "JSONRPC", "protocolVersion": "0.3"},
    ]
    assert card["capabilities"]["streaming"] is True
    assert "image/jpeg" in card["defaultInputModes"]
    assert "application/json" in card["defaultOutputModes"]
    skill = card["skills"][0]
    assert skill["id"] == "recognize-household-objects"
    assert set(skill["inputModes"]) == {"image/jpeg", "image/png", "image/webp"}
    # Bearer auth is advertised so callers know how to authenticate.
    assert card["securitySchemes"]["bearer"]["httpAuthSecurityScheme"]["scheme"] == "Bearer"
    assert card["securityRequirements"] == [{"schemes": {"bearer": {}}}]
    # v0.3 compatibility fields for agents on the older SDK.
    assert card["url"] == "http://agent.example.ts.net:9100/a2a"


def test_agent_card_without_token_declares_no_security():
    with TestClient(make_app(token="", public_url="")) as client:
        card = client.get("/.well-known/agent-card.json").json()
    assert "securitySchemes" not in card
    assert card["supportedInterfaces"][0]["url"] == "http://127.0.0.1:9100/a2a"


# --------------------------------------------------------------------------- valid requests

def test_valid_image_returns_completed_task_with_real_detections(a2a, detector, fallback):
    response = rpc(a2a, [image_part()])
    assert response.status_code == 200
    task = response.json()["result"]["task"]

    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert "chair (91%)" in task["status"]["message"]["parts"][0]["text"]
    artifact = task["artifacts"][0]
    assert artifact["name"] == "detections"
    data = artifact["parts"][0]["data"]
    assert artifact["parts"][0]["mediaType"] == "application/json"
    assert data["source"] == "local"
    assert data["objects"] == [
        {"label": "chair", "confidence": 0.91, "box": [1.0, 2.0, 30.0, 40.0]},
        {"label": "tv", "confidence": 0.82, "box": None},
    ]
    assert detector.calls == 1
    assert fallback.calls == 0


def test_image_bytes_are_not_echoed_in_task_history(a2a):
    data = make_image_bytes()
    task = rpc(a2a, [image_part(data)]).json()["result"]["task"]
    history = json.dumps(task.get("history", []))
    assert base64.b64encode(data).decode()[:40] not in history
    assert "image omitted from history" in history


def test_optional_text_part_is_accepted(a2a):
    task = rpc(a2a, [{"text": "What is in this photo?"}, image_part()]).json()["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"


def test_png_and_data_url_inputs(a2a):
    png = rpc(a2a, [image_part(make_image_bytes("PNG"), "image/png", "a.png")]).json()["result"]["task"]
    assert png["status"]["state"] == "TASK_STATE_COMPLETED"

    data_url = "data:image/webp;base64," + base64.b64encode(make_image_bytes("WEBP")).decode()
    task = rpc(a2a, [{"url": data_url}]).json()["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"


def test_low_confidence_uses_configured_fallback(a2a, detector, fallback):
    detector.detections = [RawDetection("sofa", 0.40)]
    data = rpc(a2a, [image_part()]).json()["result"]["task"]["artifacts"][0]["parts"][0]["data"]
    assert data["source"] == "openai"
    assert data["fallback_used"] is True
    assert data["local_candidates"] == [{"label": "sofa", "confidence": 0.4, "box": None}]
    assert fallback.calls == 1


def test_task_can_be_fetched_after_completion(a2a):
    task_id = rpc(a2a, [image_part()]).json()["result"]["task"]["id"]
    body = {"jsonrpc": "2.0", "id": 2, "method": "GetTask", "params": {"id": task_id}}
    task = a2a.post("/a2a", json=body, headers=AUTH).json()["result"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert task["artifacts"][0]["parts"][0]["data"]["objects"][0]["label"] == "chair"


def test_streaming_reports_lifecycle(a2a):
    response = rpc(a2a, [image_part()], method="SendStreamingMessage")
    assert response.status_code == 200
    events = [json.loads(line[len("data:"):]) for line in response.text.splitlines() if line.startswith("data:")]
    results = [e["result"] for e in events]
    states = [r["statusUpdate"]["status"]["state"] for r in results if "statusUpdate" in r]
    assert results[0]["task"]["status"]["state"] == "TASK_STATE_SUBMITTED"
    assert states == ["TASK_STATE_WORKING", "TASK_STATE_COMPLETED"]
    assert any("artifactUpdate" in r for r in results)


def test_v0_3_client_message_send_is_supported(a2a):
    body = {
        "jsonrpc": "2.0", "id": 1, "method": "message/send",
        "params": {"message": {
            "kind": "message", "messageId": "m-03", "role": "user",
            "parts": [{"kind": "file", "file": {
                "bytes": base64.b64encode(make_image_bytes()).decode(), "mimeType": "image/jpeg", "name": "a.jpg",
            }}],
        }},
    }
    result = a2a.post("/a2a", json=body, headers={"Authorization": f"Bearer {TOKEN}"}).json()["result"]
    assert result["status"]["state"] == "completed"
    assert result["artifacts"][0]["parts"][0]["data"]["objects"][0]["label"] == "chair"


# --------------------------------------------------------------------------- invalid input

def test_unsupported_media_type_is_rejected(a2a, detector):
    error = rpc_error(rpc(a2a, [image_part(b"GIF89a....", "image/gif", "x.gif")]))
    assert error["code"] == CONTENT_TYPE_NOT_SUPPORTED
    assert detector.calls == 0


def test_mislabelled_unsupported_format_is_rejected(a2a, detector):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="BMP")
    error = rpc_error(rpc(a2a, [image_part(buffer.getvalue(), "image/png", "fake.png")]))
    assert error["code"] == CONTENT_TYPE_NOT_SUPPORTED
    assert detector.calls == 0


@pytest.mark.parametrize(
    ("parts", "expected"),
    [
        ([image_part(b"definitely not a jpeg")], "The file could not be read as a valid image."),
        ([{"text": "no image here"}], "No image found"),
        ([image_part(), image_part()], "exactly one image"),
        ([{"url": "https://example.com/cat.jpg", "mediaType": "image/jpeg"}], "not fetched"),
        ([{"url": "data:image/png;base64,@@@"}], "not valid base64"),
    ],
)
def test_invalid_requests_return_invalid_params(a2a, detector, parts, expected):
    error = rpc_error(rpc(a2a, parts))
    assert error["code"] == INVALID_PARAMS
    assert expected in error["message"]
    assert detector.calls == 0


def test_empty_image_bytes_are_rejected(a2a):
    error = rpc_error(rpc(a2a, [image_part(b"")]))
    assert error["code"] == INVALID_PARAMS


def test_oversized_image_is_rejected():
    settings = Settings(openai_api_key="k", max_upload_mb=0.0005)  # ~524 bytes
    detector = FakeDetector()
    with TestClient(make_app(detector, settings=settings)) as client:
        big = make_image_bytes("PNG") + b"\0" * 2000
        response = rpc(client, [image_part(big, "image/png", "big.png")])
    # Either the transport cap (413) or the image validator (InvalidParams) must refuse it.
    if response.status_code == 413:
        assert response.json()["code"] == "file_too_large"
    else:
        assert rpc_error(response)["code"] == INVALID_PARAMS
    assert detector.calls == 0


def test_body_above_transport_cap_is_refused_with_413(a2a):
    huge = "A" * (20 * 1024 * 1024)  # far above 10 MB of base64
    response = a2a.post("/a2a", content=huge, headers={**AUTH, "Content-Type": "application/json"})
    assert response.status_code == 413


# --------------------------------------------------------------------------- inference failures

def test_inference_failure_without_fallback_fails_the_task():
    detector = CrashingDetector(LocalInferenceError("Local object detection failed."))
    with TestClient(make_app(detector, NotConfiguredFallback())) as client:
        task = rpc(client, [image_part()]).json()["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_FAILED"
    assert "fallback_not_configured" in task["status"]["message"]["parts"][0]["text"]
    assert not task.get("artifacts")  # no fabricated detections


def test_fallback_error_fails_the_task():
    detector = FakeDetector([RawDetection("sofa", 0.30)])
    with TestClient(make_app(detector, FakeFallback(fail=True))) as client:
        task = rpc(client, [image_part()]).json()["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_FAILED"
    assert "fallback_failed" in task["status"]["message"]["parts"][0]["text"]
    assert not task.get("artifacts")


def test_unexpected_error_fails_the_task_without_leaking_details():
    with TestClient(make_app(CrashingDetector(RuntimeError("secret internal path C:\\x")))) as client:
        task = rpc(client, [image_part()]).json()["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_FAILED"
    text = task["status"]["message"]["parts"][0]["text"]
    assert "internal_error" in text
    assert "secret" not in text
    assert not task.get("artifacts")


# --------------------------------------------------------------------------- auth + config

@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong-token-0000000000"}, {"Authorization": TOKEN}])
def test_requests_without_valid_token_are_refused(a2a, detector, headers):
    response = rpc(a2a, [image_part()], headers={**headers, "A2A-Version": "1.0"})
    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith("Bearer")
    assert detector.calls == 0


def test_health_is_public_and_reports_model_state(a2a):
    body = a2a.get("/health").json()
    assert body["status"] == "ok"
    assert body["local_model_loaded"] is True
    assert body["auth_required"] is True


def test_network_bind_requires_token():
    with pytest.raises(ValueError, match="A2A_AUTH_TOKEN must be set"):
        A2ASettings(host="0.0.0.0").validate()
    with pytest.raises(ValueError, match="A2A_AUTH_TOKEN must be set"):
        A2ASettings(host="100.81.162.78").validate()
    with pytest.raises(ValueError, match="at least 16"):
        A2ASettings(host="0.0.0.0", auth_token="short").validate()
    A2ASettings(host="127.0.0.1").validate()  # loopback without a token is allowed
    A2ASettings(host="0.0.0.0", auth_token=TOKEN).validate()


def test_rejections_log_one_line_without_sdk_traceback(a2a, caplog):
    rpc(a2a, [image_part(b"definitely not a jpeg")])
    rejected = [r for r in caplog.records if "rejected" in r.getMessage()]
    assert rejected and rejected[0].levelname == "WARNING"
    assert "invalid_image" in rejected[0].getMessage()
    assert not [r for r in caplog.records if r.name.startswith("a2a.") and r.levelname == "ERROR"]


def test_agent_disables_ultralytics_autoinstall():
    import os

    assert os.environ["YOLO_AUTOINSTALL"] == "false"


def test_create_app_refuses_unsafe_config():
    with pytest.raises(ValueError):
        create_a2a_app(Settings(), A2ASettings(host="0.0.0.0"), FakeDetector(), FakeFallback())


# --------------------------------------------------------------------------- SDK client end-to-end

@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize("streaming", [False, True])
async def test_official_sdk_client_round_trip(detector, fallback, streaming):
    app = make_app(detector, fallback)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://agent.example.ts.net:9100",
                                 headers={"Authorization": f"Bearer {TOKEN}"}) as http:
        factory = ClientFactory(ClientConfig(streaming=streaming, httpx_client=http))
        client = await factory.create_from_url("http://agent.example.ts.net:9100")
        message = Message(role=Role.ROLE_USER, message_id=str(uuid.uuid4()),
                          parts=[Part(raw=make_image_bytes(), media_type="image/jpeg", filename="a.jpg")])
        events = [event async for event in client.send_message(SendMessageRequest(message=message))]

    states = [e.status_update.status.state for e in events if e.HasField("status_update")]
    tasks = [e.task for e in events if e.HasField("task")]
    if streaming:
        assert states[-1] == TaskState.TASK_STATE_COMPLETED
    else:
        assert tasks[-1].status.state == TaskState.TASK_STATE_COMPLETED
        assert get_data_parts(tasks[-1].artifacts[0].parts)[0]["objects"][0]["label"] == "chair"
    assert detector.calls == 1

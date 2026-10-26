"""A2A test client for the Household Object Recognition agent (or any A2A agent).

Standalone: needs only `pip install "a2a-sdk[http-server]" httpx`, so it can be copied
to another machine (e.g. a mentor's laptop) and pointed at this agent over the Tailnet.

Usage (from backend/, with the venv active):
  python scripts/a2a_client.py card                       # fetch + validate the Agent Card
  python scripts/a2a_client.py send photo.jpg             # recognize objects in one image
  python scripts/a2a_client.py selftest photo.jpg         # run the full check list (PASS/FAIL)
  python scripts/a2a_client.py ask "hello"                # send a text message to ANY A2A agent

Target and credentials (flags override environment variables):
  --url    / A2A_REMOTE_URL     base URL of the agent, e.g. https://narmatha.tail2e5e01.ts.net
                                (default http://127.0.0.1:9100)
  --token  / A2A_REMOTE_TOKEN   bearer token for that agent (falls back to A2A_AUTH_TOKEN)
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import mimetypes
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import get_data_parts, get_text_parts
from a2a.types import (
    AgentCard,
    GetTaskRequest,
    Message,
    Part,
    Role,
    SendMessageRequest,
    StreamResponse,
    Task,
    TaskState,
)
from a2a.utils.errors import A2AError
from google.protobuf.json_format import MessageToDict

try:  # Optional: lets the script load backend/.env when run inside this project.
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

LOOPBACK = {"127.0.0.1", "localhost", "::1"}
DEFAULT_URL = "http://127.0.0.1:9100"


# --------------------------------------------------------------------------- helpers

def http_client(token: str | None, timeout: float) -> httpx.AsyncClient:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return httpx.AsyncClient(headers=headers, timeout=timeout)


async def fetch_card(base_url: str, token: str | None, timeout: float) -> AgentCard:
    async with http_client(token, timeout) as client:
        return await A2ACardResolver(client, base_url).get_agent_card()


def validate_card(card: AgentCard, base_url: str) -> list[str]:
    """Problems that would stop a remote agent from using this card. Empty list = OK."""
    problems = []
    if not card.name:
        problems.append("card has no name")
    if not card.supported_interfaces:
        problems.append("card has no supportedInterfaces (no endpoint URL)")
    if not card.skills:
        problems.append("card declares no skills")
    caller_is_remote = (urlparse(base_url).hostname or "") not in LOOPBACK
    for interface in card.supported_interfaces:
        host = urlparse(interface.url).hostname or ""
        if caller_is_remote and host in LOOPBACK:
            problems.append(
                f"card advertises {interface.url}, which a remote machine cannot reach; "
                "set A2A_PUBLIC_URL on the agent"
            )
    return problems


def image_message(data: bytes, media_type: str, filename: str, text: str | None = None) -> Message:
    parts = [Part(raw=data, media_type=media_type, filename=filename)]
    if text:
        parts.insert(0, Part(text=text))
    return Message(role=Role.ROLE_USER, message_id=str(uuid.uuid4()), parts=parts)


async def send(base_url: str, token: str | None, message: Message, stream: bool, timeout: float) -> Task | Message:
    """Send one message via the official SDK client and return the final Task (or a direct Message)."""
    async with http_client(token, timeout) as httpx_client:
        config = ClientConfig(streaming=stream, httpx_client=httpx_client)
        client = await create_client(base_url, client_config=config)
        request = SendMessageRequest(message=message)
        request.configuration.history_length = 0  # don't echo our own message back
        final: Task | Message | None = None
        async for event in client.send_message(request):
            final = _absorb(event, final, verbose=stream)
        if final is None:
            raise RuntimeError("The agent returned no task or message.")
        if isinstance(final, Task) and stream:
            # Streaming events are partial; fetch the stored task for the complete result.
            final = await client.get_task(GetTaskRequest(id=final.id))
        return final


def _absorb(event: StreamResponse, current: Task | Message | None, verbose: bool) -> Task | Message | None:
    kind = event.WhichOneof("payload")
    if kind == "task":
        return event.task
    if kind == "message":
        return event.message
    if verbose and kind == "status_update":
        print(f"  [stream] status -> {TaskState.Name(event.status_update.status.state)}")
    if verbose and kind == "artifact_update":
        print(f"  [stream] artifact '{event.artifact_update.artifact.name}' received")
    if isinstance(current, Task) or current is None:
        task = Task() if current is None else current
        if kind == "status_update":
            task.id = event.status_update.task_id
            task.status.CopyFrom(event.status_update.status)
        return task
    return current


def detections_of(task: Task) -> list[dict]:
    for artifact in task.artifacts:
        for data in get_data_parts(artifact.parts):
            if isinstance(data, dict) and "objects" in data:
                return data["objects"]
    return []


def status_text(task: Task) -> str:
    return " ".join(get_text_parts(task.status.message.parts)) if task.status.HasField("message") else ""


def describe_error(exc: Exception) -> str:
    if isinstance(exc, A2AError):
        code = (exc.data or {}).get("code") if isinstance(exc.data, dict) else None
        return f"{type(exc).__name__}: {exc.message}" + (f" (code={code})" if code else "")
    return f"{type(exc).__name__}: {exc}"


def read_image(path: str) -> tuple[bytes, str, str]:
    file = Path(path)
    media_type = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
    return file.read_bytes(), media_type, file.name


# --------------------------------------------------------------------------- commands

async def cmd_card(args) -> int:
    card = await fetch_card(args.url, args.token, args.timeout)
    print(json.dumps(MessageToDict(card), indent=2))
    problems = validate_card(card, args.url)
    for problem in problems:
        print(f"PROBLEM: {problem}", file=sys.stderr)
    print("Agent Card OK" if not problems else "Agent Card has problems", file=sys.stderr)
    return 1 if problems else 0


async def cmd_send(args) -> int:
    data, media_type, filename = read_image(args.image)
    message = image_message(data, args.media_type or media_type, filename, args.text)
    try:
        result = await send(args.url, args.token, message, args.stream, args.timeout)
    except Exception as exc:  # noqa: BLE001 - CLI: report any failure clearly
        print(f"Request failed: {describe_error(exc)}", file=sys.stderr)
        return 1
    if isinstance(result, Message):
        print("Agent replied with a message:", " ".join(get_text_parts(result.parts)))
        return 0
    state = TaskState.Name(result.status.state)
    print(f"Task {result.id}: {state}")
    if status_text(result):
        print(f"  {status_text(result)}")
    for obj in detections_of(result):
        print(f"  - {obj['label']:<18} confidence={obj.get('confidence')}  box={obj.get('box')}")
    if args.json:
        print(json.dumps(MessageToDict(result), indent=2))
    return 0 if result.status.state == TaskState.TASK_STATE_COMPLETED else 1


async def cmd_ask(args) -> int:
    message = Message(role=Role.ROLE_USER, message_id=str(uuid.uuid4()), parts=[Part(text=args.text)])
    try:
        result = await send(args.url, args.token, message, args.stream, args.timeout)
    except Exception as exc:  # noqa: BLE001
        print(f"Request failed: {describe_error(exc)}", file=sys.stderr)
        return 1
    print(json.dumps(MessageToDict(result), indent=2))
    return 0


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


async def cmd_selftest(args) -> int:
    """The communication checklist. Prints PASS/FAIL per check; exit code 1 if any check failed."""
    checks: list[Check] = []

    async def expect_rejected(name: str, message: Message) -> None:
        try:
            result = await send(args.url, args.token, message, False, args.timeout)
        except A2AError as exc:
            checks.append(Check(name, True, f"rejected: {describe_error(exc)}"))
            return
        except Exception as exc:  # noqa: BLE001
            checks.append(Check(name, False, f"unexpected failure: {describe_error(exc)}"))
            return
        state = TaskState.Name(result.status.state) if isinstance(result, Task) else "message"
        checks.append(Check(name, state in ("TASK_STATE_FAILED", "TASK_STATE_REJECTED"), f"got {state}"))

    # 1. Agent Card retrieval and validation.
    try:
        card = await fetch_card(args.url, args.token, args.timeout)
        problems = validate_card(card, args.url)
        checks.append(Check("agent card", not problems,
                            "; ".join(problems) or f"'{card.name}' -> {card.supported_interfaces[0].url}"))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("agent card", False, describe_error(exc)))
        card = None

    # 2 + 3. Valid image -> completed task with real labels and confidence scores.
    data, media_type, filename = read_image(args.image)
    try:
        result = await send(args.url, args.token, image_message(data, media_type, filename), False, args.timeout)
        completed = isinstance(result, Task) and result.status.state == TaskState.TASK_STATE_COMPLETED
        objects = detections_of(result) if isinstance(result, Task) else []
        checks.append(Check("valid image -> completed", completed,
                            TaskState.Name(result.status.state) + (f": {status_text(result)}" if not completed else "")
                            if isinstance(result, Task) else "agent returned a message, not a task"))
        has_scores = bool(objects) and all("label" in o and isinstance(o.get("confidence"), (int, float))
                                           for o in objects)
        checks.append(Check("labels + confidence scores", has_scores,
                            ", ".join(f"{o['label']} {o['confidence']:.2f}" for o in objects) or "no objects returned"))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("valid image -> completed", False, describe_error(exc)))

    # 4. Invalid / unsupported input.
    await expect_rejected("unsupported type (image/gif)", image_message(b"GIF89a fake", "image/gif", "x.gif"))
    await expect_rejected("corrupt image bytes", image_message(b"not really a jpeg", "image/jpeg", "bad.jpg"))
    await expect_rejected("text only, no image",
                          Message(role=Role.ROLE_USER, message_id=str(uuid.uuid4()), parts=[Part(text="hi")]))
    try:
        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buffer, format="BMP")
        await expect_rejected("unsupported format sniffed (BMP labelled png)",
                              image_message(buffer.getvalue(), "image/png", "fake.png"))
    except ImportError:
        pass

    # Auth: when the card declares a security scheme, a request without the token must be refused.
    if card is not None and card.security_requirements:
        try:
            await send(args.url, None, image_message(data, media_type, filename), False, args.timeout)
            checks.append(Check("request without token refused", False, "agent accepted an unauthenticated request"))
        except Exception as exc:  # noqa: BLE001
            checks.append(Check("request without token refused", "401" in str(exc) or "nauthorized" in str(exc),
                                describe_error(exc)))

    width = max(len(c.name) for c in checks)
    print(f"\nA2A self-test against {args.url}")
    for check in checks:
        print(f"  {'PASS' if check.passed else 'FAIL'}  {check.name:<{width}}  {check.detail}")
    failed = [c for c in checks if not c.passed]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed.")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.getenv("A2A_REMOTE_URL") or DEFAULT_URL, help="agent base URL")
    parser.add_argument("--token", default=os.getenv("A2A_REMOTE_TOKEN") or os.getenv("A2A_AUTH_TOKEN") or None,
                        help="bearer token for the agent")
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds per request")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("card", help="fetch and validate the Agent Card")

    send_parser = sub.add_parser("send", help="send one image for recognition")
    send_parser.add_argument("image")
    send_parser.add_argument("--text", help="optional text part sent with the image")
    send_parser.add_argument("--media-type", help="override the detected media type")
    send_parser.add_argument("--stream", action="store_true", help="use SendStreamingMessage (SSE)")
    send_parser.add_argument("--json", action="store_true", help="print the full task JSON")

    ask_parser = sub.add_parser("ask", help="send a text message to any A2A agent")
    ask_parser.add_argument("text")
    ask_parser.add_argument("--stream", action="store_true")

    selftest_parser = sub.add_parser("selftest", help="run the communication checklist")
    selftest_parser.add_argument("image", help="a real photo containing a household object")

    args = parser.parse_args()
    args.url = args.url.rstrip("/")
    handler = {"card": cmd_card, "send": cmd_send, "ask": cmd_ask, "selftest": cmd_selftest}[args.command]
    return asyncio.run(handler(args))


if __name__ == "__main__":
    sys.exit(main())

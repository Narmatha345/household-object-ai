"""A2A AgentExecutor that delegates to the existing detection pipeline.

Lifecycle of one request:
  invalid input             -> JSON-RPC error (InvalidParams / ContentTypeNotSupported), no task created
  valid image               -> task SUBMITTED -> WORKING -> COMPLETED with a "detections" artifact
  inference/fallback error  -> task FAILED with a user-safe reason, no artifact
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import time
from dataclasses import dataclass
from pathlib import PurePath

from a2a.helpers import new_data_part, new_task_from_user_message, new_text_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Message
from a2a.utils.errors import ContentTypeNotSupportedError, InvalidParamsError, TaskNotCancelableError
from PIL import Image

from app.errors import AppError, UnsupportedFileTypeError
from app.schemas import DetectionResponse
from app.services.detection_router import DetectionRouter
from app.services.image_input import ALLOWED_CONTENT_TYPES, ALLOWED_EXTENSIONS, decode_image

logger = logging.getLogger("household_object_ai.a2a")

ARTIFACT_NAME = "detections"
# Expected client mistakes, returned to the caller as JSON-RPC errors.
REJECTION_ERRORS = (InvalidParamsError, ContentTypeNotSupportedError)


class ExpectedRejectionFilter(logging.Filter):
    """Drop the SDK's ERROR tracebacks for input we rejected on purpose (already logged as one line)."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not (record.exc_info and isinstance(record.exc_info[1], REJECTION_ERRORS))


@dataclass(frozen=True)
class ImageInput:
    data: bytes
    media_type: str
    filename: str


def _decode_data_url(url: str) -> tuple[bytes, str]:
    """Inline `data:image/png;base64,...` URLs. Remote http(s) URLs are never fetched."""
    header, _, payload = url.partition(",")
    if not header.endswith(";base64"):
        raise InvalidParamsError("Data URLs must be base64-encoded.", data={"code": "invalid_request"})
    media_type = header[len("data:"):-len(";base64")].lower()
    try:
        return base64.b64decode(payload, validate=True), media_type
    except (binascii.Error, ValueError) as exc:
        raise InvalidParamsError("The data URL is not valid base64.", data={"code": "invalid_request"}) from exc


def extract_image(message: Message | None) -> ImageInput:
    """Find exactly one image in the message parts. Text parts are allowed and ignored."""
    if message is None or not message.parts:
        raise InvalidParamsError("The message has no parts. Attach one image.", data={"code": "invalid_request"})

    images: list[ImageInput] = []
    for part in message.parts:
        kind = part.WhichOneof("content")
        media_type = (part.media_type or "").split(";")[0].strip().lower()
        if kind == "raw":
            images.append(ImageInput(part.raw, media_type, part.filename))
        elif kind == "url":
            if not part.url.startswith("data:"):
                raise InvalidParamsError(
                    "Image URLs are not fetched. Send the image bytes inline in a file part ('raw').",
                    data={"code": "url_not_supported"},
                )
            data, url_type = _decode_data_url(part.url)
            images.append(ImageInput(data, media_type or url_type, part.filename))
        elif kind == "data":
            raise ContentTypeNotSupportedError(
                "Structured data parts are not supported. Attach a JPEG, PNG or WEBP image.",
                data={"code": "unsupported_file_type"},
            )
        # kind == "text": optional instruction, ignored by this skill.

    if not images:
        raise InvalidParamsError("No image found in the message. Attach one JPEG, PNG or WEBP image.",
                                 data={"code": "missing_image"})
    if len(images) > 1:
        raise InvalidParamsError("Send exactly one image per request.", data={"code": "too_many_images"})

    image = images[0]
    extension = PurePath(image.filename).suffix.lower()
    # A declared type or extension must be an allowed image type; the bytes are sniffed again in decode_image.
    declared = [x for x in (image.media_type, extension) if x]
    if declared and image.media_type not in ALLOWED_CONTENT_TYPES and extension not in ALLOWED_EXTENSIONS:
        raise ContentTypeNotSupportedError(
            f"Unsupported image type {image.media_type or extension!r}. Send a JPEG, PNG or WEBP image.",
            data={"code": "unsupported_file_type"},
        )
    return image


def decode(image_input: ImageInput, max_bytes: int) -> Image.Image:
    """Reuse the web API's validation (size limit, format sniffing, EXIF rotation)."""
    try:
        return decode_image(image_input.data, max_bytes)
    except UnsupportedFileTypeError as exc:
        raise ContentTypeNotSupportedError(exc.message, data={"code": exc.code}) from exc
    except AppError as exc:
        raise InvalidParamsError(exc.message, data={"code": exc.code}) from exc


def without_image_bytes(message: Message) -> Message:
    """Copy of the message for task history, with image bytes replaced by a placeholder.

    Keeps the in-memory task store small and avoids echoing the image back to the caller.
    """
    redacted = Message()
    redacted.CopyFrom(message)
    for part in redacted.parts:
        kind = part.WhichOneof("content")
        if kind == "raw" or (kind == "url" and part.url.startswith("data:")):
            size = len(part.raw) if kind == "raw" else len(part.url)
            part.text = f"[image omitted from history: {size} bytes, {part.media_type or 'unspecified type'}]"
            part.media_type = "text/plain"
    return redacted


def summarize(result: DetectionResponse) -> str:
    if not result.objects:
        return "No household objects were detected."
    items = ", ".join(
        f"{o.label} ({o.confidence:.0%})" if o.confidence is not None else o.label for o in result.objects
    )
    return f"Detected {len(result.objects)} object(s) [source: {result.source}]: {items}"


class ObjectRecognitionExecutor(AgentExecutor):
    def __init__(self, detection_router: DetectionRouter, max_image_bytes: int) -> None:
        self._router = detection_router
        self._max_bytes = max_image_bytes

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        started = time.perf_counter()
        message = context.message

        # Validate before creating a task: bad input becomes a JSON-RPC error, not a task.
        try:
            image_input = extract_image(message)
            image = await asyncio.to_thread(decode, image_input, self._max_bytes)
        except REJECTION_ERRORS as exc:
            code = exc.data.get("code") if isinstance(exc.data, dict) else None
            logger.warning("A2A task=%s rejected: %s code=%s: %s", context.task_id, type(exc).__name__, code, exc.message)
            raise
        logger.info(
            "A2A task=%s context=%s image accepted: bytes=%d type=%s size=%dx%d",
            context.task_id, context.context_id, len(image_input.data),
            image_input.media_type or "unspecified", image.width, image.height,
        )

        task = context.current_task or new_task_from_user_message(without_image_bytes(message))
        await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work(new_text_message("Running household object detection.",
                                                  context_id=task.context_id, task_id=task.id))
        logger.info("A2A task=%s state=working", task.id)

        try:
            # YOLO inference (and the optional OpenAI call) is blocking; keep the event loop free.
            result = await asyncio.to_thread(self._router.detect, image, started)
        except AppError as exc:
            logger.warning("A2A task=%s state=failed code=%s: %s", task.id, exc.code, exc.message)
            await updater.failed(new_text_message(f"Detection failed ({exc.code}): {exc.message}",
                                                  context_id=task.context_id, task_id=task.id))
            return
        except Exception:
            logger.exception("A2A task=%s state=failed: unexpected inference error", task.id)
            await updater.failed(new_text_message("Detection failed (internal_error): an unexpected error occurred.",
                                                  context_id=task.context_id, task_id=task.id))
            return

        summary = summarize(result)
        await updater.add_artifact(
            [
                new_data_part(result.model_dump(mode="json"), media_type="application/json"),
                new_text_part(summary, media_type="text/plain"),
            ],
            name=ARTIFACT_NAME,
        )
        await updater.complete(new_text_message(summary, context_id=task.context_id, task_id=task.id))
        logger.info(
            "A2A task=%s state=completed source=%s objects=[%s] total_ms=%.1f",
            task.id, result.source,
            ", ".join(f"{o.label}:{o.confidence}" for o in result.objects),
            (time.perf_counter() - started) * 1000,
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Detection is a single short blocking call; there is nothing to interrupt.
        raise TaskNotCancelableError("Object detection tasks cannot be canceled.")

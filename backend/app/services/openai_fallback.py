"""OpenAI vision fallback, used only when the local model is not reliable.

The API key is read on the backend only and never sent to the frontend.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Protocol

from PIL import Image
from pydantic import BaseModel, Field

from app.errors import FallbackError, FallbackNotConfiguredError
from app.schemas import DetectedObject

logger = logging.getLogger(__name__)

PROMPT = """You are the fallback detector for a household object recognition app.
A local on-device model could not confidently identify objects in this image.

Identify the common household objects clearly visible in the image
(furniture, appliances, electronics, kitchenware, decor, personal items, etc.).
Use short lowercase common names (e.g. "chair", "ceiling fan", "pressure cooker").
List the most prominent object first. Give each object a confidence from 0 to 1.
If no household object is visible, return an empty objects list and say so in
the description.

Return JSON with:
- objects: list of {name, confidence}
- description: one short sentence describing the scene
- reliability: one short sentence explaining how certain you are and why"""


class _FallbackObject(BaseModel):
    name: str
    confidence: float = Field(ge=0.0, le=1.0)


class _FallbackPayload(BaseModel):
    objects: list[_FallbackObject]
    description: str
    reliability: str


class FallbackResult(BaseModel):
    objects: list[DetectedObject]
    description: str
    reliability_note: str


class FallbackDetector(Protocol):
    @property
    def is_configured(self) -> bool: ...

    def detect(self, image: Image.Image, local_hints: list[str] | None = None) -> FallbackResult: ...


def encode_image_for_upload(image: Image.Image, max_side: int) -> str:
    """Downscale and JPEG-encode the image to keep request size and cost low."""
    img = image.copy()
    img.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class OpenAIFallbackService:
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: float = 30.0,
        image_max_side: int = 1024,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_seconds
        self._image_max_side = image_max_side
        self._client = None

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key, timeout=self._timeout, max_retries=1)
        return self._client

    def detect(self, image: Image.Image, local_hints: list[str] | None = None) -> FallbackResult:
        if not self.is_configured:
            raise FallbackNotConfiguredError(
                "The local model was not confident and the OpenAI fallback is not configured."
            )

        prompt = PROMPT
        if local_hints:
            prompt += (
                "\n\nThe local model made these low-confidence guesses, which may be wrong: "
                + ", ".join(local_hints)
                + ". Verify or correct them."
            )

        try:
            response = self._get_client().responses.parse(
                model=self._model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_url": encode_image_for_upload(image, self._image_max_side),
                                "detail": "auto",
                            },
                        ],
                    }
                ],
                text_format=_FallbackPayload,
            )
            payload = response.output_parsed
        except Exception as exc:
            # Log the real error server-side; return a generic message to clients.
            logger.exception("OpenAI fallback request failed")
            raise FallbackError("The AI fallback service is currently unavailable. Please try again.") from exc

        if payload is None:
            raise FallbackError("The AI fallback service returned an unreadable response.")

        return FallbackResult(
            objects=[
                DetectedObject(label=obj.name.strip().lower(), confidence=round(obj.confidence, 4))
                for obj in payload.objects
                if obj.name.strip()
            ],
            description=payload.description,
            reliability_note=payload.reliability,
        )

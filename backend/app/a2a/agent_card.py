"""The Agent Card: the public description other agents fetch before calling this agent."""

from __future__ import annotations

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    HTTPAuthSecurityScheme,
    SecurityRequirement,
    SecurityScheme,
    StringList,
)
from a2a.utils.constants import PROTOCOL_VERSION_0_3, PROTOCOL_VERSION_1_0, TransportProtocol

from app.a2a.config import A2ASettings
from app.config import Settings

AGENT_VERSION = "0.1.0"
JSONRPC_PATH = "/a2a"
SKILL_ID = "recognize-household-objects"
IMAGE_INPUT_MODES = ["image/jpeg", "image/png", "image/webp"]
# An optional text part (e.g. "what is in this photo?") is accepted alongside the image and ignored.
INPUT_MODES = [*IMAGE_INPUT_MODES, "text/plain"]
OUTPUT_MODES = ["application/json", "text/plain"]
BEARER_SCHEME = "bearer"


def build_agent_card(a2a_settings: A2ASettings, settings: Settings) -> AgentCard:
    labels = ", ".join(settings.household_labels) if settings.household_labels else "any label the model knows"
    skill = AgentSkill(
        id=SKILL_ID,
        name="Household object recognition",
        description=(
            "Detects household objects in one photo using a local YOLO model "
            f"(confidence threshold {settings.local_confidence_threshold:.0%}). "
            "Send exactly one JPEG, PNG or WEBP image as a file part with inline bytes, "
            f"max {settings.max_upload_mb:g} MB. Returns every detected object with its label, "
            "confidence score (0-1) and bounding box. Low-confidence images go to an OpenAI "
            "fallback only when the server has one configured; otherwise the task fails. "
            f"Known classes: {labels}."
        ),
        tags=["vision", "object-detection", "household", "yolo", "image"],
        examples=[
            "Attach a photo of a living room: returns e.g. sofa 0.91, tv 0.84, table 0.78",
            "Attach a kitchen photo: returns e.g. refrigerator 0.88, microwave 0.81",
        ],
        input_modes=IMAGE_INPUT_MODES,
        output_modes=OUTPUT_MODES,
    )

    card = AgentCard(
        name="Household Object Recognition Agent",
        description=(
            "Recognizes common household objects (furniture, appliances, electronics) in photos. "
            "Wraps the Household Object AI local-first YOLO detector."
        ),
        version=AGENT_VERSION,
        # One JSON-RPC endpoint speaking both protocol versions (see enable_v0_3_compat in server.py).
        # Declaring 0.3 also makes the SDK emit the legacy `url`/`preferredTransport` card fields.
        supported_interfaces=[
            AgentInterface(
                url=f"{a2a_settings.base_url}{JSONRPC_PATH}",
                protocol_binding=TransportProtocol.JSONRPC.value,
                protocol_version=version,
            )
            for version in (PROTOCOL_VERSION_1_0, PROTOCOL_VERSION_0_3)
        ],
        capabilities=AgentCapabilities(streaming=True, push_notifications=False),
        default_input_modes=INPUT_MODES,
        default_output_modes=OUTPUT_MODES,
        skills=[skill],
    )

    if a2a_settings.auth_token:
        card.security_schemes[BEARER_SCHEME].CopyFrom(
            SecurityScheme(
                http_auth_security_scheme=HTTPAuthSecurityScheme(
                    scheme="Bearer",
                    description="Shared token issued by the agent owner. Send 'Authorization: Bearer <token>'.",
                )
            )
        )
        # Empty scope list: the scheme is required, no scopes.
        card.security_requirements.append(SecurityRequirement(schemes={BEARER_SCHEME: StringList()}))

    return card

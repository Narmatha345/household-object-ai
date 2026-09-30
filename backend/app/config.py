"""Environment-based configuration.

All tunables (threshold, model path, OpenAI settings) are read from the
environment / backend/.env so nothing is hard-coded in the ML or service code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BACKEND_DIR / ".env")

# COCO classes that are commonly found in a house. Detections outside this set
# are ignored by the local detector so that, e.g., an "airplane" false positive
# on a ceiling fan does not count as a reliable household detection.
DEFAULT_HOUSEHOLD_LABELS: tuple[str, ...] = (
    "person", "cat", "dog", "backpack", "umbrella", "handbag", "suitcase",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "bicycle",
    "sports ball", "tennis racket",
)


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number, got {raw!r}") from exc


def _list_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if raw == "*":
        return ()  # empty tuple = accept every label the model knows
    return tuple(item.strip().lower() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    openai_timeout_seconds: float = 30.0
    openai_image_max_side: int = 1024

    local_confidence_threshold: float = 0.70
    model_path: Path = BACKEND_DIR / "models" / "yolo26n.pt"
    household_labels: tuple[str, ...] = DEFAULT_HOUSEHOLD_LABELS

    max_upload_mb: float = 10.0
    cors_origins: tuple[str, ...] = field(
        default=("http://localhost:5173", "http://127.0.0.1:5173")
    )

    @property
    def openai_configured(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


def _resolve_model_path(raw: str | None) -> Path:
    if not raw:
        return Settings.model_path
    path = Path(raw)
    return path if path.is_absolute() else BACKEND_DIR / path


@lru_cache
def get_settings() -> Settings:
    threshold = _float_env("LOCAL_CONFIDENCE_THRESHOLD", 0.70)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("LOCAL_CONFIDENCE_THRESHOLD must be between 0 and 1")

    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini",
        openai_timeout_seconds=_float_env("OPENAI_TIMEOUT_SECONDS", 30.0),
        openai_image_max_side=int(_float_env("OPENAI_IMAGE_MAX_SIDE", 1024)),
        local_confidence_threshold=threshold,
        model_path=_resolve_model_path(os.getenv("MODEL_PATH")),
        household_labels=_list_env("HOUSEHOLD_LABELS", DEFAULT_HOUSEHOLD_LABELS),
        max_upload_mb=_float_env("MAX_UPLOAD_MB", 10.0),
        cors_origins=_list_env("CORS_ORIGINS", Settings().cors_origins) or ("*",),
    )

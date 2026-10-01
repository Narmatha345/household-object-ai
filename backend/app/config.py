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

# Trained household model (YOLO26n fine-tuned on data/household, V1).
DEFAULT_MODEL_PATH = BACKEND_DIR / "runs" / "household-yolo26n" / "weights" / "best.pt"

# The exact 15 classes of best.pt (model.names). Local detections with any other
# label are dropped, so stray labels such as "couch" or "potted plant" can never
# be returned as a local result.
DEFAULT_HOUSEHOLD_LABELS: tuple[str, ...] = (
    "bed", "sofa", "chair", "table", "tv", "laptop", "refrigerator", "microwave",
    "washing machine", "ceiling fan", "cupboard", "gas stove", "pressure cooker",
    "bottle", "person",
)

# Classes present in model.names that the training run had no train/val images for.
# They stay allowed (they are part of the class mapping) but are logged as unvalidated.
UNVALIDATED_LABELS: tuple[str, ...] = ("pressure cooker",)


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
    model_path: Path = DEFAULT_MODEL_PATH
    # YOLO input size (imgsz). 480 is ~40% faster than 640 on CPU.
    inference_image_size: int = 480
    # A .onnx MODEL_PATH runs on ONNX Runtime (no torch import, ~3x less memory).
    # If it fails to load, this PyTorch model is loaded instead.
    fallback_model_path: Path | None = None
    # CPU threads for inference; None = library default (all cores).
    inference_threads: int | None = None
    household_labels: tuple[str, ...] = DEFAULT_HOUSEHOLD_LABELS

    # Built React app (frontend/dist). When set and present, FastAPI serves it at "/"
    # so a single deployment hosts both the UI and the API on one origin.
    frontend_dist: Path | None = None

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


def _resolve_backend_path(raw: str | None) -> Path:
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
        model_path=_resolve_backend_path(os.getenv("MODEL_PATH")),
        inference_image_size=int(_float_env("INFERENCE_IMAGE_SIZE", 480)),
        fallback_model_path=_resolve_backend_path(os.getenv("FALLBACK_MODEL_PATH")) if os.getenv("FALLBACK_MODEL_PATH") else None,
        inference_threads=int(_float_env("INFERENCE_THREADS", 0)) or None,
        household_labels=_list_env("HOUSEHOLD_LABELS", DEFAULT_HOUSEHOLD_LABELS),
        frontend_dist=_resolve_backend_path(os.getenv("FRONTEND_DIST")) if os.getenv("FRONTEND_DIST") else None,
        max_upload_mb=_float_env("MAX_UPLOAD_MB", 10.0),
        cors_origins=_list_env("CORS_ORIGINS", Settings().cors_origins) or ("*",),
    )

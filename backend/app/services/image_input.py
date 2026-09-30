"""Validation and safe decoding of uploaded images."""

from __future__ import annotations

import io
from pathlib import PurePath

from PIL import Image, ImageOps, UnidentifiedImageError

from app.errors import EmptyUploadError, FileTooLargeError, InvalidImageError, UnsupportedFileTypeError

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


def validate_upload_metadata(filename: str | None, content_type: str | None) -> None:
    content_type = (content_type or "").split(";")[0].strip().lower()
    extension = PurePath(filename or "").suffix.lower()

    if content_type not in ALLOWED_CONTENT_TYPES and extension not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError("Unsupported file type. Please upload a JPG, PNG, or WEBP image.")


def decode_image(data: bytes, max_bytes: int) -> Image.Image:
    if not data:
        raise EmptyUploadError("The uploaded file is empty.")
    if len(data) > max_bytes:
        raise FileTooLargeError(f"The image is too large. Maximum size is {max_bytes // (1024 * 1024)} MB.")

    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            probe.verify()
        if fmt not in ALLOWED_FORMATS:
            raise UnsupportedFileTypeError("Unsupported image format. Please upload a JPG, PNG, or WEBP image.")

        image = Image.open(io.BytesIO(data))
        image.load()
        # Respect phone camera orientation before inference.
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except UnsupportedFileTypeError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError, ValueError) as exc:
        raise InvalidImageError("The file could not be read as a valid image.") from exc

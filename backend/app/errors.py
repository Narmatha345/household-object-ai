"""Application errors that map to clean, user-safe HTTP responses."""

from __future__ import annotations


class AppError(Exception):
    """Base error. `message` is safe to show to end users."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class EmptyUploadError(AppError):
    status_code = 400
    code = "empty_upload"


class UnsupportedFileTypeError(AppError):
    status_code = 415
    code = "unsupported_file_type"


class FileTooLargeError(AppError):
    status_code = 413
    code = "file_too_large"


class InvalidImageError(AppError):
    status_code = 400
    code = "invalid_image"


class ModelLoadError(AppError):
    status_code = 503
    code = "local_model_unavailable"


class LocalInferenceError(AppError):
    status_code = 500
    code = "local_inference_failed"


class FallbackNotConfiguredError(AppError):
    status_code = 503
    code = "fallback_not_configured"


class FallbackError(AppError):
    status_code = 502
    code = "fallback_failed"

from __future__ import annotations

import httpx
import openai

from app.services.openai_fallback import describe_openai_error


def _status_error(cls, status: int, error: dict):
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(status, request=request, json={"error": error})
    return cls(error["message"], response=response, body=error)


def test_no_credits_message():
    exc = _status_error(
        openai.RateLimitError,
        429,
        {"message": "You have no credits remaining.", "type": "insufficient_quota", "code": "credit_balance_exhausted"},
    )
    assert "no credits" in describe_openai_error(exc)


def test_invalid_key_message():
    exc = _status_error(
        openai.AuthenticationError,
        401,
        {"message": "Incorrect API key provided: sk-abc***", "type": "invalid_request_error", "code": "invalid_api_key"},
    )
    message = describe_openai_error(exc)
    assert "invalid" in message
    assert "sk-" not in message


def test_plain_rate_limit_message():
    exc = _status_error(openai.RateLimitError, 429, {"message": "Slow down", "type": "requests", "code": "rate_limit_exceeded"})
    assert "rate-limited" in describe_openai_error(exc)


def test_unknown_error_is_generic():
    assert "currently unavailable" in describe_openai_error(RuntimeError("boom"))

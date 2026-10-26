"""ASGI middleware for the A2A server: bearer-token auth, request size cap and access logging.

Tailnet membership is NOT treated as authorization: every non-public request
must carry the shared token when A2A_AUTH_TOKEN is set.
"""

from __future__ import annotations

import hmac
import json
import logging
import time

from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("household_object_ai.a2a")

# Discovery and liveness stay public so remote agents can find out how to authenticate.
PUBLIC_PATHS = {"/.well-known/agent-card.json", "/health"}


async def _send_json(send: Send, status: int, body: dict, headers: list[tuple[bytes, bytes]] | None = None) -> None:
    payload = json.dumps(body).encode()
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode()),
                    *(headers or [])],
    })
    await send({"type": "http.response.body", "body": payload})


class _BodyTooLarge(Exception):
    pass


class A2AGuardMiddleware:
    def __init__(self, app: ASGIApp, auth_token: str, max_body_bytes: int) -> None:
        self.app = app
        self._token = auth_token.encode()
        self._max_body = max_body_bytes

    def _authorized(self, scope: Scope) -> bool:
        if not self._token:
            return True
        for name, value in scope.get("headers", []):
            if name == b"authorization":
                scheme, _, credentials = value.partition(b" ")
                return scheme.lower() == b"bearer" and hmac.compare_digest(credentials.strip(), self._token)
        return False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        path, method = scope["path"], scope["method"]
        client = scope.get("client")
        client_ip = client[0] if client else "?"
        # Behind `tailscale serve` the socket peer is 127.0.0.1; the real Tailnet caller is forwarded.
        forwarded = next((v.decode("latin-1") for k, v in scope.get("headers", []) if k == b"x-forwarded-for"), None)
        if forwarded:
            client_ip = f"{client_ip} (forwarded for {forwarded.split(',')[0].strip()})"
        status_holder = {"status": 0}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
            await send(message)

        def log() -> None:
            # Never logs bodies, headers or tokens.
            logger.info("A2A HTTP %s %s from %s -> %d (%.1f ms)", method, path, client_ip,
                        status_holder["status"], (time.perf_counter() - started) * 1000)

        if path not in PUBLIC_PATHS and not self._authorized(scope):
            await _send_json(send_wrapper, 401, {"detail": "Missing or invalid bearer token.", "code": "unauthorized"},
                             [(b"www-authenticate", b'Bearer realm="a2a"')])
            log()
            return

        declared = next((v for k, v in scope.get("headers", []) if k == b"content-length"), None)
        if declared is not None and declared.isdigit() and int(declared) > self._max_body:
            await _send_json(send_wrapper, 413, {"detail": "Request body is too large.", "code": "file_too_large"})
            log()
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self._max_body:
                    raise _BodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send_wrapper)
        except _BodyTooLarge:
            if status_holder["status"] == 0:
                await _send_json(send_wrapper, 413, {"detail": "Request body is too large.", "code": "file_too_large"})
        finally:
            log()

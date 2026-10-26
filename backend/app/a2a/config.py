"""A2A agent settings, read from the environment / backend/.env (loaded by app.config)."""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass

import app.config  # noqa: F401  (loads backend/.env before A2A_* is read by `python -m app.a2a`)

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def is_loopback(host: str) -> bool:
    if host.lower() in LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class A2ASettings:
    # Interface the server binds to. 127.0.0.1 = this laptop only.
    host: str = "127.0.0.1"
    port: int = 9100
    # URL other agents use to reach this agent; advertised in the Agent Card.
    # Must be reachable from the remote machine (Tailnet IP / MagicDNS name), never localhost.
    public_url: str = ""
    # Shared secret for "Authorization: Bearer <token>". Empty = no auth (loopback only).
    auth_token: str = ""

    @property
    def base_url(self) -> str:
        if self.public_url:
            return self.public_url.rstrip("/")
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{host}:{self.port}"

    def validate(self) -> None:
        """Refuse configurations that would expose an unauthenticated agent to the network."""
        if not is_loopback(self.host) and not self.auth_token:
            raise ValueError(
                f"A2A_HOST={self.host} is reachable from other machines, so A2A_AUTH_TOKEN must be set. "
                "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
            )
        if self.auth_token and len(self.auth_token) < 16:
            raise ValueError("A2A_AUTH_TOKEN must be at least 16 characters long.")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("A2A_PUBLIC_URL must start with http:// or https://")


def get_a2a_settings() -> A2ASettings:
    raw_port = os.getenv("A2A_PORT", "").strip() or "9100"
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise ValueError(f"A2A_PORT must be an integer, got {raw_port!r}") from exc
    return A2ASettings(
        host=os.getenv("A2A_HOST", "").strip() or "127.0.0.1",
        port=port,
        public_url=os.getenv("A2A_PUBLIC_URL", "").strip(),
        auth_token=os.getenv("A2A_AUTH_TOKEN", "").strip(),
    )

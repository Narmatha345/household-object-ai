"""Thread-safe in-memory usage statistics (reset on restart; no database yet)."""

from __future__ import annotations

import threading

from app.schemas import StatsResponse


class StatsTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._local = 0
        self._fallback = 0
        self._failed = 0

    def record_local(self) -> None:
        with self._lock:
            self._local += 1

    def record_fallback(self) -> None:
        with self._lock:
            self._fallback += 1

    def record_failure(self) -> None:
        with self._lock:
            self._failed += 1

    def snapshot(self) -> StatsResponse:
        with self._lock:
            total = self._local + self._fallback + self._failed
            return StatsResponse(
                total_requests=total,
                local_detections=self._local,
                openai_fallbacks=self._fallback,
                failed_requests=self._failed,
                fallback_rate=round(self._fallback / total, 4) if total else 0.0,
            )

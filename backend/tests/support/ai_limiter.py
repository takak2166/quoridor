from __future__ import annotations

import threading
from dataclasses import dataclass, field

import app.services.game_service as game_service
from app.config import settings


@dataclass
class InstrumentedAiLimiter:
    """Semaphore shim that counts threads blocked in ``acquire`` (test-only)."""

    _sem: threading.Semaphore
    _stats_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    blocked_in_acquire: int = 0
    peak_blocked_in_acquire: int = 0

    @classmethod
    def with_limit(cls, limit: int) -> InstrumentedAiLimiter:
        return cls(_sem=threading.Semaphore(limit))

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        with self._stats_lock:
            self.blocked_in_acquire += 1
            if self.blocked_in_acquire > self.peak_blocked_in_acquire:
                self.peak_blocked_in_acquire = self.blocked_in_acquire
        try:
            if timeout is not None:
                return self._sem.acquire(blocking=blocking, timeout=timeout)
            return self._sem.acquire(blocking=blocking)
        finally:
            with self._stats_lock:
                self.blocked_in_acquire -= 1

    def release(self) -> None:
        self._sem.release()


def install_instrumented_process_ai_limiter(*, limit: int) -> InstrumentedAiLimiter:
    limiter = InstrumentedAiLimiter.with_limit(limit)
    with game_service._process_ai_limiter_lock:
        game_service._process_ai_limiter = limiter
    return limiter


def reset_process_ai_limiter_for_tests(*, limit: int | None = None) -> threading.Semaphore:
    """Test-only: replace the process-wide AI semaphore backing ``get_process_ai_limiter``.

    Existing ``GameService`` instances keep their previous semaphore reference;
    only new calls to ``get_process_ai_limiter()`` / new services see the reset.
    """
    value = settings.ai_concurrent_limit if limit is None else limit
    with game_service._process_ai_limiter_lock:
        game_service._process_ai_limiter = threading.Semaphore(value)
    return game_service._process_ai_limiter

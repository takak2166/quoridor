from __future__ import annotations

import threading

import app.services.game_service as game_service
from app.config import settings


def reset_process_ai_limiter_for_tests(*, limit: int | None = None) -> threading.Semaphore:
    """Test-only: replace the process-wide AI semaphore backing ``get_process_ai_limiter``.

    Existing ``GameService`` instances keep their previous semaphore reference;
    only new calls to ``get_process_ai_limiter()`` / new services see the reset.
    """
    value = settings.ai_concurrent_limit if limit is None else limit
    with game_service._process_ai_limiter_lock:
        game_service._process_ai_limiter = threading.Semaphore(value)
    return game_service._process_ai_limiter

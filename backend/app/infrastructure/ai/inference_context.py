"""Per-session scope for stateful AI policies (e.g. cached PPO loop history)."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

inference_session_id: ContextVar[str | None] = ContextVar("inference_session_id", default=None)


@contextmanager
def inference_session(session_id: str):
    token = inference_session_id.set(session_id)
    try:
        yield
    finally:
        inference_session_id.reset(token)

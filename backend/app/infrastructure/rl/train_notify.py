"""Notify an external webhook when PPO training finishes."""

from __future__ import annotations

import json
import logging
import os
import socket
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ENV_WEBHOOK_URL = "QUORIDOR_TRAIN_WEBHOOK_URL"
DEFAULT_TIMEOUT_SEC = 10.0


def _load_dotenv_files() -> None:
    """Load repo/backend ``.env`` without overriding existing process env."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    here = Path(__file__).resolve()
    candidates = (
        here.parents[4] / ".env",  # repo root
        here.parents[3] / ".env",  # backend/
        Path.cwd() / ".env",
    )
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        load_dotenv(path, override=False)


def resolve_webhook_url(cli_url: str | None = None) -> str | None:
    """Prefer CLI override, then ``QUORIDOR_TRAIN_WEBHOOK_URL`` (env / ``.env``)."""
    if cli_url is not None and cli_url.strip():
        return cli_url.strip()
    env = os.environ.get(ENV_WEBHOOK_URL, "").strip()
    if env:
        return env
    _load_dotenv_files()
    env = os.environ.get(ENV_WEBHOOK_URL, "").strip()
    return env or None


def build_training_webhook_payload(
    *,
    status: str,
    output: str,
    curriculum: str | None,
    timesteps: int,
    elapsed_sec: float | None = None,
    error: str | None = None,
    host: str | None = None,
) -> dict[str, Any]:
    """Build a Slack/Discord-friendly JSON body for training completion."""
    host_name = host or socket.gethostname()
    lines = [
        f"Quoridor PPO training {status}",
        f"host: {host_name}",
        f"output: {output}",
        f"timesteps: {timesteps}",
    ]
    if curriculum:
        lines.append(f"curriculum: {curriculum}")
    if elapsed_sec is not None:
        lines.append(f"elapsed_sec: {elapsed_sec:.1f}")
    if error:
        lines.append(f"error: {error}")
    text = "\n".join(lines)
    return {
        "text": text,  # Slack / Mattermost incoming webhooks
        "content": text,  # Discord incoming webhooks
        "status": status,
        "output": output,
        "curriculum": curriculum,
        "timesteps": timesteps,
        "elapsed_sec": elapsed_sec,
        "error": error,
        "host": host_name,
    }


def post_webhook(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> bool:
    """POST JSON to ``url``. Returns True on HTTP success; never raises."""
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            logger.info(
                "Training webhook notified (%s): HTTP %s",
                url.split("?", 1)[0],
                getattr(response, "status", "ok"),
            )
            return True
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning("Training webhook failed: %s", exc)
        return False


def notify_training_finished(
    *,
    webhook_url: str | None,
    status: str,
    output: str,
    curriculum: str | None,
    timesteps: int,
    elapsed_sec: float | None = None,
    error: str | None = None,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> bool:
    """Send completion webhook when a URL is configured. No-op if unset."""
    url = resolve_webhook_url(webhook_url)
    if not url:
        return False
    payload = build_training_webhook_payload(
        status=status,
        output=output,
        curriculum=curriculum,
        timesteps=timesteps,
        elapsed_sec=elapsed_sec,
        error=error,
    )
    return post_webhook(url, payload, timeout_sec=timeout_sec)

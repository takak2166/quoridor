from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from app.infrastructure.rl.train_notify import (
    ENV_WEBHOOK_URL,
    build_training_webhook_payload,
    notify_training_finished,
    post_webhook,
    resolve_webhook_url,
)


def test_resolve_webhook_url_prefers_cli_over_env(monkeypatch) -> None:
    monkeypatch.setenv(ENV_WEBHOOK_URL, "https://env.example/hook")
    assert resolve_webhook_url("https://cli.example/hook") == "https://cli.example/hook"
    assert resolve_webhook_url(None) == "https://env.example/hook"
    monkeypatch.delenv(ENV_WEBHOOK_URL, raising=False)
    assert resolve_webhook_url(None) is None
    assert resolve_webhook_url("  ") is None


def test_build_payload_includes_slack_and_discord_fields() -> None:
    payload = build_training_webhook_payload(
        status="success",
        output="../models/out.zip",
        curriculum="very_easy,easy,normal",
        timesteps=1_000_000,
        elapsed_sec=12.5,
        host="trainer",
    )
    assert payload["status"] == "success"
    assert "success" in payload["text"]
    assert payload["content"] == payload["text"]
    assert payload["host"] == "trainer"
    assert payload["elapsed_sec"] == 12.5


def test_post_webhook_posts_json() -> None:
    response = MagicMock()
    response.status = 200
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    with patch(
        "app.infrastructure.rl.train_notify.urllib.request.urlopen",
        return_value=response,
    ) as urlopen:
        ok = post_webhook(
            "https://hooks.example/train",
            {"text": "hi", "status": "success"},
        )
    assert ok is True
    request = urlopen.call_args.args[0]
    assert request.full_url == "https://hooks.example/train"
    assert request.get_method() == "POST"
    assert json.loads(request.data.decode())["text"] == "hi"


def test_notify_training_finished_noop_without_url(monkeypatch) -> None:
    monkeypatch.delenv(ENV_WEBHOOK_URL, raising=False)
    with patch("app.infrastructure.rl.train_notify.post_webhook") as post:
        assert (
            notify_training_finished(
                webhook_url=None,
                status="success",
                output="out.zip",
                curriculum=None,
                timesteps=10,
            )
            is False
        )
    post.assert_not_called()


def test_notify_training_finished_uses_env(monkeypatch) -> None:
    monkeypatch.setenv(ENV_WEBHOOK_URL, "https://hooks.example/env")
    with patch(
        "app.infrastructure.rl.train_notify.post_webhook",
        return_value=True,
    ) as post:
        assert (
            notify_training_finished(
                webhook_url=None,
                status="failed",
                output="out.zip",
                curriculum="easy,normal",
                timesteps=100,
                error="SystemExit(1)",
            )
            is True
        )
    assert post.call_args.args[0] == "https://hooks.example/env"
    payload = post.call_args.args[1]
    assert payload["status"] == "failed"
    assert "SystemExit(1)" in payload["text"]

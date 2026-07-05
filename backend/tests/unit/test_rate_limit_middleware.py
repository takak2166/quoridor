from fastapi.testclient import TestClient

from app.main import create_app
from app.middleware.rate_limit import RateLimitMiddleware


def test_rate_limit_ignores_xff_by_default(monkeypatch) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "trust_forwarded_for", False)
    monkeypatch.setattr(settings, "rate_limit_games_per_min", 1)
    c = TestClient(create_app())
    first = c.post(
        "/api/v1/games",
        json={"human_color": "black", "difficulty": "easy"},
        headers={"X-Forwarded-For": "1.1.1.1"},
    )
    second = c.post(
        "/api/v1/games",
        json={"human_color": "black", "difficulty": "easy"},
        headers={"X-Forwarded-For": "8.8.8.8"},
    )
    assert first.status_code == 201
    assert second.status_code == 429


def test_rate_limit_can_trust_xff(monkeypatch) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "trust_forwarded_for", True)
    monkeypatch.setattr(settings, "rate_limit_games_per_min", 1)
    c = TestClient(create_app())
    first = c.post(
        "/api/v1/games",
        json={"human_color": "black", "difficulty": "easy"},
        headers={"X-Forwarded-For": "1.1.1.1"},
    )
    second = c.post(
        "/api/v1/games",
        json={"human_color": "black", "difficulty": "easy"},
        headers={"X-Forwarded-For": "8.8.8.8"},
    )
    assert first.status_code == 201
    assert second.status_code == 201


def test_rate_limit_cleanup_removes_stale_empty_keys(monkeypatch) -> None:
    middleware = RateLimitMiddleware(app=lambda scope, receive, send: None)
    middleware._move_hits["stale"] = [0.0]
    monkeypatch.setattr("app.middleware.rate_limit.time.time", lambda: 120.0)
    assert middleware._check(middleware._game_hits, "active", limit=5, window=60.0)
    assert "stale" not in middleware._move_hits

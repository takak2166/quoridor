from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.infrastructure.persistence.game_repository import game_repository
from app.main import create_app
from app.services.game_service import get_process_ai_limiter, reset_process_ai_limiter_for_tests
from tests.support.fake_ai import AiConcurrencyTracker, FakeAiProvider


@pytest.fixture(autouse=True)
def clear_game_repository() -> None:
    with game_repository._lock:
        game_repository._sessions.clear()
    yield
    with game_repository._lock:
        game_repository._sessions.clear()


@pytest.fixture(autouse=True)
def reset_ai_limiter_after_test() -> None:
    yield
    reset_process_ai_limiter_for_tests(limit=settings.ai_concurrent_limit)


def test_parallel_create_via_get_game_service_respects_ai_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ai_concurrent_limit", 1)
    monkeypatch.setattr(settings, "rate_limit_games_per_min", 10_000)
    reset_process_ai_limiter_for_tests(limit=1)

    tracker = AiConcurrencyTracker()
    fake = FakeAiProvider(delay_ms=150, concurrency_tracker=tracker)

    import app.infrastructure.ai.factory as ai_factory

    monkeypatch.setattr(ai_factory, "ai_for_difficulty", lambda _difficulty: fake)

    app = create_app()
    client = TestClient(app)

    def create_white_game() -> int:
        resp = client.post("/api/v1/games", json={"human_color": "white", "difficulty": "easy"})
        return resp.status_code

    with ThreadPoolExecutor(max_workers=6) as pool:
        statuses = list(pool.map(lambda _: create_white_game(), range(6)))

    assert tracker.peak <= settings.ai_concurrent_limit
    assert all(status in {201, 503} for status in statuses)
    assert get_process_ai_limiter() is get_process_ai_limiter()


def test_ai_concurrency_limit_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "ai_concurrent_limit", 1)
    monkeypatch.setattr(settings, "rate_limit_games_per_min", 10_000)
    reset_process_ai_limiter_for_tests(limit=1)

    fake = FakeAiProvider(delay_ms=700)

    import app.infrastructure.ai.factory as ai_factory

    monkeypatch.setattr(ai_factory, "ai_for_difficulty", lambda _difficulty: fake)

    app = create_app()
    client = TestClient(app)

    def create_white_game() -> tuple[int, dict[str, object] | None]:
        resp = client.post("/api/v1/games", json={"human_color": "white", "difficulty": "easy"})
        body: dict[str, object] | None = None
        if resp.headers.get("content-type", "").startswith("application/json"):
            body = resp.json()
        return resp.status_code, body

    with ThreadPoolExecutor(max_workers=15) as pool:
        results = list(pool.map(lambda _: create_white_game(), range(15)))

    limit_responses = [
        body
        for status, body in results
        if status == 503 and body is not None
    ]
    assert limit_responses, "expected at least one concurrency-limit 503"
    for body in limit_responses:
        error = body["detail"]["error"]  # type: ignore[index]
        assert error["code"] == "AI_FAILURE"
        assert error["details"]["reason"] == "concurrency_limit"


def test_parallel_play_move_via_get_game_service_respects_ai_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ai_concurrent_limit", 1)
    monkeypatch.setattr(settings, "rate_limit_games_per_min", 10_000)
    monkeypatch.setattr(settings, "rate_limit_moves_per_min", 10_000)
    reset_process_ai_limiter_for_tests(limit=1)

    tracker = AiConcurrencyTracker()
    fake = FakeAiProvider(delay_ms=150, concurrency_tracker=tracker)

    import app.infrastructure.ai.factory as ai_factory

    monkeypatch.setattr(ai_factory, "ai_for_difficulty", lambda _difficulty: fake)

    app = create_app()
    client = TestClient(app)

    sessions: list[tuple[str, str]] = []
    for _ in range(6):
        create = client.post("/api/v1/games", json={"human_color": "black", "difficulty": "easy"})
        assert create.status_code == 201
        data = create.json()
        sessions.append((data["game_id"], data["session_token"]))

    def play_up(game_id: str, token: str) -> int:
        resp = client.post(
            f"/api/v1/games/{game_id}/moves",
            json={"action": {"type": "move", "direction": "up"}},
            headers={"X-Quoridor-Session": token},
        )
        return resp.status_code

    with ThreadPoolExecutor(max_workers=6) as pool:
        statuses = list(
            pool.map(lambda args: play_up(*args), sessions),
        )

    assert tracker.peak <= settings.ai_concurrent_limit
    assert all(status in {200, 503} for status in statuses)

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
from fastapi.testclient import TestClient

import app.services.game_service as game_service
from app.config import settings
from app.main import create_app
from app.middleware.metrics import metrics_store
from app.services.game_service import get_game_service
from tests.support.ai_limiter import (
    install_instrumented_process_ai_limiter,
    reset_process_ai_limiter_for_tests,
)
from tests.support.fake_ai import AiConcurrencyTracker, FakeAiProvider


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

    def create_white_game() -> tuple[int, dict[str, object] | None]:
        resp = client.post("/api/v1/games", json={"human_color": "white", "difficulty": "easy"})
        body: dict[str, object] | None = None
        if resp.headers.get("content-type", "").startswith("application/json"):
            body = resp.json()
        return resp.status_code, body

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: create_white_game(), range(6)))

    statuses = [status for status, _ in results]
    assert tracker.peak <= settings.ai_concurrent_limit
    assert tracker.peak >= 1
    assert all(status in {201, 503} for status in statuses)
    assert 201 in statuses
    limit_statuses = [status for status, _ in results if status == 503]
    assert len(limit_statuses) < len(results), "expected at least one successful create under contention"
    for status, body in results:
        if status != 503:
            continue
        assert body is not None
        error = body["detail"]["error"]  # type: ignore[index]
        assert error["code"] == "AI_FAILURE"
        assert error["details"]["reason"] == "concurrency_limit"


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
        statuses = list(pool.map(lambda args: play_up(*args), sessions))

    assert tracker.peak <= settings.ai_concurrent_limit
    assert tracker.peak >= 1
    assert all(status in {200, 503} for status in statuses)
    assert 200 in statuses


def test_play_move_concurrency_limit_returns_503_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "ai_concurrent_limit", 1)
    monkeypatch.setattr(settings, "rate_limit_games_per_min", 10_000)
    monkeypatch.setattr(settings, "rate_limit_moves_per_min", 10_000)
    reset_process_ai_limiter_for_tests(limit=1)

    fake = FakeAiProvider(delay_ms=700)

    import app.infrastructure.ai.factory as ai_factory

    monkeypatch.setattr(ai_factory, "ai_for_difficulty", lambda _difficulty: fake)

    app = create_app()
    client = TestClient(app)

    sessions: list[tuple[str, str]] = []
    for _ in range(12):
        create = client.post("/api/v1/games", json={"human_color": "black", "difficulty": "easy"})
        assert create.status_code == 201
        data = create.json()
        sessions.append((data["game_id"], data["session_token"]))

    def play_up(args: tuple[str, str]) -> tuple[int, dict[str, object] | None]:
        game_id, token = args
        resp = client.post(
            f"/api/v1/games/{game_id}/moves",
            json={"action": {"type": "move", "direction": "up"}},
            headers={"X-Quoridor-Session": token},
        )
        body: dict[str, object] | None = None
        if resp.headers.get("content-type", "").startswith("application/json"):
            body = resp.json()
        return resp.status_code, body

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(play_up, sessions))

    limit_responses = [body for status, body in results if status == 503 and body is not None]
    assert limit_responses, "expected at least one play_move concurrency-limit 503"
    for body in limit_responses:
        error = body["detail"]["error"]  # type: ignore[index]
        assert error["code"] == "AI_FAILURE"
        assert error["details"]["reason"] == "concurrency_limit"


def test_get_game_service_instances_share_process_ai_limiter() -> None:
    reset_process_ai_limiter_for_tests(limit=4)
    first = get_game_service()
    second = get_game_service()
    assert first._ai_limiter is second._ai_limiter
    assert first._ai_limiter is game_service.get_process_ai_limiter()


def test_cross_request_create_and_play_share_ai_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ai_concurrent_limit", 1)
    monkeypatch.setattr(settings, "rate_limit_games_per_min", 10_000)
    monkeypatch.setattr(settings, "rate_limit_moves_per_min", 10_000)
    reset_process_ai_limiter_for_tests(limit=1)

    tracker = AiConcurrencyTracker()
    fake = FakeAiProvider(delay_ms=700, concurrency_tracker=tracker)

    import app.infrastructure.ai.factory as ai_factory

    monkeypatch.setattr(ai_factory, "ai_for_difficulty", lambda _difficulty: fake)

    app = create_app()
    client = TestClient(app)

    black_sessions: list[tuple[str, str]] = []
    for _ in range(3):
        create = client.post("/api/v1/games", json={"human_color": "black", "difficulty": "easy"})
        assert create.status_code == 201
        data = create.json()
        black_sessions.append((data["game_id"], data["session_token"]))

    def create_white() -> tuple[int, dict[str, object] | None]:
        resp = client.post("/api/v1/games", json={"human_color": "white", "difficulty": "easy"})
        body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else None
        return resp.status_code, body

    def play_black(session: tuple[str, str]) -> tuple[int, dict[str, object] | None]:
        game_id, token = session
        resp = client.post(
            f"/api/v1/games/{game_id}/moves",
            json={"action": {"type": "move", "direction": "up"}},
            headers={"X-Quoridor-Session": token},
        )
        body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else None
        return resp.status_code, body

    sessions_for_play = black_sessions
    for _ in range(5):
        create = client.post("/api/v1/games", json={"human_color": "black", "difficulty": "easy"})
        assert create.status_code == 201
        data = create.json()
        sessions_for_play.append((data["game_id"], data["session_token"]))

    tasks: list[tuple[str, tuple[str, str] | None]] = (
        [("create", None)] * 6 + [("play", s) for s in sessions_for_play]
    )

    def run(task: tuple[str, tuple[str, str] | None]) -> tuple[int, dict[str, object] | None]:
        kind, session = task
        if kind == "create":
            return create_white()
        assert session is not None
        return play_black(session)

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(run, tasks))

    assert tracker.peak <= 1
    statuses = [status for status, _ in results]
    assert any(s == 201 for s in statuses), "expected cross-request create to hit shared AI limiter"
    assert any(s == 200 for s in statuses), "expected cross-request play_move to hit shared AI limiter"
    for status, body in results:
        if status != 503 or body is None:
            continue
        error = body["detail"]["error"]  # type: ignore[index]
        assert error["code"] == "AI_FAILURE"
        assert error["details"]["reason"] == "concurrency_limit"


def test_slots_released_after_contention_allows_further_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ai_concurrent_limit", 1)
    monkeypatch.setattr(settings, "rate_limit_games_per_min", 10_000)
    reset_process_ai_limiter_for_tests(limit=1)

    tracker = AiConcurrencyTracker()
    fake = FakeAiProvider(delay_ms=80, concurrency_tracker=tracker)

    import app.infrastructure.ai.factory as ai_factory

    monkeypatch.setattr(ai_factory, "ai_for_difficulty", lambda _difficulty: fake)

    app = create_app()
    client = TestClient(app)

    def create_white() -> int:
        return client.post(
            "/api/v1/games", json={"human_color": "white", "difficulty": "easy"}
        ).status_code

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: create_white(), range(4)))

    assert tracker.current == 0
    after = client.post("/api/v1/games", json={"human_color": "white", "difficulty": "easy"})
    assert after.status_code == 201


def test_play_move_concurrency_limit_records_rollback_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ai_concurrent_limit", 1)
    monkeypatch.setattr(settings, "rate_limit_games_per_min", 10_000)
    monkeypatch.setattr(settings, "rate_limit_moves_per_min", 10_000)
    reset_process_ai_limiter_for_tests(limit=1)

    fake = FakeAiProvider(delay_ms=700)

    import app.infrastructure.ai.factory as ai_factory

    monkeypatch.setattr(ai_factory, "ai_for_difficulty", lambda _difficulty: fake)

    app = create_app()
    client = TestClient(app)

    sessions: list[tuple[str, str]] = []
    for _ in range(15):
        create = client.post("/api/v1/games", json={"human_color": "black", "difficulty": "easy"})
        assert create.status_code == 201
        data = create.json()
        sessions.append((data["game_id"], data["session_token"]))

    rollback_before = metrics_store.ai_rollback_total

    def play_up(args: tuple[str, str]) -> int:
        game_id, token = args
        return client.post(
            f"/api/v1/games/{game_id}/moves",
            json={"action": {"type": "move", "direction": "up"}},
            headers={"X-Quoridor-Session": token},
        ).status_code

    with ThreadPoolExecutor(max_workers=15) as pool:
        statuses = list(pool.map(play_up, sessions))

    assert any(s == 503 for s in statuses)
    assert metrics_store.ai_rollback_total > rollback_before


def test_create_concurrency_limit_records_rollback_and_no_active_game(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.infrastructure.persistence.game_repository import game_repository

    monkeypatch.setattr(settings, "ai_concurrent_limit", 1)
    monkeypatch.setattr(settings, "rate_limit_games_per_min", 10_000)
    reset_process_ai_limiter_for_tests(limit=1)

    fake = FakeAiProvider(delay_ms=700)

    import app.infrastructure.ai.factory as ai_factory

    monkeypatch.setattr(ai_factory, "ai_for_difficulty", lambda _difficulty: fake)

    app = create_app()
    client = TestClient(app)

    active_before = game_repository.active_count()
    rollback_before = metrics_store.ai_rollback_total

    def create_white() -> int:
        return client.post(
            "/api/v1/games", json={"human_color": "white", "difficulty": "easy"}
        ).status_code

    with ThreadPoolExecutor(max_workers=15) as pool:
        statuses = list(pool.map(lambda _: create_white(), range(15)))

    assert any(s == 503 for s in statuses), f"statuses={statuses}"
    assert metrics_store.ai_rollback_total > rollback_before
    successes = sum(1 for s in statuses if s == 201)
    assert game_repository.active_count() == active_before + successes


def test_acquire_contention_increments_blocked_acquire_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ai_concurrent_limit", 1)
    monkeypatch.setattr(settings, "rate_limit_games_per_min", 10_000)
    limiter = install_instrumented_process_ai_limiter(limit=1)

    fake = FakeAiProvider(delay_ms=200)

    import app.infrastructure.ai.factory as ai_factory

    monkeypatch.setattr(ai_factory, "ai_for_difficulty", lambda _difficulty: fake)

    app = create_app()
    client = TestClient(app)

    def create_white() -> int:
        return client.post(
            "/api/v1/games", json={"human_color": "white", "difficulty": "easy"}
        ).status_code

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(create_white) for _ in range(5)]
        for fut in as_completed(futures):
            assert fut.result() in {201, 503}

    assert limiter.peak_blocked_in_acquire >= 2, (
        "expected multiple threads to block on shared acquire under limit=1"
    )


def test_inference_failure_after_contention_releases_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ai_concurrent_limit", 1)
    monkeypatch.setattr(settings, "rate_limit_games_per_min", 10_000)
    reset_process_ai_limiter_for_tests(limit=1)

    call_count = 0

    class FlakyAi(FakeAiProvider):
        def select_move(self, state, color):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("inference boom")
            return super().select_move(state, color)

    fake = FlakyAi(delay_ms=50)

    import app.infrastructure.ai.factory as ai_factory

    monkeypatch.setattr(ai_factory, "ai_for_difficulty", lambda _difficulty: fake)

    app = create_app()
    client = TestClient(app)

    first = client.post("/api/v1/games", json={"human_color": "white", "difficulty": "easy"})
    assert first.status_code == 503

    second = client.post("/api/v1/games", json={"human_color": "white", "difficulty": "easy"})
    assert second.status_code == 201

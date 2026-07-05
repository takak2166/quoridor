import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.game_service import GameService, get_game_service
from tests.support.fake_ai import FakeAiProvider


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    svc = GameService(ai_factory=lambda _: FakeAiProvider())
    app.dependency_overrides[get_game_service] = lambda: svc
    return TestClient(app)


def test_health(client: TestClient) -> None:
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 200


def test_create_game_white_cpu_first(client: TestClient) -> None:
    resp = client.post("/api/v1/games", json={"human_color": "white", "difficulty": "easy"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["session_token"]
    assert data["cpu_move"] is not None
    assert data["turn"] == "human"


def test_create_game_black_human_first(client: TestClient) -> None:
    resp = client.post("/api/v1/games", json={"human_color": "black", "difficulty": "easy"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["cpu_move"] is None
    assert data["turn"] == "human"


def test_illegal_move(client: TestClient) -> None:
    create = client.post("/api/v1/games", json={"human_color": "black", "difficulty": "easy"}).json()
    headers = {"X-Quoridor-Session": create["session_token"]}
    gid = create["game_id"]
    before = create["state"]
    resp = client.post(
        f"/api/v1/games/{gid}/moves",
        json={"action": {"type": "move", "direction": "down"}},
        headers=headers,
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"]["code"] == "ILLEGAL_MOVE"
    get_resp = client.get(f"/api/v1/games/{gid}", headers=headers)
    assert get_resp.json()["state"] == before


def test_session_expired(client: TestClient) -> None:
    create = client.post("/api/v1/games", json={"human_color": "black", "difficulty": "easy"}).json()
    resp = client.post(
        f"/api/v1/games/{create['game_id']}/moves",
        json={"action": {"type": "move", "direction": "up"}},
        headers={"X-Quoridor-Session": "bad-token"},
    )
    assert resp.status_code == 403


def test_get_game_requires_session_header(client: TestClient) -> None:
    create = client.post("/api/v1/games", json={"human_color": "black", "difficulty": "easy"}).json()
    resp = client.get(f"/api/v1/games/{create['game_id']}")
    assert resp.status_code == 422


def test_get_legal_actions_requires_session_header(client: TestClient) -> None:
    create = client.post("/api/v1/games", json={"human_color": "black", "difficulty": "easy"}).json()
    resp = client.get(f"/api/v1/games/{create['game_id']}/legal-actions")
    assert resp.status_code == 422


def test_get_legal_actions_returns_actions(client: TestClient) -> None:
    create = client.post("/api/v1/games", json={"human_color": "black", "difficulty": "easy"}).json()
    headers = {"X-Quoridor-Session": create["session_token"]}
    resp = client.get(f"/api/v1/games/{create['game_id']}/legal-actions", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["actions"]


def test_ai_failure(client: TestClient) -> None:
    app = create_app()
    svc = GameService(ai_factory=lambda _: FakeAiProvider(fail=True))
    app.dependency_overrides[get_game_service] = lambda: svc
    c = TestClient(app)
    create = c.post("/api/v1/games", json={"human_color": "black", "difficulty": "easy"}).json()
    headers = {"X-Quoridor-Session": create["session_token"]}
    resp = c.post(
        f"/api/v1/games/{create['game_id']}/moves",
        json={"action": {"type": "move", "direction": "up"}},
        headers=headers,
    )
    assert resp.status_code == 503
    assert resp.json()["detail"]["error"]["code"] == "AI_FAILURE"


def test_diagonal_jump_with_to_returns_200(client_with_j2) -> None:
    client, game_id, token = client_with_j2
    headers = {"X-Quoridor-Session": token}
    resp = client.post(
        f"/api/v1/games/{game_id}/moves",
        json={"action": {"type": "move", "direction": "up", "to": {"row": 4, "col": 5}}},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["state"]["white"]["row"] == 4
    assert resp.json()["state"]["white"]["col"] == 5
    assert resp.json()["human_move"]["to"] == {"row": 4, "col": 5}


def test_diagonal_jump_direction_only_returns_400_ambiguous(client_with_j2) -> None:
    client, game_id, token = client_with_j2
    headers = {"X-Quoridor-Session": token}
    resp = client.post(
        f"/api/v1/games/{game_id}/moves",
        json={"action": {"type": "move", "direction": "up"}},
        headers=headers,
    )
    assert resp.status_code == 400
    body = resp.json()["detail"]["error"]
    assert body["code"] == "ILLEGAL_MOVE"
    assert body["details"]["reason"] == "ambiguous"


def test_ai_failure_preserves_prior_state(client: TestClient) -> None:
    app = create_app()
    svc = GameService(ai_factory=lambda _: FakeAiProvider(fail=True))
    app.dependency_overrides[get_game_service] = lambda: svc
    c = TestClient(app)
    create = c.post("/api/v1/games", json={"human_color": "black", "difficulty": "easy"}).json()
    headers = {"X-Quoridor-Session": create["session_token"]}
    gid = create["game_id"]
    before = c.get(f"/api/v1/games/{gid}", headers=headers).json()
    resp = c.post(
        f"/api/v1/games/{gid}/moves",
        json={"action": {"type": "move", "direction": "up"}},
        headers=headers,
    )
    assert resp.status_code == 503
    after = c.get(f"/api/v1/games/{gid}", headers=headers).json()
    assert after["state"] == before["state"]
    assert after["winner"] == before["winner"]


def test_ai_failure_on_create_white_cpu_first(client: TestClient) -> None:
    app = create_app()
    svc = GameService(ai_factory=lambda _: FakeAiProvider(fail=True))
    app.dependency_overrides[get_game_service] = lambda: svc
    c = TestClient(app)
    resp = c.post("/api/v1/games", json={"human_color": "white", "difficulty": "easy"})
    assert resp.status_code == 503
    assert resp.json()["detail"]["error"]["code"] == "AI_FAILURE"


def test_ai_unexpected_exception_preserves_prior_state(client: TestClient) -> None:
    app = create_app()
    svc = GameService(ai_factory=lambda _: FakeAiProvider(exception_to_raise=ValueError("boom")))
    app.dependency_overrides[get_game_service] = lambda: svc
    c = TestClient(app)
    create = c.post("/api/v1/games", json={"human_color": "black", "difficulty": "easy"}).json()
    headers = {"X-Quoridor-Session": create["session_token"]}
    gid = create["game_id"]
    before = c.get(f"/api/v1/games/{gid}", headers=headers).json()
    resp = c.post(
        f"/api/v1/games/{gid}/moves",
        json={"action": {"type": "move", "direction": "up"}},
        headers=headers,
    )
    assert resp.status_code == 503
    assert resp.json()["detail"]["error"]["code"] == "AI_FAILURE"
    after = c.get(f"/api/v1/games/{gid}", headers=headers).json()
    assert after["state"] == before["state"]
    assert after["winner"] == before["winner"]


def test_ai_illegal_move_preserves_prior_state(client: TestClient) -> None:
    app = create_app()
    svc = GameService(ai_factory=lambda _: FakeAiProvider(illegal_move=True))
    app.dependency_overrides[get_game_service] = lambda: svc
    c = TestClient(app)
    create = c.post("/api/v1/games", json={"human_color": "black", "difficulty": "easy"}).json()
    headers = {"X-Quoridor-Session": create["session_token"]}
    gid = create["game_id"]
    before = c.get(f"/api/v1/games/{gid}", headers=headers).json()
    resp = c.post(
        f"/api/v1/games/{gid}/moves",
        json={"action": {"type": "move", "direction": "up"}},
        headers=headers,
    )
    assert resp.status_code == 503
    assert resp.json()["detail"]["error"]["code"] == "AI_FAILURE"
    after = c.get(f"/api/v1/games/{gid}", headers=headers).json()
    assert after["state"] == before["state"]
    assert after["winner"] == before["winner"]


def test_parallel_play_move_requests_are_serialized() -> None:
    app = create_app()
    svc = GameService(ai_factory=lambda _: FakeAiProvider(delay_ms=250))
    app.dependency_overrides[get_game_service] = lambda: svc
    c = TestClient(app)

    create = c.post("/api/v1/games", json={"human_color": "black", "difficulty": "easy"}).json()
    headers = {"X-Quoridor-Session": create["session_token"]}
    gid = create["game_id"]

    def post_move() -> int:
        resp = c.post(
            f"/api/v1/games/{gid}/moves",
            json={"action": {"type": "move", "direction": "up"}},
            headers=headers,
        )
        return resp.status_code

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = [future.result() for future in (pool.submit(post_move), pool.submit(post_move))]
    elapsed = time.perf_counter() - start
    assert elapsed >= 0.40
    assert all(status in {200, 400, 409} for status in statuses)
    state = c.get(f"/api/v1/games/{gid}", headers=headers).json()["state"]
    assert 0 <= state["white"]["row"] <= 8
    assert 0 <= state["black"]["row"] <= 8


def test_rate_limit_response_shape(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "rate_limit_games_per_min", 1)
    first = client.post("/api/v1/games", json={"human_color": "black", "difficulty": "easy"})
    assert first.status_code == 201
    second = client.post("/api/v1/games", json={"human_color": "black", "difficulty": "easy"})
    assert second.status_code == 429
    assert second.json()["detail"]["error"]["code"] == "RATE_LIMITED"


def test_wrong_turn(client: TestClient) -> None:
    from dataclasses import replace

    from app.infrastructure.persistence.game_repository import game_repository

    create = client.post("/api/v1/games", json={"human_color": "white", "difficulty": "easy"}).json()
    record = game_repository.get(create["game_id"])
    assert record is not None
    record.game.state = replace(record.game.state, current_player="black")
    headers = {"X-Quoridor-Session": create["session_token"]}
    resp = client.post(
        f"/api/v1/games/{create['game_id']}/moves",
        json={"action": {"type": "move", "direction": "up"}},
        headers=headers,
    )
    assert resp.status_code == 409
    assert resp.json()["detail"]["error"]["code"] == "WRONG_TURN"


def test_game_over(client: TestClient) -> None:
    from app.infrastructure.persistence.game_repository import game_repository
    from quoridor.domain.game import Game
    from quoridor.domain.state import QuoridorState, empty_walls

    create = client.post("/api/v1/games", json={"human_color": "black", "difficulty": "easy"}).json()
    record = game_repository.get(create["game_id"])
    assert record is not None
    record.game = Game(
        state=QuoridorState(
            white=(0, 4),
            black=(4, 4),
            white_walls_remaining=10,
            black_walls_remaining=10,
            horizontal_walls=empty_walls(),
            vertical_walls=empty_walls(),
            current_player="black",
        ),
        winner="white",
    )
    headers = {"X-Quoridor-Session": create["session_token"]}
    resp = client.post(
        f"/api/v1/games/{create['game_id']}/moves",
        json={"action": {"type": "move", "direction": "up"}},
        headers=headers,
    )
    assert resp.status_code == 409
    assert resp.json()["detail"]["error"]["code"] == "GAME_OVER"


def test_game_capacity_exceeded(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "max_concurrent_games", 1)
    first = client.post("/api/v1/games", json={"human_color": "black", "difficulty": "easy"})
    assert first.status_code == 201
    second = client.post("/api/v1/games", json={"human_color": "black", "difficulty": "easy"})
    assert second.status_code == 503
    assert second.json()["detail"]["error"]["code"] == "GAME_CAPACITY_EXCEEDED"


def test_delete_game(client: TestClient) -> None:
    create = client.post("/api/v1/games", json={"human_color": "black", "difficulty": "easy"}).json()
    headers = {"X-Quoridor-Session": create["session_token"]}
    delete = client.delete(f"/api/v1/games/{create['game_id']}", headers=headers)
    assert delete.status_code == 204
    get_resp = client.get(f"/api/v1/games/{create['game_id']}", headers=headers)
    assert get_resp.status_code == 403

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.infrastructure.persistence.game_repository import game_repository
from app.main import create_app
from app.services.game_service import GameService, get_game_service
from quoridor.domain.game import Game
from tests.support.fake_ai import FakeAiProvider
from tests.unit.fixtures.plan_fixtures import JUMP_CASES


def build_game_from_fixture(fixture_id: str) -> Game:
    for case in JUMP_CASES:
        if case["id"] == fixture_id:
            return Game(state=case["state"])
    raise KeyError(fixture_id)


@pytest.fixture(autouse=True)
def clear_game_repository() -> None:
    with game_repository._lock:
        game_repository._sessions.clear()
    yield
    with game_repository._lock:
        game_repository._sessions.clear()


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    svc = GameService(ai_factory=lambda _: FakeAiProvider())
    app.dependency_overrides[get_game_service] = lambda: svc
    return TestClient(app)


@pytest.fixture
def client_with_j2(client: TestClient) -> tuple[TestClient, str, str]:
    create = client.post("/api/v1/games", json={"human_color": "white", "difficulty": "easy"})
    assert create.status_code == 201
    data = create.json()
    record = game_repository.get(data["game_id"])
    assert record is not None
    record.game = build_game_from_fixture("J.2-DIAG-BOTH")
    return client, data["game_id"], data["session_token"]

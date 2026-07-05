from quoridor.domain.state import initial_state


def test_hard_policy_returns_legal_move_without_model() -> None:
    from app.infrastructure.ai.factory import HardPolicy

    policy = HardPolicy(model_path="models/missing_model.zip")
    action = policy.select_move(initial_state(), "black")
    assert action is not None


def test_expert_policy_returns_legal_move_without_model() -> None:
    from app.infrastructure.ai.factory import ExpertMCTSPolicy

    policy = ExpertMCTSPolicy(model_path="models/missing_model.zip")
    action = policy.select_move(initial_state(), "black")
    assert action is not None


def test_model_status_reports_availability() -> None:
    from app.infrastructure.ai.factory import model_status

    status = model_status()
    assert "hard" in status
    assert "expert" in status
    assert status["effective_ai"]["very_easy"] == "minimax"
    expert = status["effective_ai"]["expert"]
    assert expert in {"mcts", "unavailable"}
    if status["expert"]["file_present"] and status["expert"]["loadable"]:
        assert expert == "mcts"
    else:
        assert expert == "unavailable"
    assert status["hard"]["file_present"] is False or isinstance(status["hard"]["file_present"], bool)


def test_very_easy_factory_returns_move_only_policy() -> None:
    from app.infrastructure.ai.factory import ai_for_difficulty
    from quoridor.domain.actions import Move, WallSlot
    from tests.unit.fixtures.plan_fixtures import build_state

    policy = ai_for_difficulty("very_easy")
    states = [
        initial_state(),
        build_state(white=(6, 4), black=(2, 4), current="black"),
    ]
    for state in states:
        action = policy.select_move(state, state.current_player)
        assert isinstance(action, Move)
        assert not isinstance(action, WallSlot)


def test_health_ready_shape() -> None:
    from fastapi.testclient import TestClient

    from app.infrastructure.ai.factory import model_status
    from app.main import create_app

    app = create_app()
    c = TestClient(app)
    resp = c.get("/health/ready")
    assert resp.status_code == 200
    data = resp.json()
    expected_expert = model_status()["effective_ai"]["expert"]
    assert data["effective_ai"]["expert"] == expected_expert
    assert "file_present" in data["models"]["hard"]

from unittest.mock import patch

from app.infrastructure.ai.mcts import mcts_search
from app.middleware.metrics import metrics_store
from quoridor.domain.state import initial_state
from quoridor.rules import get_legal_actions


class FakeAi:
    def action_prior(self, state, color):
        import numpy as np

        from quoridor.domain.actions import NUM_ACTIONS, encode

        legal = get_legal_actions(state)
        prior = np.zeros(NUM_ACTIONS, dtype=np.float64)
        for action in legal:
            prior[encode(action)] = 1.0
        prior /= prior.sum()
        return prior

    def value(self, state, color) -> float:
        return 0.0


def test_mcts_budget_stops_within_wall_clock() -> None:
    ai = FakeAi()
    state = initial_state()
    legal = get_legal_actions(state)
    before = len(metrics_store.mcts_sim_count)
    clock = {"t": 0.0}

    def fake_perf_counter() -> float:
        clock["t"] += 0.01
        return clock["t"]

    with patch("app.infrastructure.ai.mcts.time.perf_counter", side_effect=fake_perf_counter):
        clock["t"] = 0.0
        mcts_search(
            state,
            "black",
            prior_fn=ai.action_prior,
            value_fn=ai.value,
            legal_actions=legal,
            budget_ms=450,
        )

    assert clock["t"] <= 0.5
    assert len(metrics_store.mcts_sim_count) == before + 1

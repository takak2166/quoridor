import math
from unittest.mock import patch

import numpy as np

from app.infrastructure.ai.mcts import _cache_prior, _MCTSNode, _normalized_prior, mcts_search
from app.middleware.metrics import metrics_store
from quoridor.domain.actions import NUM_ACTIONS, encode
from quoridor.domain.state import initial_state
from quoridor.rules import get_legal_actions


class FakeAi:
    def action_prior(self, state, color):
        import numpy as np

        from quoridor.domain.actions import NUM_ACTIONS, encode

        legal = get_legal_actions(state)
        prior = np.zeros(NUM_ACTIONS, dtype=np.float64)
        from_pos = state.pawn(color)
        for action in legal:
            prior[encode(action, from_pos=from_pos)] = 1.0
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


def test_cache_prior_calls_prior_fn_once() -> None:
    ai = FakeAi()
    calls = {"n": 0}
    original = ai.action_prior

    def counting_prior(state, color):
        calls["n"] += 1
        return original(state, color)

    state = initial_state()
    node = _MCTSNode(state=state, parent=None, action=None, player="black", untried=[])
    first = _cache_prior(node, counting_prior)
    second = _cache_prior(node, counting_prior)
    assert calls["n"] == 1
    assert first is second


def test_mcts_prior_fn_called_once_per_tree_node() -> None:
    """Shallow legal set + fake clock forces _select_child; duplicate (state, color) calls fail."""
    ai = FakeAi()
    seen: set[tuple[int, str]] = set()
    duplicates: list[tuple[int, str]] = []
    original = ai.action_prior

    def counting_prior(state, color):
        key = (id(state), color)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
        return original(state, color)

    state = initial_state()
    legal = get_legal_actions(state)[:2]
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
            prior_fn=counting_prior,
            value_fn=ai.value,
            legal_actions=legal,
            budget_ms=450,
        )

    assert len(metrics_store.mcts_sim_count) == before + 1
    sims = metrics_store.mcts_sim_count[-1]
    assert sims > 5
    assert duplicates == []
    assert len(seen) < sims


def test_normalized_prior_uniform_on_non_finite_weights() -> None:
    state = initial_state()
    legal = get_legal_actions(state)[:3]
    from_pos = state.pawn("black")
    uniform = 1.0 / len(legal)

    def make_prior(*values: float) -> np.ndarray:
        prior = np.zeros(NUM_ACTIONS, dtype=np.float64)
        for action, value in zip(legal, values, strict=True):
            prior[encode(action, from_pos=from_pos)] = value
        return prior

    for bad_values in (
        (math.nan, math.nan, math.nan),
        (math.inf, 0.0, 0.0),
        (1.0, math.nan, 2.0),
    ):
        norm = _normalized_prior(make_prior(*bad_values), legal, from_pos)
        assert all(math.isfinite(v) for v in norm.values())
        assert all(abs(v - uniform) < 1e-12 for v in norm.values())

    good = _normalized_prior(make_prior(1.0, 2.0, 3.0), legal, from_pos)
    assert math.isfinite(sum(good.values()))
    assert abs(sum(good.values()) - 1.0) < 1e-12


def test_mcts_search_with_non_finite_prior_fn() -> None:
    state = initial_state()
    legal = get_legal_actions(state)[:2]
    from_pos = state.pawn("black")

    def toxic_prior(s, color):
        prior = np.zeros(NUM_ACTIONS, dtype=np.float64)
        for action in legal:
            prior[encode(action, from_pos=from_pos)] = math.nan
        return prior

    action = mcts_search(
        state,
        "black",
        prior_fn=toxic_prior,
        value_fn=lambda _s, _c: 0.0,
        legal_actions=legal,
        budget_ms=50,
    )
    assert action in legal


def test_mcts_expands_opponent_nodes_with_side_to_move_prior() -> None:
    """Priors must use the player to move; root-color encoding crashes on White moves."""
    ai = FakeAi()
    state = initial_state()
    legal = get_legal_actions(state)[:2]
    mcts_search(
        state,
        "black",
        prior_fn=ai.action_prior,
        value_fn=ai.value,
        legal_actions=legal,
        budget_ms=80,
    )

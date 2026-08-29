from __future__ import annotations

import numpy as np

from app.infrastructure.rl.white_demonstrations import (
    collect_white_win_transitions,
    greedy_race_action,
    is_greedy_race_action,
)
from quoridor.domain.actions import FORWARD_STEP_INDEX, Move
from quoridor.domain.state import initial_state
from quoridor.rules import apply_action, get_legal_actions


def test_greedy_race_advances_black_forward() -> None:
    state = initial_state()
    action = greedy_race_action(state, "black")
    assert isinstance(action, Move)
    assert action.to == (1, 4)


def test_greedy_race_advances_white_forward() -> None:
    state = initial_state()
    black_fwd = next(
        action
        for action in get_legal_actions(state)
        if isinstance(action, Move) and action.to == (1, 4)
    )
    state = apply_action(state, black_fwd)
    action = greedy_race_action(state, "white")
    assert isinstance(action, Move)
    assert action.to == (7, 4)
    assert is_greedy_race_action(state, "white", action)


def test_collect_white_win_transitions_against_random() -> None:
    transitions = collect_white_win_transitions(n_wins=1, max_games=40, seed=0)
    assert transitions
    assert all(item.obs.shape == (135,) for item in transitions)
    assert all(item.mask.any() for item in transitions)
    assert any(item.action == FORWARD_STEP_INDEX for item in transitions)


def test_collect_black_win_transitions_against_random_white() -> None:
    from app.infrastructure.rl.white_demonstrations import (
        _random_legal_action,
        collect_win_transitions,
        greedy_race_action,
    )
    from app.mappers.observation_mapper import to_observation
    from quoridor.domain.state import initial_state

    def choose(state, color, rng):
        if color == "black":
            return greedy_race_action(state, "black")
        return _random_legal_action(state, rng)

    transitions = collect_win_transitions(
        target="black",
        choose=choose,
        n_wins=1,
        max_games=40,
        seed=0,
        log_label="test-black-vs-random",
    )
    assert transitions
    assert all(item.obs.shape == (135,) for item in transitions)
    opening = to_observation(initial_state(), "black")
    assert any(np.array_equal(item.obs, opening) for item in transitions)


def test_collect_black_wins_vs_normal_wires_factory(monkeypatch) -> None:
    """Keep this off real minimax: greedy Black vs random White is a fast stand-in."""
    import random

    from app.infrastructure.rl import white_demonstrations as wd
    from app.infrastructure.rl.white_demonstrations import (
        _random_legal_action,
        greedy_race_action,
    )

    class _FakeNormal:
        def select_move(self, state, color):
            if color == "black":
                return greedy_race_action(state, "black")
            return _random_legal_action(state, random.Random(0))

    monkeypatch.setattr(
        "app.infrastructure.ai.factory.ai_for_difficulty",
        lambda _difficulty: _FakeNormal(),
    )
    transitions = wd.collect_black_wins_vs_normal(n_wins=1, max_games=40, seed=0)
    assert transitions
    assert all(item.obs.shape == (135,) for item in transitions)

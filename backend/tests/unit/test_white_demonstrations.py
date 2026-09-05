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


def test_collect_black_wins_vs_normal_uses_chooser(monkeypatch) -> None:
    """Keep this off real minimax: greedy Black vs random White is a fast stand-in."""
    import random

    from app.infrastructure.rl import white_demonstrations as wd
    from app.infrastructure.rl.white_demonstrations import (
        _random_legal_action,
        greedy_race_action,
    )

    def fake_chooser():
        def choose(state, color, rng):
            if color == "black":
                return greedy_race_action(state, "black")
            return _random_legal_action(state, random.Random(0))

        return choose

    monkeypatch.setattr(wd, "_normal_chooser", fake_chooser)
    transitions = wd.collect_black_wins_vs_normal(n_wins=1, max_games=40, seed=0)
    assert transitions
    assert all(item.obs.shape == (135,) for item in transitions)


def test_collect_black_wins_expert_vs_normal_uses_chooser(monkeypatch) -> None:
    import random

    from app.infrastructure.rl import white_demonstrations as wd
    from app.infrastructure.rl.white_demonstrations import (
        _random_legal_action,
        greedy_race_action,
    )

    def fake_chooser(*, budget_ms=450):
        del budget_ms

        def choose(state, color, rng):
            if color == "black":
                return greedy_race_action(state, "black")
            return _random_legal_action(state, random.Random(0))

        return choose

    monkeypatch.setattr(wd, "_expert_vs_normal_chooser", fake_chooser)
    transitions = wd.collect_black_wins_expert_vs_normal(n_wins=1, max_games=40, seed=0)
    assert transitions
    assert all(item.obs.shape == (135,) for item in transitions)


def test_black_transitions_from_m14_scoresheet() -> None:
    from pathlib import Path

    from app.infrastructure.rl.white_demonstrations import (
        black_transitions_from_scoresheet,
        load_black_win_transitions,
    )
    from quoridor.domain.actions import FORWARD_STEP_INDEX

    fixture = Path(__file__).parent / "fixtures" / "black_win_vs_normal_m14.txt"
    transitions = black_transitions_from_scoresheet(fixture.read_text(encoding="utf-8"))
    assert len(transitions) == 32
    assert transitions[0].action == FORWARD_STEP_INDEX
    assert all(item.obs.shape == (135,) for item in transitions)
    assert all(item.mask.any() for item in transitions)
    assert all(item.mask[item.action] for item in transitions)

    loaded = load_black_win_transitions(fixture, upsample_m14=3)
    assert len(loaded) == 32 * 3


def test_load_scoresheets_skips_index_without_scoresheet(tmp_path) -> None:
    from pathlib import Path

    from app.infrastructure.rl.white_demonstrations import load_black_win_transitions

    fixture = Path(__file__).parent / "fixtures" / "black_win_vs_normal_m14.txt"
    (tmp_path / "keep.txt").write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "index.txt").write_text("M(1, 4)\tplies=63\n", encoding="utf-8")
    loaded = load_black_win_transitions(tmp_path, upsample_m14=1)
    assert len(loaded) == 32


def test_load_black_win_transitions_missing_path() -> None:
    from pathlib import Path

    import pytest

    from app.infrastructure.rl.white_demonstrations import load_black_win_transitions

    with pytest.raises(FileNotFoundError, match="black-win scoresheets"):
        load_black_win_transitions(Path("/no/such/scoresheets"))


def test_load_ten_pawn_first_400ms_scoresheets() -> None:
    from pathlib import Path

    from app.infrastructure.rl.white_demonstrations import load_black_win_transitions
    from quoridor.domain.actions import FORWARD_STEP_INDEX

    fixture_dir = Path(__file__).parent / "fixtures" / "black_wins_vs_400ms_pawn"
    loaded = load_black_win_transitions(fixture_dir, upsample_m14=1)
    assert len(list(fixture_dir.glob("*.txt"))) == 10
    assert len(loaded) == 365
    assert all(item.obs.shape == (135,) for item in loaded)
    assert all(item.mask.any() for item in loaded)
    assert all(item.mask[item.action] for item in loaded)
    assert any(item.action == FORWARD_STEP_INDEX for item in loaded)
    assert len(load_black_win_transitions(fixture_dir, upsample_m14=2)) == 501

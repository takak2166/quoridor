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
    from app.mappers.observation_mapper import SECOND_PLAYER_OBS_INDEX

    assert all(item.obs[SECOND_PLAYER_OBS_INDEX] == 0.0 for item in transitions)
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

    from app.infrastructure.rl.white_demonstrations import (
        black_transitions_from_scoresheet,
        load_black_win_transitions,
    )
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
    main = next(fixture_dir.glob("*M14_M15_M25.txt"))
    main_n = len(black_transitions_from_scoresheet(main.read_text(encoding="utf-8")))
    heavy = load_black_win_transitions(
        fixture_dir,
        upsample_m14=1,
        upsample_stem="M14_M15_M25",
        upsample_heavy=12,
    )
    assert len(heavy) == 365 + main_n * 11


def test_white_transitions_from_scoresheet_fixture() -> None:
    from pathlib import Path

    from app.infrastructure.rl.white_demonstrations import (
        load_white_win_transitions,
        white_transitions_from_scoresheet,
    )
    from app.mappers.observation_mapper import SECOND_PLAYER_OBS_INDEX
    from quoridor.domain.actions import FORWARD_STEP_INDEX

    fixture = Path(__file__).parent / "fixtures" / "white_win_vs_random.txt"
    transitions = white_transitions_from_scoresheet(fixture.read_text(encoding="utf-8"))
    assert transitions
    assert all(item.obs.shape == (135,) for item in transitions)
    assert all(item.obs[SECOND_PLAYER_OBS_INDEX] == 0.0 for item in transitions)
    assert all(item.mask.any() for item in transitions)
    assert all(item.mask[item.action] for item in transitions)
    assert any(item.action == FORWARD_STEP_INDEX for item in transitions)
    loaded = load_white_win_transitions(fixture, upsample=3)
    assert len(loaded) == len(transitions) * 3
    skipped = load_white_win_transitions(fixture, upsample=1, opening_pawn_plies=100)
    assert skipped == []


def test_white_transitions_set_second_player_bit_when_enabled(monkeypatch) -> None:
    from pathlib import Path

    from app.config import settings
    from app.infrastructure.rl.white_demonstrations import white_transitions_from_scoresheet
    from app.mappers.observation_mapper import SECOND_PLAYER_OBS_INDEX

    monkeypatch.setattr(settings, "second_player_obs_bit", True)
    fixture = Path(__file__).parent / "fixtures" / "white_win_vs_random.txt"
    transitions = white_transitions_from_scoresheet(fixture.read_text(encoding="utf-8"))
    assert transitions
    assert all(item.obs[SECOND_PLAYER_OBS_INDEX] == 1.0 for item in transitions)


def test_load_white_win_transitions_missing_path() -> None:
    from pathlib import Path

    import pytest

    from app.infrastructure.rl.white_demonstrations import load_white_win_transitions

    with pytest.raises(FileNotFoundError, match="white-win scoresheets"):
        load_white_win_transitions(Path("/no/such/white-scoresheets"))


def test_teacher_book_prefers_stem_and_matches_opening() -> None:
    from pathlib import Path

    from app.infrastructure.rl.white_demonstrations import load_teacher_book
    from quoridor.domain.actions import Move
    from quoridor.domain.state import initial_state

    black = Path(__file__).parent / "fixtures" / "black_win_vs_normal_m14.txt"
    white = Path(__file__).parent / "fixtures" / "white_win_vs_random.txt"
    book = load_teacher_book(black_source=black, white_source=white)
    opening = initial_state()
    teacher = book.action_for(opening, "black")
    assert isinstance(teacher, Move)
    assert teacher.to == (1, 4)
    assert book.matches(opening, "black", Move(direction="up", to=(1, 4)))
    assert not book.matches(opening, "white", Move(direction="up", to=(1, 4)))


def test_teacher_book_prefer_stem_overwrites_conflicts(tmp_path) -> None:
    from pathlib import Path

    from app.infrastructure.rl.hunt_black_wins import parse_scoresheet, resolve_prefix_action
    from app.infrastructure.rl.white_demonstrations import load_teacher_book
    from quoridor.domain.actions import Move
    from quoridor.domain.game import Game

    fixture_dir = Path(__file__).parent / "fixtures" / "black_wins_vs_400ms_pawn"
    other_src = next(fixture_dir.glob("*M14_M15_M16.txt"))
    main_src = next(fixture_dir.glob("*M14_M15_M25.txt"))
    # Sorted ingest would let zzz_other win the shared prefix; prefer-stem must undo that.
    (tmp_path / "aaa_M14_M15_M25.txt").write_text(main_src.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "zzz_other.txt").write_text(other_src.read_text(encoding="utf-8"), encoding="utf-8")

    book = load_teacher_book(black_source=tmp_path, black_prefer_stem="M14_M15_M25")
    game = Game.from_initial()
    for spec in parse_scoresheet(main_src.read_text(encoding="utf-8"))[:4]:
        action = resolve_prefix_action(game.state, spec)
        assert action is not None
        game.play(action)
    teacher = book.action_for(game.state, "black")
    assert isinstance(teacher, Move)
    assert teacher.to == (2, 5)

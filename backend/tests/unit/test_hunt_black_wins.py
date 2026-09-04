from app.infrastructure.rl.hunt_black_wins import (
    apply_prefix,
    encode_prefix_action,
    play_opening_vs_normal,
    resolve_prefix_action,
)
from quoridor.domain.actions import Move
from quoridor.domain.state import initial_state
from quoridor.rules import get_legal_actions


def test_resolve_prefix_matches_first_forward_move() -> None:
    state = initial_state()
    forward = next(
        action
        for action in get_legal_actions(state)
        if isinstance(action, Move) and action.to == (1, 4)
    )
    assert resolve_prefix_action(state, encode_prefix_action(forward)) == forward


def test_apply_prefix_second_pawn_move_by_destination() -> None:
    after = apply_prefix((("M", 1, 4), ("M", 7, 4), ("M", 1, 5)))
    assert after is not None
    assert after.black == (1, 5)
    assert after.white == (7, 4)


def test_illegal_prefix_is_reported_without_search() -> None:
    result = play_opening_vs_normal(((("M", 8, 8),), 8, "normal", "bad"))
    assert result.winner == "illegal"
    assert result.plies == 0


def test_factory_five_tuple_illegal_prefix_skips_search() -> None:
    result = play_opening_vs_normal(((("M", 8, 8),), 8, "normal", "factory", "bad"))
    assert result.winner == "illegal"
    assert result.plies == 0


def test_factory_normal_policy_is_live_400ms() -> None:
    from app.infrastructure.ai.minimax import NormalMinimaxPolicy
    from app.infrastructure.rl.hunt_black_wins import _factory_normal_policy

    policy = _factory_normal_policy()
    assert isinstance(policy, NormalMinimaxPolicy)
    assert policy.config.time_budget_ms == 400
    assert policy.config.max_nodes == 1200


def test_replay_recorded_black_win_vs_normal() -> None:
    from pathlib import Path

    from app.infrastructure.rl.hunt_black_wins import parse_scoresheet, replay_scoresheet

    fixture = Path(__file__).parent / "fixtures" / "black_win_vs_normal_h73.txt"
    text = fixture.read_text(encoding="utf-8")
    assert len(parse_scoresheet(text)) == 79
    result = replay_scoresheet(text)
    assert result.winner == "black"
    assert result.plies == 79


def test_m14_scoresheet_is_the_hard_opening_line() -> None:
    from pathlib import Path

    from app.infrastructure.rl.hunt_black_wins import parse_scoresheet

    fixture = Path(__file__).parent / "fixtures" / "black_win_vs_normal_m14.txt"
    specs = parse_scoresheet(fixture.read_text(encoding="utf-8"))
    assert len(specs) == 63
    assert specs[0] == ("M", 1, 4)
    assert specs[1] == ("H", 1, 3)
    assert specs[2] == ("M", 1, 5)
    assert specs[3] == ("M", 8, 5)


def test_replay_ten_black_wins_vs_400ms_factory() -> None:
    from pathlib import Path

    from app.infrastructure.rl.hunt_black_wins import parse_scoresheet, replay_scoresheet

    fixtures = sorted((Path(__file__).parent / "fixtures" / "black_wins_vs_400ms").glob("*.txt"))
    assert len(fixtures) == 10
    firsts: set[tuple] = set()
    for path in fixtures:
        text = path.read_text(encoding="utf-8")
        specs = parse_scoresheet(text)
        assert specs, path.name
        firsts.add(specs[0])
        result = replay_scoresheet(text)
        assert result.winner == "black", path.name
        assert result.plies == len(specs), path.name
    assert ("M", 1, 4) in firsts
    assert len(firsts) == 10

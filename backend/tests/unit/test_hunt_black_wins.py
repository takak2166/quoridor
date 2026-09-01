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

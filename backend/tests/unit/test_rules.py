import pytest

from quoridor.domain.actions import Move, WallSlot, encode
from quoridor.domain.state import GOAL_ROW, QuoridorState, empty_walls, initial_state
from quoridor.rules import (
    AmbiguousMoveError,
    get_legal_actions,
    is_action_legal,
    move_destinations,
    resolve_move,
)
from tests.unit.fixtures.plan_fixtures import B1_CASES, B4_CASE, JUMP_CASES, WALL_STEP_CASES


@pytest.mark.parametrize("case", JUMP_CASES, ids=lambda c: c["id"])
def test_jump_fixtures(case: dict) -> None:
    dests = move_destinations(case["state"], case["direction"])
    assert dests == case["expected_destinations"]


@pytest.mark.parametrize("case", B1_CASES, ids=lambda c: c["id"])
def test_b1_fixtures(case: dict) -> None:
    assert is_action_legal(case["state"], case["action"]) == case["expected_legal"]


def test_b4_fixture() -> None:
    assert is_action_legal(B4_CASE["state"], B4_CASE["action"]) == B4_CASE["expected_legal"]


@pytest.mark.parametrize("case", WALL_STEP_CASES, ids=lambda c: c["id"])
def test_wall_step_fixtures(case: dict) -> None:
    dests = move_destinations(case["state"], case["direction"])
    assert dests == case["expected_destinations"]


def test_goal_white() -> None:
    from quoridor.domain.state import QuoridorState, empty_walls
    from quoridor.rules import check_winner

    state = QuoridorState(
        white=(0, 4),
        black=(4, 4),
        white_walls_remaining=10,
        black_walls_remaining=10,
        horizontal_walls=empty_walls(),
        vertical_walls=empty_walls(),
        current_player="white",
    )
    assert check_winner(state) == "white"
    assert GOAL_ROW["white"] == 0


def test_goal_black() -> None:
    from quoridor.domain.state import QuoridorState, empty_walls
    from quoridor.rules import check_winner

    state = QuoridorState(
        white=(4, 4),
        black=(8, 4),
        white_walls_remaining=10,
        black_walls_remaining=10,
        horizontal_walls=empty_walls(),
        vertical_walls=empty_walls(),
        current_player="black",
    )
    assert check_winner(state) == "black"
    assert GOAL_ROW["black"] == 8


def test_resolve_move_single_dest_allows_omitted_to() -> None:
    case = next(c for c in JUMP_CASES if c["id"] == "J.1-STRAIGHT")
    move = resolve_move(case["state"], "up")
    assert move.to == (3, 4)


def test_resolve_move_ambiguous_requires_to() -> None:
    case = next(c for c in JUMP_CASES if c["id"] == "J.2-DIAG-BOTH")
    with pytest.raises(AmbiguousMoveError, match="ambiguous move"):
        resolve_move(case["state"], "up")


def test_legal_actions_returns_distinct_moves_per_destination() -> None:
    case = next(c for c in JUMP_CASES if c["id"] == "J.2-DIAG-BOTH")
    legal = [a for a in get_legal_actions(case["state"]) if isinstance(a, Move)]
    up_moves = [a for a in legal if a.direction == "up"]
    assert len(up_moves) == 2
    assert {m.to for m in up_moves} == {(4, 3), (4, 5)}


def test_is_action_legal_matches_direction_and_to() -> None:
    case = next(c for c in JUMP_CASES if c["id"] == "J.2-DIAG-BOTH")
    assert is_action_legal(case["state"], Move(direction="up", to=(4, 5)))
    assert is_action_legal(case["state"], Move(direction="up", to=(4, 3)))
    assert not is_action_legal(case["state"], Move(direction="up", to=None))


def test_action_mask_allows_ambiguous_encode_index() -> None:
    import numpy as np

    case = next(c for c in JUMP_CASES if c["id"] == "J.2-DIAG-BOTH")
    legal = get_legal_actions(case["state"])
    mask = np.zeros(132, dtype=bool)
    for action in legal:
        mask[encode(action)] = True
    unique_indices = len({encode(a) for a in legal})
    assert mask.sum() == unique_indices
    assert mask.sum() < len(legal)


def test_walls_remaining_zero_rejects_wall() -> None:
    state = QuoridorState(
        white=(8, 4),
        black=(0, 4),
        white_walls_remaining=0,
        black_walls_remaining=10,
        horizontal_walls=empty_walls(),
        vertical_walls=empty_walls(),
        current_player="white",
    )
    wall = WallSlot(orientation="horizontal", row=3, col=3)
    assert not is_action_legal(state, wall)


def test_action_mask_shape_initial_state() -> None:
    import numpy as np

    state = initial_state()
    legal = get_legal_actions(state)
    mask = np.zeros(132, dtype=bool)
    for action in legal:
        mask[encode(action)] = True
    assert mask.shape == (132,)
    assert mask.sum() == len({encode(a) for a in legal})

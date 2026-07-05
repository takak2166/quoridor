from dataclasses import replace

import pytest

from quoridor.domain.actions import WallSlot
from quoridor.domain.state import QuoridorState, initial_state
from quoridor.pathfinding import SimpleDistanceCache, both_reachable, both_reachable_one_pass, can_reach_goal
from quoridor.rules import _wall_maintains_paths, _wall_passes_filters, get_legal_actions
from tests.unit.fixtures.plan_fixtures import B1_CASES, B4_CASE, JUMP_CASES, build_state


def test_both_reachable_one_pass_does_not_loop_after_goal_found() -> None:
    """Regression: clearing the queue when a goal is found avoids an infinite loop."""
    state = replace(
        build_state(
            white=(8, 5),
            black=(0, 4),
            current="black",
            h=frozenset((0, c) for c in range(6)),
        ),
        black_walls_remaining=1,
        white_walls_remaining=2,
    )
    temp = state.with_wall("vertical", 0, 6)
    assert both_reachable_one_pass(temp) is False
    assert both_reachable(temp) is False
    get_legal_actions(state)


def test_both_reachable_matches_separate_bfs() -> None:
    states: list[QuoridorState] = [initial_state(), *[case["state"] for case in JUMP_CASES], B4_CASE["state"]]
    for state in states:
        expected = can_reach_goal(state, "white") and can_reach_goal(state, "black")
        assert both_reachable_one_pass(state) == expected
        assert both_reachable(state) == expected


def _all_wall_slots() -> list[WallSlot]:
    slots: list[WallSlot] = []
    for row in range(8):
        for col in range(8):
            slots.append(WallSlot(orientation="horizontal", row=row, col=col))
            slots.append(WallSlot(orientation="vertical", row=row, col=col))
    return slots


def _legal_walls_without_geometry(state: QuoridorState) -> set[WallSlot]:
    cache = SimpleDistanceCache()
    legal: set[WallSlot] = set()
    for wall in _all_wall_slots():
        if not _wall_passes_filters(state, wall):
            continue
        temp = state.with_wall(wall.orientation, wall.row, wall.col)
        if both_reachable(temp, cache=cache):
            legal.add(wall)
    return legal


def test_cannot_build_full_horizontal_barrier_far_from_pawns() -> None:
    """Regression: path checks must not be skipped based on pawn corridor heuristics."""
    state = build_state(white=(8, 4), black=(8, 6), current="black")
    temp = state
    for col in range(7):
        wall = WallSlot("horizontal", 5, col)
        assert _wall_maintains_paths(temp, wall)
        temp = temp.with_wall("horizontal", 5, col)
    assert can_reach_goal(temp, "white")
    assert can_reach_goal(temp, "black")
    assert not _wall_maintains_paths(temp, WallSlot("horizontal", 5, 7))
    assert not can_reach_goal(temp.with_wall("horizontal", 5, 7), "white")


@pytest.mark.parametrize(
    "state",
    [
        initial_state(),
        build_state(white=(8, 4), black=(8, 6), current="black"),
        *[c["state"] for c in B1_CASES],
        B4_CASE["state"],
    ],
)
def test_wall_candidate_reduction_preserves_legal(state: QuoridorState) -> None:
    reference = _legal_walls_without_geometry(state)
    optimized = {action for action in get_legal_actions(state) if isinstance(action, WallSlot)}
    assert optimized == reference

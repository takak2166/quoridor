"""Wall placement path checks and pawn passability (TAK-101 regression coverage)."""

from __future__ import annotations

from collections import deque

import pytest

from quoridor.pathfinding import (
    _bfs_distance,
    _expand_bfs_neighbors,
    both_reachable,
    can_reach_goal,
)
from quoridor.rules import _wall_maintains_paths, is_action_legal
from tests.unit.fixtures.plan_fixtures import PF_CASES, WALL_PATH_CASES, build_state


def _strict_both_reachable(state) -> bool:
    return can_reach_goal(state, "white") and can_reach_goal(state, "black")


def _bfs_visits_opponent_cell(state, color: str, *, ghost_opponent: bool) -> bool:
    opponent = state.pawn("black" if color == "white" else "white")
    start = state.pawn(color)
    visited: set[tuple[int, int]] = {start}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        pos = queue.popleft()
        if pos == opponent:
            return True
        for nxt in _expand_bfs_neighbors(
            state,
            color,
            pos,
            opponent,
            ghost_opponent=ghost_opponent,
        ):
            if nxt in visited:
                continue
            visited.add(nxt)
            queue.append(nxt)
    return False


@pytest.mark.parametrize("case", WALL_PATH_CASES, ids=lambda c: c["id"])
def test_wall_path_fixtures(case: dict) -> None:
    state = case["state"]
    wall = case["wall"]
    expected = case["expected_legal"]
    assert _wall_maintains_paths(state, wall) == expected
    assert is_action_legal(state, wall) == expected


@pytest.mark.parametrize("case", WALL_PATH_CASES, ids=lambda c: c["id"])
def test_wall_path_matches_strict_reachability(case: dict) -> None:
    state = case["state"]
    wall = case["wall"]
    temp = state.with_wall(wall.orientation, wall.row, wall.col)
    assert _wall_maintains_paths(state, wall) == _strict_both_reachable(temp)
    assert both_reachable(temp) == _strict_both_reachable(temp)


def test_adjacent_opponent_cell_not_neighbor_in_strict_wall_bfs() -> None:
    """PF.1: adjacent pawn blocks stepping onto the opponent cell during path checks."""
    state = build_state(white=(4, 4), black=(4, 5), current="white")
    opponent = state.black
    neighbors = _expand_bfs_neighbors(
        state,
        "white",
        state.white,
        opponent,
        ghost_opponent=False,
    )
    assert opponent not in neighbors


def test_non_adjacent_opponent_traversable_only_in_evaluation_bfs() -> None:
    """PF.4: evaluation BFS may cross a non-adjacent opponent; strict BFS must not."""
    state = build_state(white=(5, 4), black=(3, 4), current="white")
    assert _bfs_visits_opponent_cell(state, "white", ghost_opponent=True)
    assert not _bfs_visits_opponent_cell(state, "white", ghost_opponent=False)
    assert _bfs_distance(state, "white", for_evaluation=True) == 5
    assert _bfs_distance(state, "white", for_evaluation=False) == 4


def test_wall_path_check_uses_strict_not_evaluation_bfs() -> None:
    """Wall legality must follow strict reachability, not evaluation ghost traversal."""
    pf4 = next(c for c in PF_CASES if c["id"] == "PF.4-GHOST-NON-ADJ")
    state = pf4["state"]
    assert _bfs_distance(state, "white", for_evaluation=True) != _bfs_distance(
        state, "white", for_evaluation=False
    )
    assert both_reachable(state) == _strict_both_reachable(state)

    corridor = next(c for c in WALL_PATH_CASES if c["id"] == "W.PATH-NON-ADJ-LEGAL")
    wall = corridor["wall"]
    temp = corridor["state"].with_wall(wall.orientation, wall.row, wall.col)
    assert _wall_maintains_paths(corridor["state"], wall)
    assert both_reachable(temp) == _strict_both_reachable(temp)


def test_barrier_last_wall_illegal_despite_pawns_far_from_wall_row() -> None:
    """Regression: path checks must run even when pawns are far from the candidate wall."""
    case = next(c for c in WALL_PATH_CASES if c["id"] == "W.PATH-BARRIER-LAST-ILLEGAL")
    state = case["state"]
    wall = case["wall"]
    temp = state.with_wall(wall.orientation, wall.row, wall.col)
    assert can_reach_goal(state, "white")
    assert can_reach_goal(state, "black")
    assert not can_reach_goal(temp, "white")
    assert can_reach_goal(temp, "black")
    assert not _wall_maintains_paths(state, wall)

import pytest

from quoridor.domain.actions import WallSlot
from quoridor.domain.state import initial_state
from quoridor.pathfinding import SimpleDistanceCache, _bfs_distance, can_reach_goal, distances
from quoridor.rules import apply_action, get_legal_actions
from tests.unit.fixtures.plan_fixtures import PF_CASES


@pytest.mark.parametrize("case", PF_CASES, ids=lambda c: c["id"])
def test_bfs_distance(case: dict) -> None:
    dist = _bfs_distance(case["state"], case["color"], for_evaluation=True)
    assert dist == case["expected_distance"]


@pytest.mark.parametrize("case", PF_CASES, ids=lambda c: c["id"])
def test_bfs_distance_strict_matches_reachability(case: dict) -> None:
    dist = _bfs_distance(case["state"], case["color"], for_evaluation=False)
    if case["expected_distance"] is None:
        assert dist is None
    else:
        assert dist is not None


@pytest.mark.parametrize("case", PF_CASES, ids=lambda c: c["id"])
def test_can_reach_goal(case: dict) -> None:
    assert can_reach_goal(case["state"], case["color"]) == (case["expected_distance"] is not None)


def test_legal_wall_scan_does_not_poison_distance_cache() -> None:
    """both_reachable used to cache success as (0, 0), so wall deltas became 0."""
    state = apply_action(initial_state(), WallSlot("horizontal", 4, 4))
    cache = SimpleDistanceCache()
    legal = get_legal_actions(state, dist_cache=cache)
    wall = next(action for action in legal if isinstance(action, WallSlot))
    after = state.with_wall(wall.orientation, wall.row, wall.col)
    assert distances(after, cache) == distances(after)
    assert distances(after, cache) != (0, 0)

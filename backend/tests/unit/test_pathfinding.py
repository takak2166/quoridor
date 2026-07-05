import pytest

from quoridor.pathfinding import _bfs_distance, can_reach_goal
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

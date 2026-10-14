import time

import pytest

from quoridor.domain.actions import WallSlot
from quoridor.domain.game import Game
from quoridor.domain.state import initial_state
from quoridor.rules import get_legal_actions
from tests.unit.fixtures.plan_fixtures import seed3_wall_separated_choke


def _p99_ms(samples: list[float]) -> float:
    samples.sort()
    return samples[int(len(samples) * 0.99)]


@pytest.mark.slow
def test_legal_moves_perf_typical() -> None:
    state = initial_state()
    samples: list[float] = []
    for _ in range(100):
        get_legal_actions(state)
    for _ in range(2000):
        start = time.perf_counter()
        get_legal_actions(state)
        samples.append((time.perf_counter() - start) * 1000)
    p99 = _p99_ms(samples)
    assert p99 < 50.0, f"P99 {p99}ms exceeds threshold"


@pytest.mark.slow
def test_legal_moves_perf_with_walls() -> None:
    """Many walls force path BFS on each wall candidate (not the empty-board fast path)."""
    state = seed3_wall_separated_choke()
    samples: list[float] = []
    for _ in range(100):
        get_legal_actions(state)
    for _ in range(2000):
        start = time.perf_counter()
        get_legal_actions(state)
        samples.append((time.perf_counter() - start) * 1000)
    p99 = _p99_ms(samples)
    assert p99 < 300.0, f"P99 {p99}ms exceeds threshold"


@pytest.mark.slow
def test_game_play_perf_wall_with_walls() -> None:
    state = seed3_wall_separated_choke()
    legal = get_legal_actions(state)
    wall = next(a for a in legal if isinstance(a, WallSlot))
    samples: list[float] = []
    for _ in range(100):
        Game(state=state.copy()).play(wall)
    for _ in range(2000):
        game = Game(state=state.copy())
        start = time.perf_counter()
        game.play(wall)
        samples.append((time.perf_counter() - start) * 1000)
    p99 = _p99_ms(samples)
    assert p99 < 15.0, f"P99 {p99}ms exceeds threshold"

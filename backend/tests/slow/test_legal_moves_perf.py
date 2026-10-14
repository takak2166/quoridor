"""Slow perf tests for legal move generation.

Reuses ``tests.unit.fixtures`` board builders intentionally (see ``seed3_wall_separated_choke``).
"""

import time

import pytest

from quoridor.domain.actions import WallSlot
from quoridor.domain.game import Game
from quoridor.domain.state import initial_state
from quoridor.rules import get_legal_actions
from tests.unit.fixtures.plan_fixtures import seed3_wall_separated_choke

# P99 ceilings (ms), from 2000 timed samples after 100 warmup on the post–Game.play-cleanup branch:
# empty board get_legal_actions ~5 ms p99 -> 50 (~10x headroom);
# seed3 get_legal_actions ~209 ms p99 -> 300 (~1.4x);
# seed3 Game.play(one legal wall) ~3.8 ms p99 -> 15 (~4x).
_P99_MS_TYPICAL = 50.0
_P99_MS_WITH_WALLS = 300.0
_P99_MS_GAME_PLAY_WALL = 15.0


def _p99_ms(samples: list[float]) -> float:
    if not samples:
        return 0.0
    samples.sort()
    p99_idx = min(int(len(samples) * 0.99), len(samples) - 1)
    return samples[p99_idx]


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
    assert p99 < _P99_MS_TYPICAL, f"P99 {p99}ms exceeds threshold"


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
    assert p99 < _P99_MS_WITH_WALLS, f"P99 {p99}ms exceeds threshold"


@pytest.mark.slow
def test_game_play_perf_wall_with_walls() -> None:
    """Times a single legal wall via Game.play (one apply_action path check), not full get_legal_actions."""
    state = seed3_wall_separated_choke()
    legal = get_legal_actions(state)
    wall_candidates = [a for a in legal if isinstance(a, WallSlot)]
    assert wall_candidates, "seed3_wall_separated_choke must include at least one legal wall"
    wall = wall_candidates[0]
    samples: list[float] = []
    for _ in range(100):
        Game(state=state.copy()).play(wall)
    for _ in range(2000):
        game = Game(state=state.copy())
        start = time.perf_counter()
        game.play(wall)
        samples.append((time.perf_counter() - start) * 1000)
    p99 = _p99_ms(samples)
    assert p99 < _P99_MS_GAME_PLAY_WALL, f"P99 {p99}ms exceeds threshold"

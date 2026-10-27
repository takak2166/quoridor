"""Slow perf tests for legal move generation.

Reuses ``tests.unit.fixtures`` board builders intentionally (see ``seed3_wall_separated_choke``).
"""

import time

import pytest

from quoridor.domain.actions import WallSlot
from quoridor.domain.game import Game
from quoridor.domain.state import QuoridorState, initial_state
from quoridor.rules import get_legal_actions
from tests.unit.fixtures.plan_fixtures import seed3_wall_separated_choke

# P99 ceilings (ms), from 2000 timed samples after 100 warmup (100+2000 per test):
# Calibrated on ubuntu24 self-hosted worker via `make test-slow` (pytest -m slow);
# see commit SHA in git blame on this block when revisiting thresholds.
# empty board get_legal_actions ~5 ms p99 -> 50 (~10x headroom);
# seed3 get_legal_actions ~209 ms p99 -> 300 (~1.4x);
# seed3 Game.play(one legal wall) ~3.8 ms p99 -> 15 (~4x).
_P99_MS_TYPICAL = 50.0
_P99_MS_WITH_WALLS = 300.0
_P99_MS_GAME_PLAY_WALL = 15.0

# Fixed legal wall on seed3_wall_separated_choke (not get_legal_actions enumeration order).
_PERF_WALL_SEED3 = WallSlot("horizontal", 3, 4)


def _state_has_placed_walls(state: QuoridorState) -> bool:
    return any(any(row) for row in state.horizontal_walls) or any(
        any(row) for row in state.vertical_walls
    )


def _p99_ms(samples: list[float]) -> float:
    assert samples, "perf samples must not be empty"
    sorted_samples = sorted(samples)
    p99_idx = min(int(len(sorted_samples) * 0.99), len(sorted_samples) - 1)
    return sorted_samples[p99_idx]


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
    assert p99 < _P99_MS_TYPICAL, f"p99={p99:.1f}ms > {_P99_MS_TYPICAL}ms"


@pytest.mark.slow
def test_legal_moves_perf_with_walls() -> None:
    """Many walls force path BFS on each wall candidate (not the empty-board fast path)."""
    state = seed3_wall_separated_choke()
    assert _state_has_placed_walls(state), (
        "seed3_wall_separated_choke must include placed walls for BFS path checks"
    )
    samples: list[float] = []
    for _ in range(100):
        get_legal_actions(state)
    for _ in range(2000):
        start = time.perf_counter()
        get_legal_actions(state)
        samples.append((time.perf_counter() - start) * 1000)
    p99 = _p99_ms(samples)
    assert p99 < _P99_MS_WITH_WALLS, f"p99={p99:.1f}ms > {_P99_MS_WITH_WALLS}ms"


@pytest.mark.slow
def test_game_play_perf_wall_with_walls() -> None:
    """Times a single legal wall via Game.play (one apply_action path check), not full get_legal_actions."""
    state = seed3_wall_separated_choke()
    assert _state_has_placed_walls(state), (
        "seed3_wall_separated_choke must include placed walls for BFS path checks"
    )
    legal = get_legal_actions(state)
    assert _PERF_WALL_SEED3 in legal, f"{_PERF_WALL_SEED3} must be legal on seed3 choke board"
    wall = _PERF_WALL_SEED3
    samples: list[float] = []
    for _ in range(100):
        Game(state=state.copy()).play(wall)
    for _ in range(2000):
        game = Game(state=state.copy())
        start = time.perf_counter()
        game.play(wall)
        samples.append((time.perf_counter() - start) * 1000)
    p99 = _p99_ms(samples)
    assert p99 < _P99_MS_GAME_PLAY_WALL, f"p99={p99:.1f}ms > {_P99_MS_GAME_PLAY_WALL}ms"

"""Potential-based reward shaping for Quoridor RL."""

from __future__ import annotations

from quoridor.domain.state import Color, QuoridorState
from quoridor.pathfinding import DistanceCache, distances

# Denominator for Φ = (enemy_dist − agent_dist) / scale.
# Smaller scale → larger per-step shaping magnitude (stronger distance signal).
DEFAULT_POTENTIAL_SCALE = 8.0


def potential(
    state: QuoridorState,
    agent_color: Color,
    cache: DistanceCache | None,
    *,
    terminated: bool = False,
    scale: float = DEFAULT_POTENTIAL_SCALE,
) -> float:
    if terminated:
        return 0.0
    if scale <= 0:
        raise ValueError(f"potential scale must be positive, got {scale}")

    dist_white, dist_black = distances(state, cache)
    agent_dist = dist_white if agent_color == "white" else dist_black
    enemy_dist = dist_black if agent_color == "white" else dist_white

    if agent_dist is None:
        return -1.0
    if enemy_dist is None:
        return 1.0
    return (enemy_dist - agent_dist) / scale


def shaped_step_reward(
    *,
    state_before: QuoridorState,
    state_after: QuoridorState,
    agent_color: Color,
    cache: DistanceCache | None,
    gamma: float,
    terminal_reward: float,
    terminated: bool,
    potential_scale: float = DEFAULT_POTENTIAL_SCALE,
) -> float:
    phi_before = potential(
        state_before,
        agent_color,
        cache,
        terminated=False,
        scale=potential_scale,
    )
    phi_after = potential(
        state_after,
        agent_color,
        cache,
        terminated=terminated,
        scale=potential_scale,
    )
    return terminal_reward + gamma * phi_after - phi_before

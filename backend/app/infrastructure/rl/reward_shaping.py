"""Potential-based reward shaping and revisit penalties for Quoridor RL."""

from __future__ import annotations

from quoridor.domain.state import Color, QuoridorState
from quoridor.pathfinding import DistanceCache, distances

# Denominator for Φ = (enemy_dist − agent_dist) / scale.
DEFAULT_POTENTIAL_SCALE = 8.0

# Time-decayed revisit penalty: −α * γ_r^age for revisiting a recent cell.
DEFAULT_REVISIT_ALPHA = 0.15
DEFAULT_REVISIT_DECAY = 0.5
DEFAULT_REVISIT_MAX_AGE = 4


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


def revisit_penalty(
    cell: tuple[int, int],
    path: list[tuple[int, int]],
    *,
    alpha: float = DEFAULT_REVISIT_ALPHA,
    decay: float = DEFAULT_REVISIT_DECAY,
    max_age: int = DEFAULT_REVISIT_MAX_AGE,
) -> float:
    """Return a non-positive penalty if ``cell`` appears in recent ``path``.

    ``path`` is oldest→newest and should already include prior positions only
    (caller appends ``cell`` after computing the penalty). Age 0 = most recent
    prior visit. Penalty is ``−α * decay**age`` for the nearest visit within
    ``max_age``; otherwise 0.
    """
    if alpha == 0 or max_age <= 0 or not path:
        return 0.0
    # Search newest first.
    limit = min(max_age, len(path))
    for age in range(limit):
        if path[-(age + 1)] == cell:
            return -float(alpha) * (float(decay) ** age)
    return 0.0

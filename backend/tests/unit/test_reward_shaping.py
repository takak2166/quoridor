"""Tests for potential shaping and revisit penalty."""

from __future__ import annotations

from app.infrastructure.rl.reward_shaping import (
    potential,
    revisit_penalty,
    shaped_step_reward,
)
from quoridor.domain.actions import Move
from quoridor.domain.state import QuoridorState, empty_walls, initial_state
from quoridor.rules import apply_action


def test_potential_increases_when_black_advances() -> None:
    before = initial_state()  # black to move at (0, 4)
    after = apply_action(before, Move(direction="up", to=(1, 4)))
    phi_before = potential(before, "black", None, scale=8.0)
    phi_after = potential(after, "black", None, scale=8.0)
    assert phi_after > phi_before


def test_revisit_penalty_nearest_only() -> None:
    path = [(1, 4), (2, 4), (1, 4)]
    assert revisit_penalty((1, 4), path, alpha=0.15, decay=0.5, max_age=4) == -0.15
    assert revisit_penalty((2, 4), path, alpha=0.15, decay=0.5, max_age=4) == -0.075


def test_shaped_forward_step_is_positive_for_black() -> None:
    before = initial_state()
    after = apply_action(before, Move(direction="up", to=(1, 4)))
    reward = shaped_step_reward(
        state_before=before,
        state_after=after,
        agent_color="black",
        cache=None,
        gamma=0.99,
        terminal_reward=0.0,
        terminated=False,
        potential_scale=8.0,
    )
    assert reward > 0.0


def test_shaped_terminal_includes_potential_delta() -> None:
    before = QuoridorState(
        white=(1, 4),
        black=(5, 0),
        white_walls_remaining=10,
        black_walls_remaining=10,
        horizontal_walls=empty_walls(),
        vertical_walls=empty_walls(),
        current_player="white",
    )
    after = apply_action(before, Move(direction="up", to=(0, 4)))
    reward = shaped_step_reward(
        state_before=before,
        state_after=after,
        agent_color="white",
        cache=None,
        gamma=0.99,
        terminal_reward=1.0,
        terminated=True,
        potential_scale=8.0,
    )
    # Terminal φ_after=0 so reward = 1 - φ_before
    assert reward == 1.0 - potential(before, "white", None, scale=8.0)

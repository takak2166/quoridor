"""Tests for restored agent-frame + destination encoding."""

from __future__ import annotations

import numpy as np

from app.infrastructure.ai.action_mask import legal_action_mask_agent_frame
from app.infrastructure.rl.action_resolution import resolve_agent_index_to_action
from app.infrastructure.rl.env import QuoridorEnv
from app.infrastructure.rl.reward_shaping import revisit_penalty
from app.mappers.observation_mapper import to_observation
from quoridor.agent_frame import action_to_agent_frame, state_to_agent_frame
from quoridor.domain.actions import NUM_ACTIONS, Move, encode
from quoridor.domain.state import initial_state
from quoridor.rules import get_legal_actions


def test_num_actions_destination_space() -> None:
    assert NUM_ACTIONS == 209


def test_opening_observations_match() -> None:
    state = initial_state()
    np.testing.assert_array_equal(to_observation(state, "white"), to_observation(state, "black"))


def test_revisit_penalty_decays() -> None:
    path = [(1, 4), (2, 4), (3, 4)]
    assert revisit_penalty((3, 4), path, alpha=0.15, decay=0.5, max_age=4) == -0.15
    assert revisit_penalty((2, 4), path, alpha=0.15, decay=0.5, max_age=4) == -0.15 * 0.5
    assert revisit_penalty((0, 0), path, alpha=0.15, decay=0.5, max_age=4) == 0.0


def test_env_opening_wall_free_masks_walls() -> None:
    env = QuoridorEnv(
        agent_color="black",
        opponent="random",
        opening_wall_free_plies=4,
        max_wall_candidates=10,
        reward_shaping=False,
    )
    env.reset(options={"agent_color": "black"})
    mask = env._mask()
    from quoridor.domain.actions import decode, WallSlot

    assert all(isinstance(decode(int(i)), Move) for i in np.flatnonzero(mask))


def test_agent_frame_mask_resolves() -> None:
    state = initial_state()
    legal = get_legal_actions(state)
    mask = legal_action_mask_agent_frame(legal, "black")
    for idx in np.flatnonzero(mask):
        assert resolve_agent_index_to_action(int(idx), legal, "black") in legal


def test_black_forward_is_agent_framed() -> None:
    env = QuoridorEnv(agent_color="black", opponent="random", reward_shaping=False)
    env.reset(options={"agent_color": "black"})
    # Absolute (1,4) → agent-frame (7,4)
    assert env._mask()[encode(Move(direction="up", to=(7, 4)))]
    assert not env._mask()[encode(Move(direction="up", to=(1, 4)))]


def test_state_to_agent_frame_black_at_bottom() -> None:
    framed = state_to_agent_frame(initial_state(), "black")
    assert framed.black == (8, 4)
    assert framed.white == (0, 4)

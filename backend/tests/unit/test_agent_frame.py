"""Tests for agent-frame + relative delta action encoding."""

from __future__ import annotations

import numpy as np

from app.infrastructure.ai.action_mask import legal_action_mask_agent_frame
from app.infrastructure.rl.action_resolution import resolve_agent_index_to_action
from app.infrastructure.rl.env import QuoridorEnv
from app.infrastructure.rl.reward_shaping import revisit_penalty
from app.mappers.observation_mapper import to_observation
from quoridor.agent_frame import pawn_to_agent_frame, state_to_agent_frame
from quoridor.domain.actions import FORWARD_STEP_INDEX, NUM_ACTIONS, Move
from quoridor.domain.state import initial_state
from quoridor.rules import get_legal_actions


def test_num_actions_relative_delta_space() -> None:
    assert NUM_ACTIONS == 140


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
    from quoridor.domain.actions import is_move_index

    assert all(is_move_index(int(i)) for i in np.flatnonzero(mask))


def test_agent_frame_mask_resolves() -> None:
    state = initial_state()
    legal = get_legal_actions(state)
    from_pos = state.pawn(state.current_player)
    mask = legal_action_mask_agent_frame(legal, "black", from_pos=from_pos)
    for idx in np.flatnonzero(mask):
        assert (
            resolve_agent_index_to_action(int(idx), legal, "black", from_pos=from_pos)
            in legal
        )


def test_black_forward_is_agent_framed() -> None:
    env = QuoridorEnv(agent_color="black", opponent="random", reward_shaping=False)
    env.reset(options={"agent_color": "black"})
    # Absolute forward for black is (+1,0); agent-frame forward is always index 0.
    assert env._mask()[FORWARD_STEP_INDEX]
    # Backward step must not be the only legal opening move.
    assert env._mask().sum() >= 3


def test_forward_step_same_index_for_both_colors() -> None:
    for color in ("white", "black"):
        env = QuoridorEnv(
            agent_color=color,
            opponent="random",
            reward_shaping=False,
            opening_wall_free_plies=2,
        )
        env.reset(seed=0, options={"agent_color": color})
        assert env._mask()[FORWARD_STEP_INDEX]
        legal = get_legal_actions(env._state)
        resolved = resolve_agent_index_to_action(
            FORWARD_STEP_INDEX,
            legal,
            color,
            from_pos=env._state.pawn(color),
        )
        assert isinstance(resolved, Move)
        assert resolved.to is not None
        before = pawn_to_agent_frame(env._state.pawn(color), color)
        after = pawn_to_agent_frame(resolved.to, color)
        assert after[0] == before[0] - 1
        env.close()


def test_state_to_agent_frame_black_at_bottom() -> None:
    framed = state_to_agent_frame(initial_state(), "black")
    assert framed.black == (8, 4)
    assert framed.white == (0, 4)

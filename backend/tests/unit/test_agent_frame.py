"""Tests for agent-centric board frame transforms (direction action space)."""

from __future__ import annotations

import numpy as np

from app.infrastructure.ai.action_mask import legal_action_mask_agent_frame
from app.infrastructure.rl.env import QuoridorEnv
from app.mappers.observation_mapper import to_observation
from quoridor.agent_frame import (
    action_from_agent_frame,
    action_to_agent_frame,
    pawn_from_agent_frame,
    pawn_to_agent_frame,
    state_to_agent_frame,
    wall_to_agent_frame,
)
from quoridor.domain.actions import Move, WallSlot, encode
from quoridor.domain.state import QuoridorState, empty_walls, initial_state
from quoridor.rules import get_legal_actions


def test_pawn_flip_roundtrip() -> None:
    for pos in ((0, 4), (8, 4), (3, 2), (5, 7)):
        assert pawn_from_agent_frame(pawn_to_agent_frame(pos, "black"), "black") == pos
        assert pawn_to_agent_frame(pos, "white") == pos


def test_wall_flip_roundtrip() -> None:
    wall = WallSlot(orientation="horizontal", row=2, col=3)
    framed = wall_to_agent_frame(wall, "black")
    assert framed.row == 5
    assert action_from_agent_frame(framed, "black") == wall
    assert wall_to_agent_frame(wall, "white") == wall


def test_move_directions_pass_through() -> None:
    for direction in ("up", "down", "left", "right"):
        move = Move(direction=direction)  # type: ignore[arg-type]
        assert action_to_agent_frame(move, "black") is move or action_to_agent_frame(
            move, "black"
        ) == move
        assert encode(action_to_agent_frame(move, "black")) == encode(move)


def test_state_to_agent_frame_puts_black_at_bottom() -> None:
    state = initial_state()
    framed = state_to_agent_frame(state, "black")
    assert framed.black == (8, 4)
    assert framed.white == (0, 4)
    assert state_to_agent_frame(state, "white") is state


def test_opening_observations_match_for_both_colors() -> None:
    state = initial_state()
    obs_w = to_observation(state, "white")
    obs_b = to_observation(state, "black")
    np.testing.assert_array_equal(obs_w, obs_b)
    assert obs_w[134] == 1.0
    # Agent pawn at bottom row 8 in both frames.
    assert obs_w[0] == 1.0
    assert obs_w[1] == 4.0 / 8.0


def test_agent_frame_mask_covers_all_legal() -> None:
    state = initial_state()
    assert state.current_player == "black"
    legal = get_legal_actions(state)
    mask = legal_action_mask_agent_frame(legal, "black")
    assert int(mask.sum()) == len({encode(action_to_agent_frame(a, "black")) for a in legal})


def test_env_black_wall_step_uses_agent_frame_index() -> None:
    env = QuoridorEnv(agent_color="black", opponent="random")
    env.reset(options={"agent_color": "black"})
    legal = get_legal_actions(env._state, dist_cache=env._cache)
    wall = next(
        a
        for a in legal
        if isinstance(a, WallSlot) and a.orientation == "horizontal" and a.row == 0
    )
    framed_idx = encode(action_to_agent_frame(wall, "black"))
    assert env._mask()[framed_idx]
    assert framed_idx != encode(wall)
    env.step(framed_idx)
    assert env._state.horizontal_walls[wall.row][wall.col] is True


def test_state_walls_flip_with_viewer() -> None:
    h = [list(row) for row in empty_walls()]
    h[1][2] = True
    state = QuoridorState(
        white=(8, 4),
        black=(0, 4),
        white_walls_remaining=10,
        black_walls_remaining=9,
        horizontal_walls=tuple(tuple(r) for r in h),
        vertical_walls=empty_walls(),
        current_player="black",
    )
    framed = state_to_agent_frame(state, "black")
    assert framed.horizontal_walls[6][2] is True
    assert framed.horizontal_walls[1][2] is False

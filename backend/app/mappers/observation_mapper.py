from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quoridor.agent_frame import state_to_agent_frame
from quoridor.domain.state import Color, QuoridorState

# Last channel: unused (0) unless second_player_bit is on, then 1 for White.
SECOND_PLAYER_OBS_INDEX = 134


def to_observation(
    state: QuoridorState,
    agent_color: Color,
    *,
    second_player_bit: bool | None = None,
) -> NDArray[np.float32]:
    """Build a 135-d observation in the agent frame (goal always toward row 0)."""
    if second_player_bit is None:
        from app.config import settings

        second_player_bit = settings.second_player_obs_bit
    framed = state_to_agent_frame(state, agent_color)
    if agent_color == "white":
        player, enemy = framed.white, framed.black
        pw, ew = framed.white_walls_remaining, framed.black_walls_remaining
    else:
        player, enemy = framed.black, framed.white
        pw, ew = framed.black_walls_remaining, framed.white_walls_remaining

    obs = np.zeros(135, dtype=np.float32)
    obs[0] = player[0] / 8.0
    obs[1] = player[1] / 8.0
    obs[2] = enemy[0] / 8.0
    obs[3] = enemy[1] / 8.0
    obs[4] = pw / 10.0
    obs[5] = ew / 10.0
    idx = 6
    for row in range(8):
        for col in range(8):
            obs[idx] = 1.0 if framed.horizontal_walls[row][col] else 0.0
            idx += 1
    for row in range(8):
        for col in range(8):
            obs[idx] = 1.0 if framed.vertical_walls[row][col] else 0.0
            idx += 1
    obs[SECOND_PLAYER_OBS_INDEX] = 1.0 if second_player_bit and agent_color == "white" else 0.0
    return obs

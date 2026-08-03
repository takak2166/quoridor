from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quoridor.agent_frame import state_to_agent_frame
from quoridor.domain.state import Color, QuoridorState


def to_observation(state: QuoridorState, agent_color: Color) -> NDArray[np.float32]:
    """Build a 135-d observation in the agent frame (goal always toward row 0)."""
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
    # Agent frame is always White-like; keep a constant channel for shape stability.
    obs[134] = 1.0
    return obs

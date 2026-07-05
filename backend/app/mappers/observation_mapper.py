from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quoridor.domain.state import Color, QuoridorState


def to_observation(state: QuoridorState, agent_color: Color) -> NDArray[np.float32]:
    if agent_color == "white":
        player, enemy = state.white, state.black
        pw, ew = state.white_walls_remaining, state.black_walls_remaining
        is_white = 1.0
    else:
        player, enemy = state.black, state.white
        pw, ew = state.black_walls_remaining, state.white_walls_remaining
        is_white = 0.0

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
            obs[idx] = 1.0 if state.horizontal_walls[row][col] else 0.0
            idx += 1
    for row in range(8):
        for col in range(8):
            obs[idx] = 1.0 if state.vertical_walls[row][col] else 0.0
            idx += 1
    obs[134] = is_white
    return obs

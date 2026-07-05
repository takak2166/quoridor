from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quoridor.domain.actions import NUM_ACTIONS, Action, encode
from quoridor.domain.state import QuoridorState
from quoridor.rules import get_legal_actions


def legal_action_mask(
    state: QuoridorState,
    legal: list[Action] | None = None,
) -> NDArray[np.bool_]:
    if legal is None:
        legal = get_legal_actions(state)
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for action in legal:
        mask[encode(action)] = True
    return mask

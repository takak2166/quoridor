from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quoridor.agent_frame import action_to_agent_frame
from quoridor.domain.actions import NUM_ACTIONS, Action, encode
from quoridor.domain.state import Color, QuoridorState
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


def legal_action_mask_agent_frame(
    legal: list[Action],
    viewer: Color,
) -> NDArray[np.bool_]:
    """Mask over action indices expressed in the viewer's agent frame."""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for action in legal:
        mask[encode(action_to_agent_frame(action, viewer))] = True
    return mask

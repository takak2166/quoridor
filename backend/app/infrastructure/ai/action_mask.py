from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from app.infrastructure.ai.search_actions import search_actions
from quoridor.agent_frame import encode_for_viewer
from quoridor.domain.actions import NUM_ACTIONS, Action, encode
from quoridor.domain.state import Color, QuoridorState
from quoridor.pathfinding import DistanceCache
from quoridor.rules import get_legal_actions


def legal_actions_for_policy(
    state: QuoridorState,
    cache: DistanceCache | None,
    max_wall_candidates: int | None,
) -> list[Action]:
    legal = get_legal_actions(state, dist_cache=cache)
    if max_wall_candidates is None:
        return legal
    return search_actions(state, legal, cache, max_wall_candidates)


def legal_action_mask(
    state: QuoridorState,
    legal: list[Action] | None = None,
) -> NDArray[np.bool_]:
    if legal is None:
        legal = get_legal_actions(state)
    from_pos = state.pawn(state.current_player)
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for action in legal:
        mask[encode(action, from_pos=from_pos)] = True
    return mask


def legal_action_mask_agent_frame(
    legal: list[Action],
    viewer: Color,
    *,
    from_pos: tuple[int, int],
) -> NDArray[np.bool_]:
    """Mask over action indices expressed in the viewer's agent frame."""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for action in legal:
        mask[encode_for_viewer(action, from_pos, viewer)] = True
    return mask

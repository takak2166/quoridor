from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from app.infrastructure.ai.search_actions import search_actions
from quoridor.agent_frame import encode_for_viewer
from quoridor.domain.actions import NUM_ACTIONS, Action, Move, encode
from quoridor.domain.state import WALLS_INITIAL, Color, QuoridorState
from quoridor.pathfinding import DistanceCache
from quoridor.rules import get_legal_actions

_START_PAWN: dict[Color, tuple[int, int]] = {"white": (8, 4), "black": (0, 4)}


def estimated_agent_plies(state: QuoridorState, color: Color) -> int:
    """Lower-bound of how many actions ``color`` has already taken.

    Matches ``QuoridorEnv._agent_plies_played`` in the opening (pawn steps
    plus walls used). Used so PPO inference can apply the same
    ``opening_wall_free_plies`` mask as training.
    """
    start_row, start_col = _START_PAWN[color]
    row, col = state.pawn(color)
    pawn_steps = abs(row - start_row) + abs(col - start_col)
    walls_used = WALLS_INITIAL - state.walls_remaining(color)
    return pawn_steps + max(0, walls_used)


def filter_opening_wall_free(
    legal: list[Action],
    state: QuoridorState,
    color: Color,
    opening_wall_free_plies: int,
) -> list[Action]:
    """Drop walls until ``color`` has taken ``opening_wall_free_plies`` actions."""
    if opening_wall_free_plies <= 0:
        return legal
    if estimated_agent_plies(state, color) >= opening_wall_free_plies:
        return legal
    moves = [action for action in legal if isinstance(action, Move)]
    return moves or legal


def legal_actions_for_policy(
    state: QuoridorState,
    cache: DistanceCache | None,
    max_wall_candidates: int | None,
    *,
    color: Color | None = None,
    opening_wall_free_plies: int = 0,
) -> list[Action]:
    legal = get_legal_actions(state, dist_cache=cache)
    if max_wall_candidates is None:
        selected = legal
    else:
        selected = search_actions(state, legal, cache, max_wall_candidates)
    viewer = color if color is not None else state.current_player
    return filter_opening_wall_free(selected, state, viewer, opening_wall_free_plies)


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

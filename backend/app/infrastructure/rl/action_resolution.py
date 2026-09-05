"""Resolve policy action indices to fully specified legal actions."""

from __future__ import annotations

from quoridor.agent_frame import pawn_from_agent_frame, pawn_to_agent_frame, wall_from_agent_frame
from quoridor.domain.actions import (
    Action,
    Move,
    WallSlot,
    decode,
    is_move_index,
    move_delta,
)
from quoridor.domain.state import Color


def resolve_index_to_action(
    index: int,
    legal: list[Action],
    *,
    from_pos: tuple[int, int],
) -> Action:
    """Map a discrete absolute-delta action index to the matching entry in ``legal``."""
    if is_move_index(index):
        dr, dc = move_delta(index)
        absolute_to = (from_pos[0] + dr, from_pos[1] + dc)
        for action in legal:
            if isinstance(action, Move) and action.to == absolute_to:
                return action
        raise ValueError(f"Move delta {dr, dc} to {absolute_to} not in legal set")

    decoded = decode(index)
    if not isinstance(decoded, WallSlot):
        raise ValueError(f"Expected wall index, got {index}")
    for action in legal:
        if (
            isinstance(action, WallSlot)
            and action.orientation == decoded.orientation
            and action.row == decoded.row
            and action.col == decoded.col
        ):
            return action
    raise ValueError(f"Wall action {index} not in legal set")


def resolve_agent_index_to_action(
    index: int,
    legal: list[Action],
    viewer: Color,
    *,
    from_pos: tuple[int, int],
) -> Action:
    """Map a policy index in agent-frame space to an absolute legal action."""
    if is_move_index(index):
        dr, dc = move_delta(index)
        af_from = pawn_to_agent_frame(from_pos, viewer)
        af_to = (af_from[0] + dr, af_from[1] + dc)
        absolute_to = pawn_from_agent_frame(af_to, viewer)
        for action in legal:
            if isinstance(action, Move) and action.to == absolute_to:
                return action
        raise ValueError(
            f"Agent-frame move {index} delta {dr, dc} -> absolute {absolute_to} not in legal set"
        )

    decoded = decode(index)
    if not isinstance(decoded, WallSlot):
        raise ValueError(f"Expected wall index, got {index}")
    absolute = wall_from_agent_frame(decoded, viewer)
    for action in legal:
        if (
            isinstance(action, WallSlot)
            and action.orientation == absolute.orientation
            and action.row == absolute.row
            and action.col == absolute.col
        ):
            return action
    raise ValueError(f"Wall action {index} (absolute {absolute}) not in legal set")

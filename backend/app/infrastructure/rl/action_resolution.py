"""Resolve policy action indices to fully specified legal actions."""

from __future__ import annotations

from quoridor.agent_frame import action_from_agent_frame
from quoridor.domain.actions import Action, Move, WallSlot, decode
from quoridor.domain.state import Color


def resolve_index_to_action(index: int, legal: list[Action]) -> Action:
    """Map a discrete absolute action index to the matching entry in ``legal``."""
    decoded = decode(index)
    if isinstance(decoded, WallSlot):
        for action in legal:
            if (
                isinstance(action, WallSlot)
                and action.orientation == decoded.orientation
                and action.row == decoded.row
                and action.col == decoded.col
            ):
                return action
        raise ValueError(f"Wall action {index} not in legal set")
    if decoded.to is None:
        raise ValueError(f"Move action {index} missing destination")
    for action in legal:
        if isinstance(action, Move) and action.to == decoded.to:
            return action
    raise ValueError(f"Move to {decoded.to} not in legal set")


def resolve_agent_index_to_action(
    index: int,
    legal: list[Action],
    viewer: Color,
) -> Action:
    """Map a policy index in agent-frame space to an absolute legal action."""
    absolute = action_from_agent_frame(decode(index), viewer)
    if isinstance(absolute, WallSlot):
        for action in legal:
            if (
                isinstance(action, WallSlot)
                and action.orientation == absolute.orientation
                and action.row == absolute.row
                and action.col == absolute.col
            ):
                return action
        raise ValueError(f"Wall action {index} (absolute {absolute}) not in legal set")
    if absolute.to is None:
        raise ValueError(f"Move action {index} missing destination")
    for action in legal:
        if isinstance(action, Move) and action.to == absolute.to:
            return action
    raise ValueError(f"Move to {absolute.to} not in legal set")

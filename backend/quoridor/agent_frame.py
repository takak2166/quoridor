"""Agent-centric board frame: viewer always plays toward decreasing row (White-like).

Canonical (absolute) board: White goal row 0, Black goal row 8.
Agent frame: the acting color is mapped so its goal is always row 0.

Move *directions* are already agent-semantic via ``absolute_delta`` (up = toward
goal), so they pass through unchanged. Wall slots and pawn coordinates flip for
Black. Use the same helpers for PPO obs/action masks and UI display.
"""

from __future__ import annotations

from quoridor.domain.actions import Action, Move, WallSlot, encode
from quoridor.domain.state import Color, QuoridorState, empty_walls


def needs_flip(viewer: Color) -> bool:
    """Black's absolute orientation is inverted relative to White-like agent frame."""
    return viewer == "black"


def flip_pawn(pos: tuple[int, int]) -> tuple[int, int]:
    row, col = pos
    return (8 - row, col)


def flip_wall_row(row: int) -> int:
    """Map an 8x8 wall-slot row under vertical board flip."""
    return 7 - row


def pawn_to_agent_frame(pos: tuple[int, int], viewer: Color) -> tuple[int, int]:
    return flip_pawn(pos) if needs_flip(viewer) else pos


def pawn_from_agent_frame(pos: tuple[int, int], viewer: Color) -> tuple[int, int]:
    return flip_pawn(pos) if needs_flip(viewer) else pos


def wall_to_agent_frame(wall: WallSlot, viewer: Color) -> WallSlot:
    if not needs_flip(viewer):
        return wall
    return WallSlot(
        orientation=wall.orientation,
        row=flip_wall_row(wall.row),
        col=wall.col,
    )


def wall_from_agent_frame(wall: WallSlot, viewer: Color) -> WallSlot:
    return wall_to_agent_frame(wall, viewer)


def action_to_agent_frame(action: Action, viewer: Color) -> Action:
    """Map an absolute action into the viewer's agent frame.

    Move directions are already relative to the acting color, so only wall
    slots are transformed.
    """
    if isinstance(action, Move):
        return action
    return wall_to_agent_frame(action, viewer)


def action_from_agent_frame(action: Action, viewer: Color) -> Action:
    if isinstance(action, Move):
        return action
    return wall_from_agent_frame(action, viewer)


def action_index_to_agent_frame(index: int, viewer: Color) -> int:
    from quoridor.domain.actions import decode

    return encode(action_to_agent_frame(decode(index), viewer))


def action_index_from_agent_frame(index: int, viewer: Color) -> int:
    from quoridor.domain.actions import decode

    return encode(action_from_agent_frame(decode(index), viewer))


def _flip_wall_grid(
    walls: tuple[tuple[bool, ...], ...],
) -> tuple[tuple[bool, ...], ...]:
    grid = [list(row) for row in empty_walls()]
    for row in range(8):
        for col in range(8):
            if walls[row][col]:
                grid[flip_wall_row(row)][col] = True
    return tuple(tuple(r) for r in grid)


def state_to_agent_frame(state: QuoridorState, viewer: Color) -> QuoridorState:
    """Return a state expressed in the viewer agent frame (goal always row 0)."""
    if not needs_flip(viewer):
        return state
    return QuoridorState(
        white=flip_pawn(state.white),
        black=flip_pawn(state.black),
        white_walls_remaining=state.white_walls_remaining,
        black_walls_remaining=state.black_walls_remaining,
        horizontal_walls=_flip_wall_grid(state.horizontal_walls),
        vertical_walls=_flip_wall_grid(state.vertical_walls),
        current_player=state.current_player,
    )

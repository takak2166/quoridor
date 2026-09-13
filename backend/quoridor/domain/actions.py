from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from quoridor.domain.state import Color

Direction = Literal["up", "down", "left", "right"]

BOARD_SIZE = 9

# Agent-frame / absolute board deltas that Quoridor pawn moves can produce.
# Index 0 is always forward step toward decreasing row in agent frame.
AGENT_MOVE_DELTAS: tuple[tuple[int, int], ...] = (
    (-1, 0),  # 0  forward step
    (1, 0),  # 1  backward step
    (0, -1),  # 2  left
    (0, 1),  # 3  right
    (-2, 0),  # 4  forward jump
    (2, 0),  # 5  backward jump
    (0, -2),  # 6  left jump
    (0, 2),  # 7  right jump
    (-1, -1),  # 8  forward-left diagonal
    (-1, 1),  # 9  forward-right diagonal
    (1, -1),  # 10 backward-left diagonal
    (1, 1),  # 11 backward-right diagonal
)
DELTA_TO_INDEX: dict[tuple[int, int], int] = {
    delta: index for index, delta in enumerate(AGENT_MOVE_DELTAS)
}
FORWARD_STEP_INDEX = 0

MOVE_ACTION_COUNT = len(AGENT_MOVE_DELTAS)  # 12
H_WALL_OFFSET = MOVE_ACTION_COUNT
V_WALL_OFFSET = H_WALL_OFFSET + 64
NUM_ACTIONS = V_WALL_OFFSET + 64  # 140


@dataclass(frozen=True)
class Move:
    direction: Direction
    to: tuple[int, int] | None = None


@dataclass(frozen=True)
class WallSlot:
    orientation: Literal["horizontal", "vertical"]
    row: int
    col: int


Action = Move | WallSlot


def cell_index(row: int, col: int) -> int:
    return row * BOARD_SIZE + col


def cell_from_index(index: int) -> tuple[int, int]:
    return index // BOARD_SIZE, index % BOARD_SIZE


def is_move_index(index: int) -> bool:
    return 0 <= index < MOVE_ACTION_COUNT


def move_delta(index: int) -> tuple[int, int]:
    if not is_move_index(index):
        raise ValueError(f"Not a move action index: {index}")
    return AGENT_MOVE_DELTAS[index]


def _primary_direction(dr: int, dc: int) -> Direction:
    if abs(dr) >= abs(dc):
        return "up" if dr < 0 else "down"
    return "left" if dc < 0 else "right"


def encode(action: Action, *, from_pos: tuple[int, int] | None = None) -> int:
    """Encode an action into the discrete space.

    Moves are relative deltas from ``from_pos`` to ``action.to``.
    Walls do not need ``from_pos``.
    """
    if isinstance(action, Move):
        if action.to is None:
            raise ValueError("Move encoding requires explicit destination")
        if from_pos is None:
            raise ValueError("Move encoding requires from_pos")
        delta = (action.to[0] - from_pos[0], action.to[1] - from_pos[1])
        try:
            return DELTA_TO_INDEX[delta]
        except KeyError as exc:
            raise ValueError(f"Unsupported move delta {delta} from {from_pos} to {action.to}") from exc
    if action.orientation == "horizontal":
        return H_WALL_OFFSET + action.row * 8 + action.col
    return V_WALL_OFFSET + action.row * 8 + action.col


def decode(index: int) -> Action:
    """Decode an action index.

    For move indices, ``Move.to`` holds the **delta** ``(dr, dc)``, not a board
    cell. Resolve against a pawn position before applying to the board.
    """
    if index < H_WALL_OFFSET:
        dr, dc = AGENT_MOVE_DELTAS[index]
        return Move(direction=_primary_direction(dr, dc), to=(dr, dc))
    if index < V_WALL_OFFSET:
        idx = index - H_WALL_OFFSET
        return WallSlot(orientation="horizontal", row=idx // 8, col=idx % 8)
    idx = index - V_WALL_OFFSET
    return WallSlot(orientation="vertical", row=idx // 8, col=idx % 8)


def action_child_key(action: Action) -> tuple[int, tuple[int, int] | None]:
    """Stable MCTS child key; moves key by destination cell, walls by slot index."""
    if isinstance(action, Move):
        if action.to is None:
            return (-1, None)
        return (cell_index(*action.to), action.to)
    return (encode(action), None)


def absolute_delta(color: Color, direction: Direction) -> tuple[int, int]:
    if direction == "left":
        return (0, -1)
    if direction == "right":
        return (0, 1)
    if direction == "up":
        return (-1, 0) if color == "white" else (1, 0)
    return (1, 0) if color == "white" else (-1, 0)

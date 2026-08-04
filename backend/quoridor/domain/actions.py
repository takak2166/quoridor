from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from quoridor.domain.state import Color

Direction = Literal["up", "down", "left", "right"]

BOARD_SIZE = 9
MOVE_ACTION_COUNT = BOARD_SIZE * BOARD_SIZE  # 81 cells
H_WALL_OFFSET = MOVE_ACTION_COUNT
V_WALL_OFFSET = H_WALL_OFFSET + 64
NUM_ACTIONS = V_WALL_OFFSET + 64  # 209


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


def encode(action: Action) -> int:
    if isinstance(action, Move):
        if action.to is None:
            raise ValueError("Move encoding requires explicit destination")
        row, col = action.to
        return cell_index(row, col)
    if action.orientation == "horizontal":
        return H_WALL_OFFSET + action.row * 8 + action.col
    return V_WALL_OFFSET + action.row * 8 + action.col


def decode(index: int) -> Action:
    if index < H_WALL_OFFSET:
        row, col = cell_from_index(index)
        return Move(direction="up", to=(row, col))
    if index < V_WALL_OFFSET:
        idx = index - H_WALL_OFFSET
        return WallSlot(orientation="horizontal", row=idx // 8, col=idx % 8)
    idx = index - V_WALL_OFFSET
    return WallSlot(orientation="vertical", row=idx // 8, col=idx % 8)


def action_child_key(action: Action) -> tuple[int, tuple[int, int] | None]:
    if isinstance(action, Move):
        return (encode(action), action.to)
    return (encode(action), None)


def absolute_delta(color: Color, direction: Direction) -> tuple[int, int]:
    if direction == "left":
        return (0, -1)
    if direction == "right":
        return (0, 1)
    if direction == "up":
        return (-1, 0) if color == "white" else (1, 0)
    return (1, 0) if color == "white" else (-1, 0)

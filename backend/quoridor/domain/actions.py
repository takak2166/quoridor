from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from quoridor.domain.state import Color

Direction = Literal["up", "down", "left", "right"]


class DirectionEnum(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


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

NUM_ACTIONS = 132


def encode(action: Action) -> int:
    if isinstance(action, Move):
        mapping = {"up": 0, "down": 1, "left": 2, "right": 3}
        return mapping[action.direction]
    if action.orientation == "horizontal":
        return 4 + action.row * 8 + action.col
    return 68 + action.row * 8 + action.col


def decode(index: int) -> Action:
    if index < 4:
        dirs: list[Direction] = ["up", "down", "left", "right"]
        return Move(direction=dirs[index], to=None)
    if index < 68:
        idx = index - 4
        return WallSlot(orientation="horizontal", row=idx // 8, col=idx % 8)
    idx = index - 68
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

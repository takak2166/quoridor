from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Color = Literal["white", "black"]

WALLS_INITIAL = 10
GOAL_ROW: dict[Color, int] = {"white": 0, "black": 8}


def empty_walls() -> tuple[tuple[bool, ...], ...]:
    return tuple(tuple(False for _ in range(8)) for _ in range(8))


@dataclass(frozen=True)
class QuoridorState:
    white: tuple[int, int]
    black: tuple[int, int]
    white_walls_remaining: int
    black_walls_remaining: int
    horizontal_walls: tuple[tuple[bool, ...], ...]
    vertical_walls: tuple[tuple[bool, ...], ...]
    current_player: Color

    def copy(self) -> QuoridorState:
        return QuoridorState(
            white=self.white,
            black=self.black,
            white_walls_remaining=self.white_walls_remaining,
            black_walls_remaining=self.black_walls_remaining,
            horizontal_walls=self.horizontal_walls,
            vertical_walls=self.vertical_walls,
            current_player=self.current_player,
        )

    def pawn(self, color: Color) -> tuple[int, int]:
        return self.white if color == "white" else self.black

    def walls_remaining(self, color: Color) -> int:
        return self.white_walls_remaining if color == "white" else self.black_walls_remaining

    def with_pawn(self, color: Color, pos: tuple[int, int]) -> QuoridorState:
        if color == "white":
            return QuoridorState(
                white=pos,
                black=self.black,
                white_walls_remaining=self.white_walls_remaining,
                black_walls_remaining=self.black_walls_remaining,
                horizontal_walls=self.horizontal_walls,
                vertical_walls=self.vertical_walls,
                current_player=self.current_player,
            )
        return QuoridorState(
            white=self.white,
            black=pos,
            white_walls_remaining=self.white_walls_remaining,
            black_walls_remaining=self.black_walls_remaining,
            horizontal_walls=self.horizontal_walls,
            vertical_walls=self.vertical_walls,
            current_player=self.current_player,
        )

    def with_wall(
        self,
        orientation: Literal["horizontal", "vertical"],
        row: int,
        col: int,
    ) -> QuoridorState:
        h = [list(r) for r in self.horizontal_walls]
        v = [list(r) for r in self.vertical_walls]
        wr = self.white_walls_remaining
        br = self.black_walls_remaining
        if orientation == "horizontal":
            h[row][col] = True
        else:
            v[row][col] = True
        if self.current_player == "white":
            wr -= 1
        else:
            br -= 1
        return QuoridorState(
            white=self.white,
            black=self.black,
            white_walls_remaining=wr,
            black_walls_remaining=br,
            horizontal_walls=tuple(tuple(r) for r in h),
            vertical_walls=tuple(tuple(r) for r in v),
            current_player=self.current_player,
        )

    def switch_turn(self) -> QuoridorState:
        nxt: Color = "black" if self.current_player == "white" else "white"
        return QuoridorState(
            white=self.white,
            black=self.black,
            white_walls_remaining=self.white_walls_remaining,
            black_walls_remaining=self.black_walls_remaining,
            horizontal_walls=self.horizontal_walls,
            vertical_walls=self.vertical_walls,
            current_player=nxt,
        )


def initial_state() -> QuoridorState:
    return QuoridorState(
        white=(8, 4),
        black=(0, 4),
        white_walls_remaining=WALLS_INITIAL,
        black_walls_remaining=WALLS_INITIAL,
        horizontal_walls=empty_walls(),
        vertical_walls=empty_walls(),
        current_player="black",
    )


def position_key(state: QuoridorState) -> tuple[object, ...]:
    """Hashable key for full board position (used for repetition detection)."""
    h_bits = tuple(cell for row in state.horizontal_walls for cell in row)
    v_bits = tuple(cell for row in state.vertical_walls for cell in row)
    return (
        state.white,
        state.black,
        h_bits,
        v_bits,
        state.current_player,
        state.white_walls_remaining,
        state.black_walls_remaining,
    )


def state_hash(state: QuoridorState) -> int:
    return hash(position_key(state)[:5])

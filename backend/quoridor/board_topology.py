from __future__ import annotations

from quoridor.domain.state import QuoridorState


def _in_wall_bounds(row: int, col: int) -> bool:
    return 0 <= row <= 7 and 0 <= col <= 7


def is_horizontal_wall(state: QuoridorState, row: int, col: int) -> bool:
    if not _in_wall_bounds(row, col):
        return False
    return state.horizontal_walls[row][col]


def is_vertical_wall(state: QuoridorState, row: int, col: int) -> bool:
    if not _in_wall_bounds(row, col):
        return False
    return state.vertical_walls[row][col]


def can_step(state: QuoridorState, from_pos: tuple[int, int], to_pos: tuple[int, int]) -> bool:
    fr, fc = from_pos
    tr, tc = to_pos
    if not (0 <= tr <= 8 and 0 <= tc <= 8):
        return False
    dr, dc = tr - fr, tc - fc
    if abs(dr) + abs(dc) != 1:
        return False
    if dc != 0:
        if dc == 1:
            return not is_vertical_wall(state, fr, fc) and not is_vertical_wall(state, fr - 1, fc)
        return not is_vertical_wall(state, fr, fc - 1) and not is_vertical_wall(state, fr - 1, fc - 1)
    if dr == 1:
        return not is_horizontal_wall(state, fr, fc) and not is_horizontal_wall(state, fr, fc - 1)
    return not is_horizontal_wall(state, fr - 1, fc) and not is_horizontal_wall(state, fr - 1, fc - 1)

from __future__ import annotations

from dataclasses import dataclass

from quoridor.domain.actions import Action, Direction, Move, WallSlot
from quoridor.domain.state import GOAL_ROW, Color, QuoridorState
from quoridor.pathfinding import DistanceCache, SimpleDistanceCache, both_reachable
from quoridor.pawn_moves import step_destinations_in_direction


class AmbiguousMoveError(ValueError):
    pass


@dataclass(frozen=True)
class PlayResult:
    state: QuoridorState
    winner: Color | None


def check_winner(state: QuoridorState) -> Color | None:
    if state.white[0] == GOAL_ROW["white"]:
        return "white"
    if state.black[0] == GOAL_ROW["black"]:
        return "black"
    return None


def move_destinations(state: QuoridorState, direction: Direction) -> frozenset[tuple[int, int]]:
    color = state.current_player
    return step_destinations_in_direction(state, color, state.pawn(color), direction)


def resolve_move(
    state: QuoridorState,
    direction: Direction,
    to: tuple[int, int] | None = None,
) -> Move:
    dests = move_destinations(state, direction)
    if not dests:
        raise ValueError("illegal move")
    if len(dests) == 1:
        only = next(iter(dests))
        if to is not None and to != only:
            raise ValueError("illegal move")
        return Move(direction=direction, to=only)
    if to is None:
        raise AmbiguousMoveError("ambiguous move")
    if to not in dests:
        raise ValueError("illegal move")
    return Move(direction=direction, to=to)


def apply_move(state: QuoridorState, move: Move) -> QuoridorState:
    resolved = resolve_move(state, move.direction, move.to)
    if resolved.to is None:
        raise ValueError("illegal move")
    color = state.current_player
    return state.with_pawn(color, resolved.to).switch_turn()


def _wall_passes_filters(state: QuoridorState, wall: WallSlot) -> bool:
    color = state.current_player
    if state.walls_remaining(color) <= 0:
        return False
    if not (0 <= wall.row <= 7 and 0 <= wall.col <= 7):
        return False
    if wall.orientation == "horizontal":
        if state.horizontal_walls[wall.row][wall.col]:
            return False
        if state.vertical_walls[wall.row][wall.col]:
            return False
        if wall.col > 0 and state.horizontal_walls[wall.row][wall.col - 1]:
            return False
        if wall.col < 7 and state.horizontal_walls[wall.row][wall.col + 1]:
            return False
    else:
        if state.vertical_walls[wall.row][wall.col]:
            return False
        if state.horizontal_walls[wall.row][wall.col]:
            return False
        if wall.row > 0 and state.vertical_walls[wall.row - 1][wall.col]:
            return False
        if wall.row < 7 and state.vertical_walls[wall.row + 1][wall.col]:
            return False
    return True


def _has_any_wall(state: QuoridorState) -> bool:
    for row in state.horizontal_walls:
        if any(row):
            return True
    for row in state.vertical_walls:
        if any(row):
            return True
    return False


def _wall_maintains_paths(
    state: QuoridorState,
    wall: WallSlot,
    cache: DistanceCache | None = None,
) -> bool:
    if not _has_any_wall(state):
        return True
    temp = state.with_wall(wall.orientation, wall.row, wall.col)
    return both_reachable(temp, cache=cache)


def get_legal_actions(
    state: QuoridorState,
    *,
    dist_cache: DistanceCache | None = None,
) -> list[Action]:
    cache = dist_cache if dist_cache is not None else SimpleDistanceCache()
    actions: list[Action] = []
    for direction in ("up", "down", "left", "right"):
        for dest in sorted(move_destinations(state, direction), key=lambda p: (p[0], p[1])):
            actions.append(Move(direction=direction, to=dest))

    for row in range(8):
        for col in range(8):
            for orientation in ("horizontal", "vertical"):
                wall = WallSlot(orientation=orientation, row=row, col=col)
                if not _wall_passes_filters(state, wall):
                    continue
                if _wall_maintains_paths(state, wall, cache=cache):
                    actions.append(wall)
    return actions


def apply_action(state: QuoridorState, action: Action) -> QuoridorState:
    if isinstance(action, Move):
        return apply_move(state, action)
    if not _wall_passes_filters(state, action) or not _wall_maintains_paths(state, action):
        raise ValueError("illegal wall")
    return state.with_wall(action.orientation, action.row, action.col).switch_turn()


def is_action_legal(state: QuoridorState, action: Action) -> bool:
    legal = get_legal_actions(state)
    if isinstance(action, Move):
        try:
            resolved = resolve_move(state, action.direction, action.to)
        except ValueError:
            return False
        return resolved in legal
    return any(
        isinstance(a, WallSlot)
        and a.orientation == action.orientation
        and a.row == action.row
        and a.col == action.col
        for a in legal
    )

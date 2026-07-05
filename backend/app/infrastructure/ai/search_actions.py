from __future__ import annotations

from quoridor.board_topology import can_step
from quoridor.domain.actions import Action, Move, WallSlot
from quoridor.domain.state import GOAL_ROW, Color, QuoridorState
from quoridor.pathfinding import DistanceCache, distance_map, distances


def split_legal_actions(legal: list[Action]) -> tuple[list[Move], list[WallSlot]]:
    moves: list[Move] = []
    walls: list[WallSlot] = []
    for action in legal:
        if isinstance(action, Move):
            moves.append(action)
        else:
            walls.append(action)
    return moves, walls


def opponent(color: Color) -> Color:
    return "black" if color == "white" else "white"


def enemy_path_delta(
    state: QuoridorState,
    wall: WallSlot,
    cache: DistanceCache | None = None,
) -> int:
    """How many cells the opponent's BFS distance grows after placing this wall."""
    player = state.current_player
    dw, db = distances(state, cache)
    enemy_before = db if player == "white" else dw
    if enemy_before is None:
        return 0
    temp = state.with_wall(wall.orientation, wall.row, wall.col)
    dw_after, db_after = distances(temp, cache)
    enemy_after = db_after if player == "white" else dw_after
    if enemy_after is None:
        return 0
    return max(0, enemy_after - enemy_before)


def _opponent_distances(state: QuoridorState, player: Color) -> dict[tuple[int, int], int]:
    """BFS distance map for the opponent from their pawn (evaluation semantics)."""
    return distance_map(state, opponent(player), for_evaluation=True)


def shortest_path_edges_blocked(
    state: QuoridorState,
    wall: WallSlot,
    cache: DistanceCache | None = None,
) -> int:
    """Count opponent shortest-path edges this wall would block (even if alternate paths exist)."""
    player = state.current_player
    dist = _opponent_distances(state, player)
    if not dist:
        return 0
    goal_row = GOAL_ROW[opponent(player)]
    goal_dist = min(
        (d for (r, _), d in dist.items() if r == goal_row),
        default=None,
    )
    if goal_dist is None:
        return 0

    temp = state.with_wall(wall.orientation, wall.row, wall.col)
    blocked = 0
    for pos, base in dist.items():
        if base >= goal_dist:
            continue
        r, c = pos
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nxt = (r + dr, c + dc)
            step_dist = dist.get(nxt)
            if step_dist != base + 1:
                continue
            if can_step(state, pos, nxt) and not can_step(temp, pos, nxt):
                blocked += 1
    return blocked


def corridor_wall_pressure(state: QuoridorState, wall: WallSlot) -> int:
    """Heuristic pressure for walls placed between the opponent and their goal."""
    player = state.current_player
    enemy = opponent(player)
    er, ec = state.pawn(enemy)
    goal_row = GOAL_ROW[enemy]

    if wall.orientation == "horizontal":
        row_lo, row_hi = wall.row, wall.row + 1
        col_lo, col_hi = wall.col, wall.col + 1
    else:
        row_lo, row_hi = wall.row, wall.row + 1
        col_lo, col_hi = wall.col, wall.col + 1

    col_score = 2 - min(abs(col_lo - ec), abs(col_hi - ec))

    if enemy == "white":
        if row_hi > er or row_lo < goal_row:
            return 0
        proximity = er - row_hi
    else:
        if row_lo < er or row_hi > goal_row:
            return 0
        proximity = row_lo - er

    if proximity > 3:
        return 0
    return max(0, (4 - proximity) * 2 + col_score)


def wall_strategic_score(
    state: QuoridorState,
    wall: WallSlot,
    cache: DistanceCache | None = None,
) -> int:
    delta = enemy_path_delta(state, wall, cache)
    blocked = shortest_path_edges_blocked(state, wall, cache)
    corridor = corridor_wall_pressure(state, wall)
    return delta * 100 + blocked * 10 + corridor


def prioritize_wall_actions(
    state: QuoridorState,
    walls: list[WallSlot],
    cache: DistanceCache | None,
    limit: int,
) -> list[WallSlot]:
    if limit <= 0 or not walls:
        return []
    ranked = sorted(
        walls,
        key=lambda wall: (
            -wall_strategic_score(state, wall, cache),
            wall.row,
            wall.col,
            wall.orientation,
        ),
    )
    return ranked[:limit]


def search_actions(
    state: QuoridorState,
    legal: list[Action],
    cache: DistanceCache | None,
    max_wall_candidates: int,
) -> list[Action]:
    """All pawn moves plus the top wall candidates by opponent path lengthening."""
    moves, walls = split_legal_actions(legal)
    if max_wall_candidates <= 0:
        return list(moves)
    top_walls = prioritize_wall_actions(state, walls, cache, max_wall_candidates)
    return [*moves, *top_walls]

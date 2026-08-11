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


def _enemy_distance(
    state: QuoridorState,
    cache: DistanceCache | None,
) -> int | None:
    dist_white, dist_black = distances(state, cache)
    return dist_black if state.current_player == "white" else dist_white


def enemy_two_wall_path_delta(
    state: QuoridorState,
    wall1: WallSlot,
    wall2: WallSlot,
    cache: DistanceCache | None = None,
) -> int:
    """Enemy shortest-path growth after placing ``wall1`` then ``wall2``."""
    from quoridor.rules import is_action_legal

    before = _enemy_distance(state, cache)
    if before is None:
        return 0
    temp = state.with_wall(wall1.orientation, wall1.row, wall1.col)
    if not is_action_legal(temp, wall2):
        return 0
    temp2 = temp.with_wall(wall2.orientation, wall2.row, wall2.col)
    after = _enemy_distance(temp2, cache)
    if after is None:
        return 0
    return max(0, after - before)


def wall_enables_path_lengthening(
    state: QuoridorState,
    wall: WallSlot,
    partner_walls: list[WallSlot],
    cache: DistanceCache | None = None,
    *,
    enemy_before: int | None = None,
) -> bool:
    """True if ``wall`` alone, or with some partner, lengthens the enemy path."""
    from quoridor.rules import is_action_legal

    if enemy_path_delta(state, wall, cache) > 0:
        return True
    before = enemy_before if enemy_before is not None else _enemy_distance(state, cache)
    if before is None:
        return False
    # Skip walls that do not touch any current shortest-path edge: they are
    # unlikely first moves of a two-wall block and are expensive to pair-search.
    if shortest_path_edges_blocked(state, wall, cache) <= 0:
        return False
    temp = state.with_wall(wall.orientation, wall.row, wall.col)
    for partner in partner_walls:
        if (
            partner.orientation == wall.orientation
            and partner.row == wall.row
            and partner.col == wall.col
        ):
            continue
        if not is_action_legal(temp, partner):
            continue
        temp2 = temp.with_wall(partner.orientation, partner.row, partner.col)
        after = _enemy_distance(temp2, cache)
        if after is not None and after > before:
            return True
    return False


def select_path_affecting_walls(
    state: QuoridorState,
    walls: list[WallSlot],
    cache: DistanceCache | None,
    limit: int,
    *,
    allow_two_wall_setup: bool = True,
    setup_partner_limit: int = 16,
) -> list[WallSlot]:
    """Walls that lengthen the enemy path alone, or as the first of a 2-wall combo."""
    if limit <= 0 or not walls:
        return []

    singles: list[WallSlot] = []
    zero_delta: list[WallSlot] = []
    for wall in walls:
        if enemy_path_delta(state, wall, cache) > 0:
            singles.append(wall)
        else:
            zero_delta.append(wall)

    setups: list[WallSlot] = []
    if allow_two_wall_setup and zero_delta:
        enemy_before = _enemy_distance(state, cache)
        partner_pool = sorted(
            walls,
            key=lambda wall: (
                -corridor_wall_pressure(state, wall),
                -shortest_path_edges_blocked(state, wall, cache),
                wall.row,
                wall.col,
                wall.orientation,
            ),
        )[:setup_partner_limit]
        # Prefer completing combos with already-known single lengtheners.
        partners = list(dict.fromkeys([*singles, *partner_pool]))
        for wall in zero_delta:
            if wall_enables_path_lengthening(
                state,
                wall,
                partners,
                cache,
                enemy_before=enemy_before,
            ):
                setups.append(wall)

    candidates = [*singles, *setups]
    if not candidates:
        return []
    ranked = sorted(
        candidates,
        key=lambda wall: (
            -wall_strategic_score(state, wall, cache),
            wall.row,
            wall.col,
            wall.orientation,
        ),
    )
    return ranked[:limit]


def prioritize_wall_actions(
    state: QuoridorState,
    walls: list[WallSlot],
    cache: DistanceCache | None,
    limit: int,
    *,
    require_path_lengthening: bool = False,
) -> list[WallSlot]:
    if limit <= 0 or not walls:
        return []
    if require_path_lengthening:
        return select_path_affecting_walls(
            state,
            walls,
            cache,
            limit,
            allow_two_wall_setup=True,
        )
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
    """All pawn moves plus path-affecting wall candidates.

    A wall is kept if it alone increases the opponent's shortest path, or if it
    is the first stone of a two-wall combo that does. Results are capped to
    ``max_wall_candidates``. If none qualify, only pawn moves are returned.
    """
    moves, walls = split_legal_actions(legal)
    if max_wall_candidates <= 0:
        return list(moves)
    top_walls = prioritize_wall_actions(
        state,
        walls,
        cache,
        max_wall_candidates,
        require_path_lengthening=True,
    )
    return [*moves, *top_walls]

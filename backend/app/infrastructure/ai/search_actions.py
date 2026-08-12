from __future__ import annotations

from quoridor.board_topology import can_step
from quoridor.domain.actions import Action, Move, WallSlot
from quoridor.domain.state import GOAL_ROW, Color, QuoridorState
from quoridor.pathfinding import DistanceCache, distance_map, distances

# Caps for two-wall setup search. Full pairwise search is too slow for PPO masks.
DEFAULT_SETUP_CANDIDATE_LIMIT = 4
DEFAULT_SETUP_PARTNER_LIMIT = 4
# Ignore path-touching walls farther than this Chebyshev distance from each other.
DEFAULT_SETUP_PARTNER_RADIUS = 2


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
    *,
    opponent_dist: dict[tuple[int, int], int] | None = None,
) -> int:
    """Count opponent shortest-path edges this wall would block (even if alternate paths exist)."""
    player = state.current_player
    dist = opponent_dist if opponent_dist is not None else _opponent_distances(state, player)
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
    *,
    delta: int | None = None,
    blocked: int | None = None,
) -> int:
    if delta is None:
        delta = enemy_path_delta(state, wall, cache)
    if blocked is None:
        blocked = shortest_path_edges_blocked(state, wall, cache)
    corridor = corridor_wall_pressure(state, wall)
    return delta * 100 + blocked * 10 + corridor


def _enemy_distance(
    state: QuoridorState,
    cache: DistanceCache | None,
) -> int | None:
    dist_white, dist_black = distances(state, cache)
    return dist_black if state.current_player == "white" else dist_white


def _wall_key(wall: WallSlot) -> tuple[str, int, int]:
    return (wall.orientation, wall.row, wall.col)


def _walls_near(a: WallSlot, b: WallSlot, radius: int) -> bool:
    return max(abs(a.row - b.row), abs(a.col - b.col)) <= radius


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
    blocked: int | None = None,
    opponent_dist: dict[tuple[int, int], int] | None = None,
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
    if blocked is None:
        blocked = shortest_path_edges_blocked(
            state, wall, cache, opponent_dist=opponent_dist
        )
    if blocked <= 0:
        return False
    temp = state.with_wall(wall.orientation, wall.row, wall.col)
    wall_id = _wall_key(wall)
    for partner in partner_walls:
        if _wall_key(partner) == wall_id:
            continue
        if not is_action_legal(temp, partner):
            continue
        temp2 = temp.with_wall(partner.orientation, partner.row, partner.col)
        after = _enemy_distance(temp2, cache)
        if after is not None and after > before:
            return True
    return False


def _rank_key_for_setup(state: QuoridorState, wall: WallSlot, blocked: int) -> tuple:
    return (
        -corridor_wall_pressure(state, wall),
        -blocked,
        wall.row,
        wall.col,
        wall.orientation,
    )


def select_path_affecting_walls(
    state: QuoridorState,
    walls: list[WallSlot],
    cache: DistanceCache | None,
    limit: int,
    *,
    allow_two_wall_setup: bool = True,
    setup_candidate_limit: int = DEFAULT_SETUP_CANDIDATE_LIMIT,
    setup_partner_limit: int = DEFAULT_SETUP_PARTNER_LIMIT,
    setup_partner_radius: int = DEFAULT_SETUP_PARTNER_RADIUS,
    verify_two_wall_pairs: bool = False,
    max_pair_checks: int = 4,
) -> list[WallSlot]:
    """Walls that lengthen the enemy path alone, or likely 2-wall setup stones.

    For PPO masks, two-wall setups are chosen heuristically: zero-delta walls that
    (1) sit in the enemy corridor and (2) touch a current shortest-path edge.
    Optional pair verification is off by default because full pair BFS is too
    expensive for per-step action masks.
    """
    if limit <= 0 or not walls:
        return []

    deltas: dict[tuple[str, int, int], int] = {}
    singles: list[WallSlot] = []
    zero_delta: list[WallSlot] = []
    for wall in walls:
        delta = enemy_path_delta(state, wall, cache)
        deltas[_wall_key(wall)] = delta
        if delta > 0:
            singles.append(wall)
        else:
            zero_delta.append(wall)

    setups: list[WallSlot] = []
    blocked_by_key: dict[tuple[str, int, int], int] = {}
    remaining = limit - len(singles)
    if allow_two_wall_setup and zero_delta and remaining > 0:
        opponent_dist = _opponent_distances(state, state.current_player)
        corridor_zeros = [
            wall for wall in zero_delta if corridor_wall_pressure(state, wall) > 0
        ]
        if not corridor_zeros:
            corridor_zeros = zero_delta

        scored_zeros: list[tuple[WallSlot, int]] = []
        for wall in corridor_zeros:
            blocked = shortest_path_edges_blocked(
                state, wall, cache, opponent_dist=opponent_dist
            )
            blocked_by_key[_wall_key(wall)] = blocked
            if blocked > 0:
                scored_zeros.append((wall, blocked))
        scored_zeros.sort(key=lambda item: _rank_key_for_setup(state, item[0], item[1]))
        setup_pool = scored_zeros[: max(0, min(setup_candidate_limit, remaining))]

        if verify_two_wall_pairs:
            enemy_before = _enemy_distance(state, cache)
            partner_scored: list[tuple[WallSlot, int]] = list(scored_zeros)
            for wall in singles:
                blocked = blocked_by_key.get(_wall_key(wall))
                if blocked is None:
                    blocked = shortest_path_edges_blocked(
                        state, wall, cache, opponent_dist=opponent_dist
                    )
                    blocked_by_key[_wall_key(wall)] = blocked
                partner_scored.append((wall, blocked))
            partner_scored.sort(
                key=lambda item: _rank_key_for_setup(state, item[0], item[1])
            )
            checks = 0
            for wall, blocked in setup_pool:
                if checks >= max_pair_checks:
                    break
                local_partners = [
                    partner
                    for partner, _ in partner_scored
                    if _wall_key(partner) != _wall_key(wall)
                    and _walls_near(wall, partner, setup_partner_radius)
                ][: max(0, setup_partner_limit)]
                if not local_partners:
                    local_partners = [
                        partner
                        for partner, _ in partner_scored
                        if _wall_key(partner) != _wall_key(wall)
                    ][: max(0, setup_partner_limit)]
                # Bound verification cost.
                budgeted = []
                for partner in local_partners:
                    if checks >= max_pair_checks:
                        break
                    checks += 1
                    budgeted.append(partner)
                if wall_enables_path_lengthening(
                    state,
                    wall,
                    budgeted,
                    cache,
                    enemy_before=enemy_before,
                    blocked=blocked,
                    opponent_dist=opponent_dist,
                ):
                    setups.append(wall)
        else:
            setups = [wall for wall, _ in setup_pool]

    candidates = [*singles, *setups]
    if not candidates:
        return []

    def _score(wall: WallSlot) -> int:
        key = _wall_key(wall)
        return wall_strategic_score(
            state,
            wall,
            cache,
            delta=deltas.get(key),
            blocked=blocked_by_key.get(key),
        )

    ranked = sorted(
        candidates,
        key=lambda wall: (
            -_score(wall),
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
    is the first stone of a capped two-wall combo that does. Results are capped to
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

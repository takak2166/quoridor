from dataclasses import replace

from quoridor.domain.actions import Move, WallSlot
from quoridor.pathfinding import SimpleDistanceCache, distances
from quoridor.rules import get_legal_actions
from tests.unit.fixtures.plan_fixtures import build_state, seed3_wall_separated_choke


def test_prioritize_walls_puts_path_lengthening_walls_first() -> None:
    from app.infrastructure.ai.search_actions import enemy_path_delta, prioritize_wall_actions, split_legal_actions

    state = build_state(
        white=(5, 4),
        black=(3, 4),
        h=frozenset({(4, 2), (4, 3), (4, 5), (4, 6)}),
        current="black",
    )
    _, walls = split_legal_actions(get_legal_actions(state))
    ranked = prioritize_wall_actions(state, walls, cache=None, limit=5)
    assert ranked
    assert enemy_path_delta(state, ranked[0], cache=None) >= enemy_path_delta(state, ranked[-1], cache=None)


def test_search_actions_includes_all_moves_and_caps_walls() -> None:
    from app.infrastructure.ai.search_actions import (
        search_actions,
        select_path_affecting_walls,
        split_legal_actions,
    )

    state = build_state(current="black")
    legal = get_legal_actions(state)
    moves, walls = split_legal_actions(legal)
    effective = select_path_affecting_walls(state, walls, cache=None, limit=64)
    filtered = search_actions(state, legal, cache=None, max_wall_candidates=8)
    assert len([a for a in filtered if isinstance(a, Move)]) == len(moves)
    assert len([a for a in filtered if isinstance(a, WallSlot)]) == min(8, len(effective))
    filtered_walls = [a for a in filtered if isinstance(a, WallSlot)]
    assert all(a in effective for a in filtered_walls)


def test_two_wall_setup_included_when_single_delta_is_zero() -> None:
    """Path-touching corridor walls with alone-Δ=0 are kept as 2-wall setups."""
    from app.infrastructure.ai.search_actions import (
        enemy_path_delta,
        enemy_two_wall_path_delta,
        search_actions,
        select_path_affecting_walls,
        shortest_path_edges_blocked,
        split_legal_actions,
    )

    state = build_state(white=(6, 4), black=(3, 4), current="white")
    setup = WallSlot(orientation="vertical", row=3, col=3)
    partner = WallSlot(orientation="horizontal", row=3, col=4)
    assert enemy_path_delta(state, setup, cache=None) == 0
    assert shortest_path_edges_blocked(state, setup, cache=None) > 0
    assert enemy_two_wall_path_delta(state, setup, partner, cache=None) > 0

    legal = get_legal_actions(state)
    _, walls = split_legal_actions(legal)
    assert setup in walls and partner in walls

    with_setup = select_path_affecting_walls(
        state, walls, cache=None, limit=20, allow_two_wall_setup=True
    )
    without_setup = select_path_affecting_walls(
        state, walls, cache=None, limit=20, allow_two_wall_setup=False
    )
    assert setup in with_setup
    assert setup not in without_setup

    filtered = search_actions(state, legal, cache=None, max_wall_candidates=20)
    assert setup in [a for a in filtered if isinstance(a, WallSlot)]


def test_search_actions_keeps_single_or_setup_walls_only() -> None:
    from app.infrastructure.ai.search_actions import (
        search_actions,
        select_path_affecting_walls,
        split_legal_actions,
    )

    state = build_state(current="black")
    legal = get_legal_actions(state)
    _, walls = split_legal_actions(legal)
    allowed = set(select_path_affecting_walls(state, walls, cache=None, limit=64))
    filtered = search_actions(state, legal, cache=None, max_wall_candidates=64)
    filtered_walls = [a for a in filtered if isinstance(a, WallSlot)]
    assert filtered_walls
    assert all(wall in allowed for wall in filtered_walls)


def test_two_phase_root_prefers_blocking_wall_in_corridor() -> None:
    from app.infrastructure.ai.minimax import MinimaxConfig, NormalMinimaxPolicy

    state = build_state(
        white=(5, 4),
        black=(3, 4),
        h=frozenset({(4, 2), (4, 3), (4, 5), (4, 6)}),
        current="black",
    )
    policy = NormalMinimaxPolicy(
        config=MinimaxConfig(
            primary_depth=1,
            fallback_depth=1,
            max_nodes=5000,
            time_budget_ms=5000,
            max_wall_candidates=12,
            two_phase_search=True,
        )
    )
    action = policy.select_move(state, "black")
    assert isinstance(action, WallSlot)


def test_shortest_path_edges_blocked_matches_full_board_scan() -> None:
    from app.infrastructure.ai.search_actions import (
        _opponent_distances,
        opponent,
        shortest_path_edges_blocked,
        split_legal_actions,
    )
    from quoridor.board_topology import can_step
    from quoridor.domain.state import GOAL_ROW

    def brute_force_blocked(state, wall, opponent_dist):
        player = state.current_player
        dist = opponent_dist
        goal_row = GOAL_ROW[opponent(player)]
        goal_dist = min((d for (r, _), d in dist.items() if r == goal_row), default=None)
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
                if dist.get(nxt) != base + 1:
                    continue
                if can_step(state, pos, nxt) and not can_step(temp, pos, nxt):
                    blocked += 1
        return blocked

    state = build_state(white=(5, 4), black=(3, 4), current="black")
    _, walls = split_legal_actions(get_legal_actions(state))
    opponent_dist = _opponent_distances(state, state.current_player)
    for wall in walls[:40]:
        fast = shortest_path_edges_blocked(state, wall, cache=None, opponent_dist=opponent_dist)
        slow = brute_force_blocked(state, wall, opponent_dist)
        assert fast == slow


def test_opening_corridor_wall_ranks_near_opponent_goal() -> None:
    from app.infrastructure.ai.search_actions import prioritize_wall_actions, split_legal_actions

    state = build_state(current="black")
    _, walls = split_legal_actions(get_legal_actions(state))
    ranked = prioritize_wall_actions(state, walls, cache=None, limit=5)
    assert ranked[0] == WallSlot(orientation="horizontal", row=7, col=3)


def test_two_wall_delta_ignores_evaluation_distance_cache_for_legality() -> None:
    """Prepopulated evaluation distances must not skip strict wall-legality checks."""
    from app.infrastructure.ai.search_actions import enemy_two_wall_path_delta

    state = replace(seed3_wall_separated_choke(), current_player="black")
    state = replace(
        state,
        horizontal_walls=tuple(
            tuple(False if (row, col) in {(0, 2), (0, 4)} else cell for col, cell in enumerate(row_cells))
            for row, row_cells in enumerate(state.horizontal_walls)
        ),
    )
    first = WallSlot(orientation="horizontal", row=0, col=4)
    second = WallSlot(orientation="horizontal", row=0, col=2)
    temp = state.with_wall(first.orientation, first.row, first.col)
    final_state = temp.with_wall(second.orientation, second.row, second.col)
    cache = SimpleDistanceCache()
    distances(final_state, cache)
    assert enemy_two_wall_path_delta(state, first, second, cache=None) == 0
    assert enemy_two_wall_path_delta(state, first, second, cache=cache) == 0

from quoridor.domain.actions import Move, WallSlot
from quoridor.rules import get_legal_actions
from tests.unit.fixtures.plan_fixtures import build_state


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


def test_opening_corridor_wall_ranks_near_opponent_goal() -> None:
    from app.infrastructure.ai.search_actions import prioritize_wall_actions, split_legal_actions

    state = build_state(current="black")
    _, walls = split_legal_actions(get_legal_actions(state))
    ranked = prioritize_wall_actions(state, walls, cache=None, limit=5)
    assert ranked[0] == WallSlot(orientation="horizontal", row=7, col=3)

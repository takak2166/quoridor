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
        enemy_path_delta,
        search_actions,
        split_legal_actions,
    )

    state = build_state(current="black")
    legal = get_legal_actions(state)
    moves, walls = split_legal_actions(legal)
    effective = [w for w in walls if enemy_path_delta(state, w, cache=None) > 0]
    filtered = search_actions(state, legal, cache=None, max_wall_candidates=8)
    assert len([a for a in filtered if isinstance(a, Move)]) == len(moves)
    assert len([a for a in filtered if isinstance(a, WallSlot)]) == min(8, len(effective))
    assert all(
        enemy_path_delta(state, a, cache=None) > 0
        for a in filtered
        if isinstance(a, WallSlot)
    )


def test_search_actions_excludes_zero_delta_walls() -> None:
    from app.infrastructure.ai.search_actions import enemy_path_delta, search_actions, split_legal_actions

    state = build_state(current="black")
    legal = get_legal_actions(state)
    _, walls = split_legal_actions(legal)
    zero_delta = [w for w in walls if enemy_path_delta(state, w, cache=None) == 0]
    assert zero_delta, "fixture should include at least one non-lengthening wall"
    filtered = search_actions(state, legal, cache=None, max_wall_candidates=64)
    filtered_walls = [a for a in filtered if isinstance(a, WallSlot)]
    assert filtered_walls
    assert not any(w in filtered_walls for w in zero_delta)


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

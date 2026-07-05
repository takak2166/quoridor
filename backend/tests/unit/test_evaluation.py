from quoridor.domain.actions import Move, WallSlot
from quoridor.domain.state import initial_state
from quoridor.pathfinding import distances
from quoridor.rules import apply_action, get_legal_actions
from tests.unit.fixtures.plan_fixtures import build_state


def test_wall_that_lengthens_enemy_path_scores_higher_than_forward_move() -> None:
    from app.infrastructure.ai.evaluation import NORMAL_EVAL_CONFIG, StateEvaluator

    state = build_state(
        white=(5, 4),
        black=(3, 4),
        h=frozenset({(4, 2), (4, 3), (4, 5), (4, 6)}),
        current="black",
    )
    evaluator = StateEvaluator(NORMAL_EVAL_CONFIG)

    move = Move(direction="up", to=(4, 4))
    move_score = evaluator.evaluate(apply_action(state, move), "black", before=state)

    best_wall_score = max(
        evaluator.evaluate(apply_action(state, action), "black", before=state)
        for action in get_legal_actions(state)
        if isinstance(action, WallSlot)
    )
    assert best_wall_score > move_score


def test_enemy_path_block_bonus_is_zero_without_walls() -> None:
    from app.infrastructure.ai.evaluation import NORMAL_EVAL_CONFIG, StateEvaluator

    state = initial_state()
    evaluator = StateEvaluator(NORMAL_EVAL_CONFIG)
    dw, db = distances(state)
    assert evaluator.evaluate(state, "black") == (db - dw) / 16.0


def test_enemy_path_block_bonus_reflects_distance_increase() -> None:
    from app.infrastructure.ai.evaluation import NORMAL_EVAL_CONFIG, StateEvaluator

    base = build_state(
        white=(5, 4),
        black=(3, 4),
        h=frozenset({(4, 2), (4, 3), (4, 5), (4, 6)}),
        current="black",
    )
    wall = WallSlot(orientation="vertical", row=5, col=2)
    blocked = apply_action(base, wall)
    evaluator = StateEvaluator(NORMAL_EVAL_CONFIG)

    enemy_before, mine_before = distances(base)
    enemy_after, mine_after = distances(blocked)
    assert enemy_after is not None and enemy_before is not None
    assert mine_after is not None and mine_before is not None
    assert enemy_after > enemy_before

    cfg = NORMAL_EVAL_CONFIG
    raw = float(enemy_after - mine_after)
    raw += cfg.wall_bonus_depth_scales[0] * cfg.enemy_path_block_weight * (enemy_after - enemy_before)
    raw -= cfg.self_path_block_weight * max(0, mine_after - mine_before)
    raw += cfg.wall_remaining_weight * (base.black_walls_remaining - 1 - base.white_walls_remaining)
    assert evaluator.evaluate(blocked, "black", before=base, remaining_depth=0) == max(
        -1.0,
        min(1.0, raw / 16.0),
    )


def test_wall_bonus_scales_down_with_remaining_depth() -> None:
    from app.infrastructure.ai.evaluation import EvalConfig, StateEvaluator

    base = build_state(
        white=(5, 4),
        black=(3, 4),
        h=frozenset({(4, 2), (4, 3), (4, 5), (4, 6)}),
        current="black",
    )
    blocked = apply_action(base, WallSlot(orientation="vertical", row=5, col=2))
    evaluator = StateEvaluator(
        EvalConfig(
            enemy_path_block_weight=2.5,
            wall_bonus_depth_scales=(1.0, 0.75, 0.5, 0.25),
        )
    )

    shallow = evaluator.evaluate(blocked, "black", before=base, remaining_depth=0)
    deep = evaluator.evaluate(blocked, "black", before=base, remaining_depth=3)
    assert shallow > deep


def test_easy_wall_bonus_is_weaker_than_normal_at_same_depth() -> None:
    from app.infrastructure.ai.evaluation import EASY_EVAL_CONFIG, NORMAL_EVAL_CONFIG, StateEvaluator

    base = build_state(
        white=(5, 4),
        black=(3, 4),
        h=frozenset({(4, 2), (4, 3), (4, 5), (4, 6)}),
        current="black",
    )
    blocked = apply_action(base, WallSlot(orientation="vertical", row=5, col=2))
    easy = StateEvaluator(EASY_EVAL_CONFIG).evaluate(blocked, "black", before=base, remaining_depth=0)
    normal = StateEvaluator(NORMAL_EVAL_CONFIG).evaluate(blocked, "black", before=base, remaining_depth=0)
    assert normal > easy


def test_minimax_prefers_wall_when_it_lengthens_opponent_path() -> None:
    from app.infrastructure.ai.minimax import MinimaxConfig, NormalMinimaxPolicy

    state = build_state(
        white=(5, 4),
        black=(3, 4),
        h=frozenset({(4, 2), (4, 3), (4, 5), (4, 6)}),
        current="black",
    )
    policy = NormalMinimaxPolicy(
        config=MinimaxConfig(primary_depth=1, fallback_depth=1, max_nodes=5000, time_budget_ms=5000)
    )
    action = policy.select_move(state, "black")
    assert isinstance(action, WallSlot)

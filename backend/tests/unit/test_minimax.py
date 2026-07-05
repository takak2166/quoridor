from quoridor.domain.state import QuoridorState, initial_state


def test_easy_maximizes_immediate_evaluation() -> None:
    from app.infrastructure.ai.evaluation import StateEvaluator
    from app.infrastructure.ai.minimax import EasyMinimaxPolicy, MinimaxConfig
    from quoridor.rules import apply_action, get_legal_actions
    from tests.unit.fixtures.plan_fixtures import build_state

    # Black is clearly ahead in the race; corridor override should not fire.
    state = build_state(white=(8, 4), black=(3, 4), current="black")
    policy = EasyMinimaxPolicy(config=MinimaxConfig())
    evaluator = StateEvaluator()

    moves, _ = __import__(
        "app.infrastructure.ai.search_actions", fromlist=["split_legal_actions"]
    ).split_legal_actions(get_legal_actions(state))
    best_eval = max(
        evaluator.evaluate(apply_action(state, action), "black") for action in moves
    )
    chosen = policy.select_move(state, "black")
    chosen_eval = evaluator.evaluate(apply_action(state, chosen), "black")
    assert chosen_eval == best_eval


def test_very_easy_maximizes_immediate_evaluation() -> None:
    from app.infrastructure.ai.evaluation import StateEvaluator
    from app.infrastructure.ai.minimax import MinimaxConfig, VeryEasyMinimaxPolicy
    from app.infrastructure.ai.search_actions import split_legal_actions
    from quoridor.rules import apply_action, get_legal_actions
    from tests.unit.fixtures.plan_fixtures import build_state

    state = build_state(white=(8, 4), black=(3, 4), current="black")
    policy = VeryEasyMinimaxPolicy(config=MinimaxConfig())
    evaluator = StateEvaluator()
    moves, _ = split_legal_actions(get_legal_actions(state))

    best_eval = max(
        evaluator.evaluate(apply_action(state, action), "black") for action in moves
    )
    chosen = policy.select_move(state, "black")
    chosen_eval = evaluator.evaluate(apply_action(state, chosen), "black")
    assert chosen_eval == best_eval


def test_behind_uses_full_depth_for_wall_search() -> None:
    """Regression: do not short-circuit to depth=1 when behind; search walls at root depth."""
    from app.infrastructure.ai.evaluation import NORMAL_EVAL_CONFIG
    from app.infrastructure.ai.minimax import MinimaxConfig, MinimaxEngine
    from quoridor.domain.actions import WallSlot
    from tests.unit.fixtures.plan_fixtures import build_state

    state = build_state(white=(6, 4), black=(0, 4), current="black")
    engine = MinimaxEngine(
        MinimaxConfig(
            primary_depth=3,
            fallback_depth=2,
            max_nodes=5000,
            time_budget_ms=5000,
            two_phase_search=True,
        ),
        NORMAL_EVAL_CONFIG,
    )
    wall_depths: list[int] = []
    original = engine._best_root_action

    def spy(
        self: MinimaxEngine,
        state: object,
        color: object,
        depth: int,
        actions: list[object],
    ) -> tuple[object, float, bool]:
        if actions and isinstance(actions[0], WallSlot):
            wall_depths.append(depth)
        return original(state, color, depth, actions)  # type: ignore[arg-type]

    engine._best_root_action = spy.__get__(engine, MinimaxEngine)  # type: ignore[method-assign]
    action = engine.select_move(state, "black")

    assert isinstance(action, WallSlot)
    assert wall_depths
    assert all(d == 3 for d in wall_depths)


def test_prefers_wall_when_behind_in_race() -> None:
    from app.infrastructure.ai.minimax import EasyMinimaxPolicy, MinimaxConfig
    from quoridor.domain.actions import Move, WallSlot
    from tests.unit.fixtures.plan_fixtures import build_state

    state = build_state(white=(6, 4), black=(0, 4), current="black")
    policy = EasyMinimaxPolicy(config=MinimaxConfig())
    action = policy.select_move(state, "black")
    assert isinstance(action, WallSlot)
    assert not isinstance(action, Move)


def test_tie_break_can_pick_wall_at_random(monkeypatch) -> None:
    from app.infrastructure.ai.evaluation import EASY_EVAL_CONFIG
    from app.infrastructure.ai.minimax import MinimaxConfig, MinimaxEngine
    from quoridor.domain.actions import Move, WallSlot

    engine = MinimaxEngine(MinimaxConfig(), EASY_EVAL_CONFIG)
    move = Move(direction="up", to=(1, 4))
    wall = WallSlot(orientation="horizontal", row=7, col=3)
    monkeypatch.setattr(
        "app.infrastructure.ai.minimax.random.choice",
        lambda options: wall,
    )
    chosen = engine._choose_move_or_wall(
        initial_state(),
        "black",
        move,
        0.5,
        wall,
        0.5,
    )
    assert chosen == wall


def test_minimax_prefers_shorter_path() -> None:
    from app.infrastructure.ai.minimax import MinimaxConfig, NormalMinimaxPolicy

    state = QuoridorState(
        white=(6, 4),
        black=(2, 4),
        white_walls_remaining=10,
        black_walls_remaining=10,
        horizontal_walls=state_walls_empty(),
        vertical_walls=state_walls_empty(),
        current_player="white",
    )
    policy = NormalMinimaxPolicy(config=MinimaxConfig(primary_depth=2, fallback_depth=1))
    action = policy.select_move(state, "white")
    assert action is not None


def state_walls_empty() -> tuple[tuple[bool, ...], ...]:
    from quoridor.domain.state import empty_walls

    return empty_walls()


def test_depth_two_does_not_invert_root_scores() -> None:
    """Regression: root must not negate _minimax when using explicit min/max inside."""
    from app.infrastructure.ai.evaluation import NORMAL_EVAL_CONFIG
    from app.infrastructure.ai.minimax import MinimaxConfig, MinimaxEngine
    from quoridor.domain.actions import Move
    from quoridor.rules import apply_action
    from tests.unit.fixtures.plan_fixtures import build_state

    state = build_state(white=(6, 4), black=(2, 4), current="black")
    engine = MinimaxEngine(
        MinimaxConfig(
            primary_depth=2,
            fallback_depth=1,
            max_nodes=5000,
            time_budget_ms=5000,
            two_phase_search=False,
        ),
        NORMAL_EVAL_CONFIG,
    )
    import math

    engine._deadline = __import__("time").perf_counter() + 10
    action, _ = engine._search_depth(state, "black", 2)

    up = Move(direction="up", to=(3, 4))
    down = Move(direction="down", to=(1, 4))
    engine._visited = 0
    up_score = engine._minimax(
        apply_action(state, up), 1, "black", -math.inf, math.inf, before=state
    )
    engine._visited = 0
    down_score = engine._minimax(
        apply_action(state, down), 1, "black", -math.inf, math.inf, before=state
    )
    assert up_score > down_score
    assert action == up


def test_very_easy_never_plays_wall() -> None:
    from app.infrastructure.ai.minimax import MinimaxConfig, VeryEasyMinimaxPolicy
    from quoridor.domain.actions import Move, WallSlot
    from tests.unit.fixtures.plan_fixtures import build_state

    policy = VeryEasyMinimaxPolicy(config=MinimaxConfig())
    states = [
        initial_state(),
        build_state(white=(6, 4), black=(2, 4), current="black"),
        build_state(white=(5, 4), black=(3, 4), current="white"),
    ]
    for state in states:
        action = policy.select_move(state, state.current_player)
        assert isinstance(action, Move)
        assert not isinstance(action, WallSlot)


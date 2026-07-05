import math

from quoridor.domain.actions import Move
from quoridor.rules import apply_action
from tests.unit.fixtures.plan_fixtures import build_state


def test_elapsed_move_penalty_reduces_score() -> None:
    from app.infrastructure.ai.evaluation import EvalConfig, StateEvaluator

    evaluator = StateEvaluator(EvalConfig(elapsed_move_penalty=0.8))
    state = build_state(white=(8, 4), black=(3, 4), current="black")
    base = evaluator.evaluate(state, "black")
    penalized = evaluator.evaluate(state, "black", plies_from_root=4)
    assert penalized < base
    assert penalized == max(-1.0, min(1.0, (base * 16.0 - 0.8 * 4) / 16.0))


def test_revisit_penalty_reduces_score() -> None:
    from app.infrastructure.ai.evaluation import EvalConfig, StateEvaluator

    evaluator = StateEvaluator(EvalConfig(revisit_penalty=3.0))
    state = build_state(white=(8, 4), black=(3, 4), current="black")
    base = evaluator.evaluate(state, "black")
    penalized = evaluator.evaluate(state, "black", position_revisits=1)
    assert penalized < base
    assert penalized == max(-1.0, min(1.0, (base * 16.0 - 3.0) / 16.0))


def test_easy_config_has_no_loop_penalties() -> None:
    from app.infrastructure.ai.evaluation import EASY_EVAL_CONFIG, StateEvaluator

    evaluator = StateEvaluator(EASY_EVAL_CONFIG)
    state = build_state(white=(8, 4), black=(3, 4), current="black")
    assert evaluator.evaluate(state, "black", plies_from_root=5, position_revisits=2) == evaluator.evaluate(
        state, "black"
    )


def test_normal_config_applies_loop_penalties() -> None:
    from app.infrastructure.ai.evaluation import NORMAL_EVAL_CONFIG, StateEvaluator

    evaluator = StateEvaluator(NORMAL_EVAL_CONFIG)
    state = build_state(white=(8, 4), black=(3, 4), current="black")
    base = evaluator.evaluate(state, "black")
    penalized = evaluator.evaluate(state, "black", plies_from_root=3, position_revisits=1)
    assert penalized < base


def test_minimax_deprioritizes_two_ply_repetition() -> None:
    """A 2-ply pawn shuffle should score worse than advancing when anti-loop is enabled."""
    from app.infrastructure.ai.evaluation import NORMAL_EVAL_CONFIG
    from app.infrastructure.ai.minimax import MinimaxConfig, MinimaxEngine

    state = build_state(white=(6, 4), black=(2, 4), current="black")
    engine = MinimaxEngine(
        MinimaxConfig(
            primary_depth=2,
            fallback_depth=1,
            max_nodes=5000,
            time_budget_ms=5000,
            two_phase_search=False,
            moves_only=True,
        ),
        NORMAL_EVAL_CONFIG,
    )
    engine._deadline = __import__("time").perf_counter() + 10
    up = Move(direction="up", to=(3, 4))
    down = Move(direction="down", to=(1, 4))
    root_path = {__import__("quoridor.domain.state", fromlist=["position_key"]).position_key(state): 1}

    engine._visited = 0
    up_score = engine._minimax(
        apply_action(state, up),
        1,
        "black",
        -math.inf,
        math.inf,
        before=state,
        path_counts=root_path,
        plies_from_root=1,
    )
    engine._visited = 0
    down_score = engine._minimax(
        apply_action(state, down),
        1,
        "black",
        -math.inf,
        math.inf,
        before=state,
        path_counts=root_path,
        plies_from_root=1,
    )
    assert up_score > down_score

    action, _ = engine._search_depth(state, "black", 2)
    assert action == up

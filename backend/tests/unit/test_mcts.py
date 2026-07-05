from quoridor.domain.actions import Move, action_child_key
from quoridor.rules import get_legal_actions
from tests.unit.fixtures.plan_fixtures import JUMP_CASES


def test_mcts_distinct_moves_do_not_collide_child_key() -> None:
    case = next(c for c in JUMP_CASES if c["id"] == "J.2-DIAG-BOTH")
    legal = get_legal_actions(case["state"])
    keys = [action_child_key(a) for a in legal if isinstance(a, Move) and a.direction == "up"]
    assert len(keys) == 2
    assert len(set(keys)) == 2

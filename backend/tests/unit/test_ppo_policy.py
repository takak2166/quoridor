import random

import numpy as np

from app.infrastructure.ai.ppo_policy import PPOPolicy
from app.infrastructure.rl.move_resolution import resolve_ambiguous_move
from quoridor.domain.actions import NUM_ACTIONS, encode
from quoridor.rules import get_legal_actions
from tests.unit.fixtures.plan_fixtures import JUMP_CASES


def test_ppo_tie_break_deterministic_with_seed() -> None:
    case = next(c for c in JUMP_CASES if c["id"] == "J.2-DIAG-BOTH")
    first = resolve_ambiguous_move(case["state"], "up", random.Random(42))
    second = resolve_ambiguous_move(case["state"], "up", random.Random(42))
    assert first == second
    assert first.to in ((4, 3), (4, 5))


def test_ppo_tie_break_not_sorted_col_zero() -> None:
    case = next(c for c in JUMP_CASES if c["id"] == "J.2-DIAG-BOTH")
    picks: set[tuple[int, int] | None] = set()
    for seed in range(200):
        move = resolve_ambiguous_move(case["state"], "up", random.Random(seed))
        picks.add(move.to)
    assert (4, 3) in picks
    assert (4, 5) in picks


def test_select_with_prior_returns_legal_move_with_to() -> None:
    case = next(c for c in JUMP_CASES if c["id"] == "J.2-DIAG-BOTH")
    legal = get_legal_actions(case["state"])
    policy = PPOPolicy(model_path="/nonexistent/model.zip")
    prior = np.zeros(NUM_ACTIONS, dtype=np.float64)
    for action in legal:
        prior[encode(action)] = 1.0
    prior /= prior.sum()
    chosen = policy._select_with_prior(prior, legal)
    assert chosen in legal
    assert chosen.to is not None

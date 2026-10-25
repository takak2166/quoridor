from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from quoridor.domain.actions import Action, action_child_key, encode
from quoridor.domain.state import Color, QuoridorState
from quoridor.rules import apply_action

PriorFn = Callable[[QuoridorState, Color], NDArray[np.floating]]


@dataclass
class _MCTSNode:
    state: QuoridorState
    parent: _MCTSNode | None
    action: Action | None
    player: Color
    visits: int = 0
    value_sum: float = 0.0
    children: dict[tuple[int, tuple[int, int] | None], _MCTSNode] = field(default_factory=dict)
    untried: list[Action] = field(default_factory=list)
    prior_vector: NDArray[np.floating] | None = field(default=None, compare=False, repr=False)


def mcts_search(
    state: QuoridorState,
    color: Color,
    prior_fn: PriorFn,
    value_fn,
    legal_actions: list[Action],
    budget_ms: int = 450,
    c_puct: float = 1.4,
) -> Action:
    """Run budgeted MCTS with PPO-style priors.

    ``prior_fn`` must be deterministic for a given ``(state, player)`` pair for the
    lifetime of the search tree; cached priors are not recomputed on revisits.
    """
    from app.middleware.metrics import metrics_store
    from quoridor.pathfinding import SimpleDistanceCache

    dist_cache = SimpleDistanceCache()
    root = _MCTSNode(state=state, parent=None, action=None, player=color, untried=list(legal_actions))
    # Warm the root prior before the budget loop (side effect only; used on first expansion/select).
    _ensure_prior_vector(root, prior_fn)
    deadline = time.perf_counter() + budget_ms / 1000
    sims = 0

    while time.perf_counter() < deadline:
        node = root
        path = [node]
        current_state = state.copy()

        while not node.untried and node.children:
            node = _select_child(node, color, c_puct, prior_fn)
            current_state = apply_action(current_state, node.action)  # type: ignore[arg-type]
            path.append(node)

        if node.untried:
            from_pos = node.state.pawn(node.player)
            expansion_prior = _normalized_prior(
                _ensure_prior_vector(node, prior_fn),
                node.untried,
                from_pos,
            )
            action = _pick_expansion(node, expansion_prior, from_pos)
            current_state = apply_action(current_state, action)
            child = _MCTSNode(
                state=current_state,
                parent=node,
                action=action,
                player="black" if node.player == "white" else "white",
                untried=_get_legal(current_state, dist_cache),
            )
            node.children[action_child_key(action)] = child
            node = child
            path.append(node)

        rollout_value = value_fn(current_state, color)
        for n in path:
            n.visits += 1
            n.value_sum += rollout_value
        sims += 1

    metrics_store.record_mcts_sims(sims)
    if not root.children:
        return legal_actions[0]
    best = max(root.children.values(), key=lambda c: c.visits)
    return best.action  # type: ignore[return-value]


def _ensure_prior_vector(node: _MCTSNode, prior_fn: PriorFn) -> NDArray[np.floating]:
    if node.prior_vector is None:
        node.prior_vector = prior_fn(node.state, node.player)
    return node.prior_vector


def _normalized_prior(
    prior: NDArray[np.floating],
    legal_actions: list[Action],
    from_pos: tuple[int, int],
) -> dict[int, float]:
    weights: dict[int, float] = {}
    for action in legal_actions:
        idx = encode(action, from_pos=from_pos)
        weights[idx] = float(prior[idx])
    total = sum(weights.values())
    if total <= 0:
        uniform = 1.0 / len(legal_actions)
        return {encode(action, from_pos=from_pos): uniform for action in legal_actions}
    return {idx: weight / total for idx, weight in weights.items()}


def _pick_expansion(
    node: _MCTSNode,
    prior: dict[int, float],
    from_pos: tuple[int, int],
) -> Action:
    if len(node.untried) == 1:
        return node.untried.pop()

    weights = [prior.get(encode(action, from_pos=from_pos), 0.0) for action in node.untried]
    total = sum(weights)
    if total <= 0:
        return node.untried.pop(random.randrange(len(node.untried)))

    pick = random.random() * total
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if pick <= cumulative:
            return node.untried.pop(index)
    return node.untried.pop()


def _select_child(node: _MCTSNode, root_color: Color, c_puct: float, prior_fn: PriorFn) -> _MCTSNode:
    total = sum(child.visits for child in node.children.values())
    child_actions = [child.action for child in node.children.values() if child.action is not None]
    from_pos = node.state.pawn(node.player)
    prior = _normalized_prior(_ensure_prior_vector(node, prior_fn), child_actions, from_pos)
    best_score = -math.inf
    best_child = next(iter(node.children.values()))
    for child in node.children.values():
        q = child.value_sum / child.visits if child.visits else 0.0
        # value_sum is from the root player's view; the opponent minimizes it.
        if node.player != root_color:
            q = -q
        p = prior.get(encode(child.action, from_pos=from_pos), 0.0)  # type: ignore[arg-type]
        u = c_puct * p * math.sqrt(total + 1) / (1 + child.visits)
        score = q + u
        if score > best_score:
            best_score = score
            best_child = child
    return best_child


def _get_legal(state: QuoridorState, dist_cache) -> list[Action]:
    from quoridor.rules import get_legal_actions

    return get_legal_actions(state, dist_cache=dist_cache)

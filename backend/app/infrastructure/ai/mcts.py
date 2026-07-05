from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from quoridor.domain.actions import Action, action_child_key, encode
from quoridor.domain.state import Color, QuoridorState
from quoridor.rules import apply_action


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


def mcts_search(
    state: QuoridorState,
    color: Color,
    prior_fn,
    value_fn,
    legal_actions: list[Action],
    budget_ms: int = 450,
    c_puct: float = 1.4,
) -> Action:
    from app.middleware.metrics import metrics_store
    from quoridor.pathfinding import SimpleDistanceCache

    dist_cache = SimpleDistanceCache()
    root = _MCTSNode(state=state, parent=None, action=None, player=color, untried=list(legal_actions))
    root_prior = _normalized_prior(prior_fn(state, color), legal_actions)
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
            action = _pick_expansion(node, root_prior if node is root else prior_fn(node.state, color))
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


def _normalized_prior(prior: NDArray[np.floating], legal_actions: list[Action]) -> dict[int, float]:
    weights: dict[int, float] = {}
    for action in legal_actions:
        weights[encode(action)] = float(prior[encode(action)])
    total = sum(weights.values())
    if total <= 0:
        uniform = 1.0 / len(legal_actions)
        return {encode(action): uniform for action in legal_actions}
    return {idx: weight / total for idx, weight in weights.items()}


def _pick_expansion(node: _MCTSNode, prior: dict[int, float]) -> Action:
    if len(node.untried) == 1:
        return node.untried.pop()

    weights = [prior.get(encode(action), 0.0) for action in node.untried]
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


def _select_child(node: _MCTSNode, root_color: Color, c_puct: float, prior_fn) -> _MCTSNode:
    total = sum(child.visits for child in node.children.values())
    child_actions = [child.action for child in node.children.values() if child.action is not None]
    prior = _normalized_prior(prior_fn(node.state, root_color), child_actions)
    best_score = -math.inf
    best_child = next(iter(node.children.values()))
    for child in node.children.values():
        q = child.value_sum / child.visits if child.visits else 0.0
        p = prior.get(encode(child.action), 0.0)  # type: ignore[arg-type]
        u = c_puct * p * math.sqrt(total + 1) / (1 + child.visits)
        score = q + u
        if score > best_score:
            best_score = score
            best_child = child
    return best_child


def _get_legal(state: QuoridorState, dist_cache) -> list[Action]:
    from quoridor.rules import get_legal_actions

    return get_legal_actions(state, dist_cache=dist_cache)

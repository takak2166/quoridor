"""Second-player racing demonstrations for PPO (collect + behavior cloning)."""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from sb3_contrib import MaskablePPO

from app.infrastructure.ai.action_mask import legal_action_mask_agent_frame
from app.mappers.observation_mapper import to_observation
from quoridor.agent_frame import encode_for_viewer
from quoridor.domain.actions import Action, Move
from quoridor.domain.game import Game
from quoridor.domain.state import Color, QuoridorState
from quoridor.pathfinding import DistanceCache, distances
from quoridor.rules import apply_action, get_legal_actions

logger = logging.getLogger(__name__)

DEFAULT_WHITE_DEMO_WINS = 48
DEFAULT_WHITE_DEMO_EPOCHS = 4
DEFAULT_WHITE_DEMO_MAX_GAMES = 240
DEFAULT_WHITE_DEMO_MAX_MOVES = 200


@dataclass(frozen=True)
class DemoTransition:
    obs: NDArray[np.float32]
    action: int
    mask: NDArray[np.bool_]


def greedy_race_action(
    state: QuoridorState,
    color: Color,
    cache: DistanceCache | None = None,
) -> Action:
    """Pick the legal pawn move that most decreases this color's BFS distance."""
    legal = get_legal_actions(state, dist_cache=cache)
    if not legal:
        raise RuntimeError("no legal moves")
    moves = [action for action in legal if isinstance(action, Move) and action.to is not None]
    if not moves:
        return legal[0]

    best: Action | None = None
    best_dist = 10_000
    best_tie = 0
    for move in moves:
        nxt = apply_action(state, move)
        dist_white, dist_black = distances(nxt, cache)
        own = dist_white if color == "white" else dist_black
        if own is None:
            continue
        # Prefer shrinking distance; then prefer the more-forward destination.
        forward_row = move.to[0] if color == "white" else -move.to[0]
        tie = -forward_row
        if own < best_dist or (own == best_dist and tie < best_tie):
            best = move
            best_dist = own
            best_tie = tie
    return best if best is not None else moves[0]


def is_greedy_race_action(
    state: QuoridorState,
    color: Color,
    action: Action,
    cache: DistanceCache | None = None,
) -> bool:
    teacher = greedy_race_action(state, color, cache)
    if isinstance(action, Move) and isinstance(teacher, Move):
        return action.to == teacher.to
    return action == teacher


def _random_legal_action(state: QuoridorState, rng: random.Random) -> Action:
    legal = get_legal_actions(state)
    if not legal:
        raise RuntimeError("no legal moves")
    return rng.choice(legal)


def _record_white_transition(state: QuoridorState, action: Action) -> DemoTransition:
    from_pos = state.pawn("white")
    legal = get_legal_actions(state)
    return DemoTransition(
        obs=to_observation(state, "white"),
        action=encode_for_viewer(action, from_pos, "white"),
        mask=legal_action_mask_agent_frame(legal, "white", from_pos=from_pos),
    )


def collect_white_win_transitions(
    *,
    n_wins: int,
    max_games: int = DEFAULT_WHITE_DEMO_MAX_GAMES,
    max_moves: int = DEFAULT_WHITE_DEMO_MAX_MOVES,
    seed: int = 0,
) -> list[DemoTransition]:
    """Play greedy-racing White vs random Black; keep transitions from White wins.

    Random Black wanders instead of racing, so second-player pawn marches can
    actually finish. Those trajectories are the positive examples PPO never
    sees against a first-player racer.
    """
    if n_wins <= 0:
        return []

    rng = random.Random(seed)
    collected: list[DemoTransition] = []
    wins = 0
    games = 0
    for game_i in range(max_games):
        games = game_i + 1
        game = Game.from_initial()
        pending: list[DemoTransition] = []
        for _ in range(max_moves):
            if game.is_finished:
                break
            color = game.state.current_player
            if color == "white":
                action = greedy_race_action(game.state, "white")
                pending.append(_record_white_transition(game.state, action))
            else:
                action = _random_legal_action(game.state, rng)
            game.play(action)
        if game.winner == "white" and pending:
            collected.extend(pending)
            wins += 1
            if wins >= n_wins:
                break

    logger.info(
        "White-win demos: wins=%d/%d games, transitions=%d (requested %d wins)",
        wins,
        games,
        len(collected),
        n_wins,
    )
    if wins < n_wins:
        logger.warning(
            "White-win demo collection stopped at %d wins (wanted %d)",
            wins,
            n_wins,
        )
    return collected


def behavior_clone(
    model: MaskablePPO,
    transitions: Sequence[DemoTransition],
    *,
    epochs: int = DEFAULT_WHITE_DEMO_EPOCHS,
    batch_size: int = 64,
) -> float:
    """Supervised CE on demonstration actions. Returns the last batch loss."""
    if not transitions:
        raise ValueError("no demonstration transitions")
    if epochs <= 0:
        return 0.0

    policy = model.policy
    optimizer = policy.optimizer
    device = model.device
    was_training = bool(policy.training)
    policy.set_training_mode(True)

    obs = np.stack([t.obs for t in transitions])
    masks = np.stack([t.mask for t in transitions])
    actions = np.asarray([t.action for t in transitions], dtype=np.int64)
    n = len(transitions)
    last_loss = 0.0
    try:
        for _ in range(epochs):
            order = np.random.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                obs_t = torch.as_tensor(obs[idx], device=device)
                mask_t = torch.as_tensor(masks[idx], device=device)
                act_t = torch.as_tensor(actions[idx], device=device)
                dist = policy.get_distribution(obs_t, action_masks=mask_t)
                loss = -dist.log_prob(act_t).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().item())
    finally:
        policy.set_training_mode(was_training)

    logger.info(
        "White-win BC: transitions=%d epochs=%d last_loss=%.4f",
        n,
        epochs,
        last_loss,
    )
    return last_loss

"""Second-player racing demonstrations for PPO (collect + behavior cloning)."""

from __future__ import annotations

import logging
import multiprocessing as mp
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from sb3_contrib import MaskablePPO

from app.infrastructure.ai.action_mask import legal_action_mask_agent_frame
from app.mappers.observation_mapper import to_observation
from quoridor.agent_frame import encode_for_viewer
from quoridor.domain.actions import Action, Move, WallSlot
from quoridor.domain.game import Game
from quoridor.domain.state import Color, QuoridorState
from quoridor.pathfinding import DistanceCache, distances
from quoridor.rules import apply_action, get_legal_actions

logger = logging.getLogger(__name__)

DEFAULT_WHITE_DEMO_WINS = 48
DEFAULT_WHITE_DEMO_EPOCHS = 4
DEFAULT_WHITE_DEMO_MAX_GAMES = 240
DEFAULT_WHITE_DEMO_MAX_MOVES = 200
DEFAULT_BLACK_DEMO_WINS = 48
DEFAULT_BLACK_VS_NORMAL_MAX_GAMES = 800
DEFAULT_BLACK_DEMO_WORKERS = 1
DEFAULT_BLACK_PROBE_GAMES = 24
DEFAULT_EXPERT_VS_NORMAL_MAX_GAMES = 64
DEFAULT_EXPERT_MCTS_BUDGET_MS = 450

Chooser = Callable[[QuoridorState, Color, random.Random], Action]


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


def _record_transition(state: QuoridorState, action: Action, viewer: Color) -> DemoTransition:
    from_pos = state.pawn(viewer)
    legal = get_legal_actions(state)
    return DemoTransition(
        obs=to_observation(state, viewer),
        action=encode_for_viewer(action, from_pos, viewer),
        mask=legal_action_mask_agent_frame(legal, viewer, from_pos=from_pos),
    )


def _record_white_transition(state: QuoridorState, action: Action) -> DemoTransition:
    return _record_transition(state, action, "white")


def collect_win_transitions(
    *,
    target: Color,
    choose: Chooser,
    n_wins: int,
    max_games: int = DEFAULT_WHITE_DEMO_MAX_GAMES,
    max_moves: int = DEFAULT_WHITE_DEMO_MAX_MOVES,
    seed: int = 0,
    reseed_stdlib: bool = False,
    log_label: str = "win demos",
    stop_if_no_wins_after: int | None = None,
) -> list[DemoTransition]:
    """Play games with ``choose``; keep ``target``'s transitions from its wins."""
    if n_wins <= 0:
        return []

    collected: list[DemoTransition] = []
    wins = 0
    games = 0
    other_wins = 0
    unfinished = 0
    for game_i in range(max_games):
        games = game_i + 1
        game_seed = seed + game_i * 1009
        if reseed_stdlib:
            random.seed(game_seed)
        rng = random.Random(game_seed)
        game = Game.from_initial()
        pending: list[DemoTransition] = []
        plies = 0
        for _ in range(max_moves):
            if game.is_finished:
                break
            color = game.state.current_player
            action = choose(game.state, color, rng)
            if color == target:
                pending.append(_record_transition(game.state, action, target))
            game.play(action)
            plies += 1
        if game.winner == target and pending:
            collected.extend(pending)
            wins += 1
            logger.info("%s: game %d target win in %d plies", log_label, games, plies)
            if wins >= n_wins:
                break
        elif game.winner is None:
            unfinished += 1
        else:
            other_wins += 1
        if stop_if_no_wins_after and games >= stop_if_no_wins_after and wins == 0:
            logger.warning(
                "%s: 0/%d target wins; stopping early",
                log_label,
                games,
            )
            break

    logger.info(
        "%s: target_wins=%d other_wins=%d unfinished=%d / %d games, "
        "transitions=%d (requested %d wins, target=%s)",
        log_label,
        wins,
        other_wins,
        unfinished,
        games,
        len(collected),
        n_wins,
        target,
    )
    if wins < n_wins:
        logger.warning(
            "%s collection stopped at %d wins (wanted %d)",
            log_label,
            wins,
            n_wins,
        )
    return collected


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

    def choose(state: QuoridorState, color: Color, rng: random.Random) -> Action:
        if color == "white":
            return greedy_race_action(state, "white")
        return _random_legal_action(state, rng)

    return collect_win_transitions(
        target="white",
        choose=choose,
        n_wins=n_wins,
        max_games=max_games,
        max_moves=max_moves,
        seed=seed,
        log_label="White-win demos",
    )


def _format_action(action: Action) -> str:
    if isinstance(action, Move):
        dest = action.to if action.to is not None else action.direction
        return f"M{dest}"
    if isinstance(action, WallSlot):
        orient = "H" if action.orientation == "horizontal" else "V"
        return f"{orient}({action.row},{action.col})"
    return str(action)


def _node_limited_normal_policy():
    from app.config import settings
    from app.infrastructure.ai.minimax import MinimaxConfig, NormalMinimaxPolicy

    # Bind search by max_nodes, not the live 400ms budget. Parallel collection
    # with a wall-clock limit aborts mid-search and invents Black wins that
    # sequential Normal vs Normal never plays (all White, 64 plies).
    return NormalMinimaxPolicy(
        config=MinimaxConfig(
            time_budget_ms=60_000,
            max_nodes=settings.minimax_max_nodes_normal,
            max_wall_candidates=10,
            two_phase_search=True,
            primary_depth=settings.minimax_depth_normal,
            fallback_depth=max(2, settings.minimax_depth_normal - 2),
        )
    )


def _normal_chooser() -> Chooser:
    policy = _node_limited_normal_policy()

    def choose(state: QuoridorState, color: Color, rng: random.Random) -> Action:
        del rng
        return policy.select_move(state, color)

    return choose


def _expert_vs_normal_chooser(*, budget_ms: int = DEFAULT_EXPERT_MCTS_BUDGET_MS) -> Chooser:
    from app.config import settings
    from app.infrastructure.ai.factory import ExpertMCTSPolicy

    expert = ExpertMCTSPolicy(model_path=settings.model_expert, budget_ms=budget_ms)
    normal = _node_limited_normal_policy()

    def choose(state: QuoridorState, color: Color, rng: random.Random) -> Action:
        del rng
        if color == "black":
            return expert.select_move(state, color)
        return normal.select_move(state, color)

    return choose


def play_normal_vs_normal_game(
    game_i: int,
    *,
    max_moves: int = DEFAULT_WHITE_DEMO_MAX_MOVES,
) -> tuple[str | None, int]:
    """One Normal vs Normal game. Importable so multiprocessing spawn can pickle it."""
    winner, plies, _pending = _play_black_demo_game((game_i * 1009, max_moves))
    return winner, plies


def _play_black_demo_game(
    payload: tuple[int, int],
) -> tuple[str | None, int, list[DemoTransition]]:
    game_seed, max_moves = payload
    random.seed(game_seed)
    choose = _normal_chooser()
    game = Game.from_initial()
    pending: list[DemoTransition] = []
    dummy = random.Random(0)
    plies = 0
    for _ in range(max_moves):
        if game.is_finished:
            break
        color = game.state.current_player
        action = choose(game.state, color, dummy)
        if color == "black":
            pending.append(_record_transition(game.state, action, "black"))
        game.play(action)
        plies += 1
    if game.winner == "black" and pending:
        return "black", plies, pending
    return game.winner, plies, []


def collect_black_wins_vs_normal(
    *,
    n_wins: int,
    max_games: int = DEFAULT_BLACK_VS_NORMAL_MAX_GAMES,
    max_moves: int = DEFAULT_WHITE_DEMO_MAX_MOVES,
    seed: int = 0,
    workers: int = DEFAULT_BLACK_DEMO_WORKERS,
) -> list[DemoTransition]:
    """Play Normal Black vs Normal White; keep Black transitions from Black wins.

    Search is node-limited (not the live 400ms budget) so parallel workers do
    not time-abort into fake first-player wins.
    """
    if n_wins <= 0:
        return []
    if workers <= 1:
        return collect_win_transitions(
            target="black",
            choose=_normal_chooser(),
            n_wins=n_wins,
            max_games=max_games,
            max_moves=max_moves,
            seed=seed,
            reseed_stdlib=True,
            log_label="Black-win vs Normal demos",
            stop_if_no_wins_after=DEFAULT_BLACK_PROBE_GAMES,
        )

    collected: list[DemoTransition] = []
    wins = 0
    games = 0
    other_wins = 0
    unfinished = 0
    batch = max(workers * 2, workers)
    payloads = [(seed + i * 1009, max_moves) for i in range(max_games)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        for start in range(0, max_games, batch):
            if wins >= n_wins:
                break
            chunk = payloads[start : start + batch]
            for winner, plies, pending in pool.map(_play_black_demo_game, chunk):
                games += 1
                if pending:
                    collected.extend(pending)
                    wins += 1
                    logger.info(
                        "Black-win vs Normal demos: game %d target win in %d plies (wins=%d/%d)",
                        games,
                        plies,
                        wins,
                        n_wins,
                    )
                    if wins >= n_wins:
                        break
                elif winner is None:
                    unfinished += 1
                else:
                    other_wins += 1
            logger.info(
                "Black-win vs Normal demos: progress games=%d wins=%d other=%d unfinished=%d",
                games,
                wins,
                other_wins,
                unfinished,
            )
            if games >= DEFAULT_BLACK_PROBE_GAMES and wins == 0:
                logger.warning(
                    "Black-win vs Normal demos: 0/%d first-player wins; stopping early",
                    games,
                )
                break

    logger.info(
        "Black-win vs Normal demos: target_wins=%d other_wins=%d unfinished=%d / %d games, "
        "transitions=%d (requested %d wins, target=black, workers=%d)",
        wins,
        other_wins,
        unfinished,
        games,
        len(collected),
        n_wins,
        workers,
    )
    if wins < n_wins:
        logger.warning(
            "Black-win vs Normal demos collection stopped at %d wins (wanted %d)",
            wins,
            n_wins,
        )
    return collected


def _play_match_with_chooser(
    choose: Chooser,
    *,
    game_seed: int,
    max_moves: int,
    record_black: bool,
) -> tuple[str | None, int, str, list[DemoTransition]]:
    random.seed(game_seed)
    game = Game.from_initial()
    pending: list[DemoTransition] = []
    dummy = random.Random(0)
    plies = 0
    opening: list[str] = []
    for _ in range(max_moves):
        if game.is_finished:
            break
        color = game.state.current_player
        action = choose(game.state, color, dummy)
        if color == "black":
            if len(opening) < 3:
                opening.append(_format_action(action))
            if record_black:
                pending.append(_record_transition(game.state, action, "black"))
        game.play(action)
        plies += 1
    if game.winner != "black":
        pending = []
    return game.winner, plies, ",".join(opening), pending


def play_expert_black_vs_normal_white(
    game_i: int,
    *,
    max_moves: int = DEFAULT_WHITE_DEMO_MAX_MOVES,
    budget_ms: int = DEFAULT_EXPERT_MCTS_BUDGET_MS,
) -> tuple[str | None, int, str]:
    """Expert (MCTS, first) vs node-limited Normal (second)."""
    winner, plies, opening, _pending = _play_match_with_chooser(
        _expert_vs_normal_chooser(budget_ms=budget_ms),
        game_seed=game_i * 1009,
        max_moves=max_moves,
        record_black=False,
    )
    return winner, plies, opening


def iter_expert_black_vs_normal_games(
    n_games: int,
    *,
    seed: int = 0,
    max_moves: int = DEFAULT_WHITE_DEMO_MAX_MOVES,
    budget_ms: int = DEFAULT_EXPERT_MCTS_BUDGET_MS,
):
    """Reuse one Expert+Normal pair so the PPO zip is loaded once."""
    choose = _expert_vs_normal_chooser(budget_ms=budget_ms)
    for game_i in range(n_games):
        winner, plies, opening, _pending = _play_match_with_chooser(
            choose,
            game_seed=seed + game_i * 1009,
            max_moves=max_moves,
            record_black=False,
        )
        yield winner, plies, opening


def collect_black_wins_expert_vs_normal(
    *,
    n_wins: int,
    max_games: int = DEFAULT_EXPERT_VS_NORMAL_MAX_GAMES,
    max_moves: int = DEFAULT_WHITE_DEMO_MAX_MOVES,
    seed: int = 0,
    budget_ms: int = DEFAULT_EXPERT_MCTS_BUDGET_MS,
    stop_if_no_wins_after: int = 8,
) -> list[DemoTransition]:
    """Play Expert Black vs Normal White sequentially; keep Black's winning games.

    Sequential on purpose: MCTS is wall-clock limited (450ms). Parallel workers
    starve the budget the same way 400ms Normal did.
    """
    if n_wins <= 0:
        return []
    return collect_win_transitions(
        target="black",
        choose=_expert_vs_normal_chooser(budget_ms=budget_ms),
        n_wins=n_wins,
        max_games=max_games,
        max_moves=max_moves,
        seed=seed,
        reseed_stdlib=True,
        log_label="Black-win Expert vs Normal demos",
        stop_if_no_wins_after=stop_if_no_wins_after,
    )


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
                _values, log_prob, _entropy = policy.evaluate_actions(
                    obs_t,
                    act_t,
                    action_masks=mask_t,
                )
                loss = -log_prob.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().item())
    finally:
        policy.set_training_mode(was_training)

    logger.info(
        "BC: transitions=%d epochs=%d last_loss=%.4f",
        n,
        epochs,
        last_loss,
    )
    return last_loss

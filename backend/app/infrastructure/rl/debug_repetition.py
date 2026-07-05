"""Deterministic repetition (千日手) hunter for easy-vs-normal.

Normal is made deterministic via a node cap (instead of a wall-clock budget) so games
are reproducible and fast. This is only for diagnosing the oscillation mechanism; it
does not change the production Normal policy.
"""

from __future__ import annotations

import argparse
import random

from app.infrastructure.ai.factory import ai_for_difficulty
from app.infrastructure.ai.minimax import MinimaxConfig, NormalMinimaxPolicy
from app.infrastructure.rl.selfplay_debug import RepetitionDebugger
from quoridor.domain.game import Game
from quoridor.domain.state import Color


def deterministic_normal(depth: int = 3, max_nodes: int = 2500) -> NormalMinimaxPolicy:
    return NormalMinimaxPolicy(
        config=MinimaxConfig(
            time_budget_ms=10_000_000,
            max_nodes=max_nodes,
            max_wall_candidates=10,
            two_phase_search=True,
            primary_depth=depth,
            fallback_depth=max(2, depth - 1),
        )
    )


def hunt(seeds: int, max_moves: int, depth: int, max_nodes: int) -> bool:
    import time

    ai_easy = ai_for_difficulty("easy")
    ai_normal = deterministic_normal(depth, max_nodes)
    for seed in range(seeds):
        random.seed(seed)
        for parity in (0, 1):
            game = Game.from_initial()
            colors: tuple[Color, Color] = ("black", "white") if parity == 0 else ("white", "black")
            debugger = RepetitionDebugger(
                seed * 2 + parity,
                difficulty_a="easy",
                difficulty_b="normal",
                colors=colors,
            )
            moves = 0
            t0 = time.perf_counter()
            while not game.is_finished and moves < max_moves:
                color = game.state.current_player
                debugger.on_before_move(game.state, moves + 1)
                ai = ai_easy if color == colors[0] else ai_normal
                state_before = game.state.copy()
                action = ai.select_move(game.state, color)
                game.play(action)
                moves += 1
                debugger.on_after_move(state_before, color, action, moves)
                if debugger.repetition_events > 0:
                    print(
                        f"Repetition captured: seed={seed} parity={parity} "
                        f"colors={colors} at move {moves}",
                        flush=True,
                    )
                    return True
            print(
                f"seed={seed} parity={parity}: {game.winner or 'draw'} "
                f"moves={moves} time={time.perf_counter() - t0:.1f}s",
                flush=True,
            )
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--max-moves", type=int, default=300)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--max-nodes", type=int, default=2500)
    args = parser.parse_args()
    found = hunt(args.seeds, args.max_moves, args.depth, args.max_nodes)
    if not found:
        print("No repetition found in the scanned seeds.")


if __name__ == "__main__":
    main()

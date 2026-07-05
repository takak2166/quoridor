"""Self-play evaluation CLI."""

from __future__ import annotations

import argparse
import random
import time

from app.infrastructure.ai.factory import ai_for_difficulty
from app.infrastructure.rl.selfplay_debug import RepetitionDebugger
from quoridor.domain.game import Game
from quoridor.domain.state import Color


def run_eval(
    games: int,
    difficulty_a: str,
    difficulty_b: str,
    *,
    min_win_rate: float | None = None,
    max_p99_ms: float | None = None,
    max_moves: int | None = None,
    debug: bool = False,
    progress: bool = False,
    stop_after_repetition: bool = False,
    seed: int | None = None,
) -> dict[str, float]:
    if seed is not None:
        random.seed(seed)
    ai_a = ai_for_difficulty(difficulty_a)
    ai_b = ai_for_difficulty(difficulty_b)
    wins_a = 0
    timeouts = 0
    total_repetitions = 0
    latencies: list[float] = []

    for i in range(games):
        game = Game.from_initial()
        colors: tuple[Color, Color] = ("black", "white")
        if i % 2 == 1:
            colors = ("white", "black")
        debugger = (
            RepetitionDebugger(
                i + 1,
                difficulty_a=difficulty_a,
                difficulty_b=difficulty_b,
                colors=colors,
            )
            if debug
            else None
        )
        moves = 0
        game_t0 = time.perf_counter()
        while not game.is_finished:
            if max_moves is not None and moves >= max_moves:
                break
            color = game.state.current_player
            if debugger is not None:
                debugger.on_before_move(game.state, moves + 1)
            ai = ai_a if color == colors[0] else ai_b
            state_before = game.state.copy()
            start = time.perf_counter()
            action = ai.select_move(game.state, color)
            latencies.append((time.perf_counter() - start) * 1000)
            game.play(action)
            moves += 1
            if debugger is not None:
                debugger.on_after_move(state_before, color, action, moves)

        game_dt = time.perf_counter() - game_t0
        if game.winner == colors[0]:
            wins_a += 1
            result = difficulty_a
        elif game.winner is not None:
            result = difficulty_b
        else:
            timeouts += 1
            result = "timeout"
        if debugger is not None:
            total_repetitions += debugger.repetition_events
        if progress:
            rep_note = (
                f" repetitions={debugger.repetition_events}"
                if debugger is not None
                else ""
            )
            print(
                f"game {i + 1:2d}/{games}: {result:7s} moves={moves:3d} "
                f"time={game_dt:5.1f}s{rep_note}",
                flush=True,
            )
        if (
            stop_after_repetition
            and debugger is not None
            and debugger.repetition_events > 0
        ):
            print(f"Stopping after repetition captured in game {i + 1}.", flush=True)
            break

    latencies.sort()
    p95_idx = int(0.95 * (len(latencies) - 1)) if latencies else 0
    p99_idx = min(int(len(latencies) * 0.99), len(latencies) - 1) if latencies else 0
    p95 = latencies[p95_idx] if latencies else 0.0
    p99 = latencies[p99_idx] if latencies else 0.0
    win_rate = wins_a / games if games else 0.0
    finished = games - timeouts

    result = {
        "games": float(games),
        "win_rate": win_rate,
        "p95_ms": p95,
        "p99_ms": p99,
        "timeouts": float(timeouts),
        "finished": float(finished),
        "repetition_events": float(total_repetitions),
    }

    if min_win_rate is not None and win_rate < min_win_rate:
        raise SystemExit(
            f"Win rate gate failed: {difficulty_a} {win_rate:.2%} < {min_win_rate:.2%}"
        )
    if max_p99_ms is not None and p99 > max_p99_ms:
        raise SystemExit(f"Latency gate failed: P99 {p99:.1f}ms > {max_p99_ms:.1f}ms")

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--difficulty-a", type=str, default="easy")
    parser.add_argument("--difficulty-b", type=str, default="normal")
    parser.add_argument("--min-win-rate", type=float, default=None)
    parser.add_argument("--max-p99-ms", type=float, default=None)
    parser.add_argument("--max-moves", type=int, default=None, help="Stop game after N plies (draw)")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Log 千日手 (position repetition) cycles with eval scores to stderr",
    )
    parser.add_argument("--progress", action="store_true", help="Print per-game summary lines")
    parser.add_argument(
        "--stop-after-repetition",
        action="store_true",
        help="Stop the whole run after the first game that produces a 千日手 (debug capture)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed RNG for reproducibility")
    args = parser.parse_args()

    result = run_eval(
        args.games,
        args.difficulty_a,
        args.difficulty_b,
        min_win_rate=args.min_win_rate,
        max_p99_ms=args.max_p99_ms,
        max_moves=args.max_moves,
        debug=args.debug,
        progress=args.progress,
        stop_after_repetition=args.stop_after_repetition,
        seed=args.seed,
    )
    print(f"Games: {int(result['games'])}")
    print(f"Finished: {int(result['finished'])} (timeouts: {int(result['timeouts'])})")
    print(f"{args.difficulty_a} win rate: {result['win_rate']:.2%}")
    print(f"P95 inference ms: {result['p95_ms']:.1f}")
    print(f"P99 inference ms: {result['p99_ms']:.1f}")
    if args.debug:
        print(f"Repetition events: {int(result['repetition_events'])}")


if __name__ == "__main__":
    main()

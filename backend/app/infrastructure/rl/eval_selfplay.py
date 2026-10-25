"""Self-play evaluation CLI."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from app.infrastructure.ai.factory import ai_for_difficulty
from app.infrastructure.rl.selfplay_debug import RepetitionDebugger
from app.infrastructure.rl.white_demonstrations import _format_action
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
    scoresheet_dir: str | Path | None = None,
    scoresheet_losses_only: bool = False,
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
        labels: list[str] = []
        from app.infrastructure.ai.inference_context import inference_session

        game_t0 = time.perf_counter()
        session_key = f"eval-{seed}-{i}" if seed is not None else f"eval-{i}"
        with inference_session(session_key):
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
                labels.append(_format_action(action))
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
        if scoresheet_dir is not None:
            lost = game.winner is not None and game.winner != colors[0]
            if lost or not scoresheet_losses_only:
                from app.infrastructure.rl.dagger_losses import write_hunt_scoresheet

                dump = Path(scoresheet_dir)
                dump.mkdir(parents=True, exist_ok=True)
                kind = "loss" if lost else (game.winner or "timeout")
                write_hunt_scoresheet(
                    dump / f"game_{i + 1:03d}_{colors[0]}_{kind}_{moves}.txt",
                    tag=f"eval-{colors[0]}-{kind}",
                    winner=game.winner,
                    plies=moves,
                    opening=",".join(labels[:10]),
                    scoresheet=",".join(labels),
                )
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


def run_eval_multi_seed(
    games: int,
    difficulty_a: str,
    difficulty_b: str,
    seeds: list[int],
    *,
    min_win_rate: float | None = None,
    max_p99_ms: float | None = None,
    max_moves: int | None = None,
    debug: bool = False,
    progress: bool = False,
    stop_after_repetition: bool = False,
    scoresheet_dir: str | Path | None = None,
    scoresheet_losses_only: bool = False,
) -> dict[str, float]:
    """Run self-play once per seed (sequentially) and gate on mean win rate."""
    if not seeds:
        raise ValueError("seeds must be non-empty")

    per_seed: list[dict[str, float]] = []
    for seed in seeds:
        seed_scoresheet = None
        if scoresheet_dir is not None:
            seed_scoresheet = Path(scoresheet_dir) / f"seed_{seed}"
        one = run_eval(
            games,
            difficulty_a,
            difficulty_b,
            min_win_rate=None,
            max_p99_ms=None,
            max_moves=max_moves,
            debug=debug,
            progress=progress,
            stop_after_repetition=stop_after_repetition,
            seed=seed,
            scoresheet_dir=seed_scoresheet,
            scoresheet_losses_only=scoresheet_losses_only,
        )
        per_seed.append(one)
        print(
            f"seed {seed}: {difficulty_a} win rate {one['win_rate']:.2%}, "
            f"P99 {one['p99_ms']:.1f}ms, timeouts {int(one['timeouts'])}",
            flush=True,
        )
        if stop_after_repetition and one["repetition_events"] > 0:
            print(
                f"Stopping multi-seed run after repetition in seed {seed}.",
                flush=True,
            )
            break

    avg_win_rate = sum(r["win_rate"] for r in per_seed) / len(per_seed)
    max_p99 = max(r["p99_ms"] for r in per_seed)
    total_timeouts = sum(int(r["timeouts"]) for r in per_seed)
    gate_note = f", gate min {min_win_rate:.0%}" if min_win_rate is not None else ""
    print(
        f"Seeds {seeds}: mean {difficulty_a} win rate {avg_win_rate:.2%}{gate_note}, "
        f"max P99 {max_p99:.1f}ms",
        flush=True,
    )

    if min_win_rate is not None and avg_win_rate < min_win_rate:
        raise SystemExit(
            f"Mean win rate gate failed: {difficulty_a} {avg_win_rate:.2%} < {min_win_rate:.2%} "
            f"across seeds {seeds}"
        )
    if max_p99_ms is not None and max_p99 > max_p99_ms:
        raise SystemExit(
            f"Latency gate failed: max P99 {max_p99:.1f}ms > {max_p99_ms:.1f}ms across seeds {seeds}"
        )

    return {
        "games": float(games * len(seeds)),
        "win_rate": avg_win_rate,
        "p95_ms": max(r["p95_ms"] for r in per_seed),
        "p99_ms": max_p99,
        "timeouts": float(total_timeouts),
        "finished": sum(r["finished"] for r in per_seed),
        "repetition_events": sum(r["repetition_events"] for r in per_seed),
        "seeds": float(len(seeds)),
    }


def _parse_seeds(raw: str | None) -> list[int] | None:
    if raw is None or not raw.strip():
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


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
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds; run sequentially and gate on mean win rate",
    )
    parser.add_argument(
        "--scoresheet-dir",
        type=str,
        default=None,
        help="Write per-game scoresheets here",
    )
    parser.add_argument(
        "--scoresheet-losses-only",
        action="store_true",
        help="When dumping scoresheets, keep only Hard losses",
    )
    args = parser.parse_args()
    if args.seed is not None and args.seeds:
        parser.error("use either --seed or --seeds, not both")

    seeds: list[int] | None = None
    if args.seeds is not None:
        seeds = _parse_seeds(args.seeds)
        if not seeds:
            parser.error("--seeds requires at least one integer seed")
    eval_kwargs = {
        "max_moves": args.max_moves,
        "debug": args.debug,
        "progress": args.progress,
        "stop_after_repetition": args.stop_after_repetition,
        "scoresheet_dir": args.scoresheet_dir,
        "scoresheet_losses_only": args.scoresheet_losses_only,
    }
    if seeds:
        result = run_eval_multi_seed(
            args.games,
            args.difficulty_a,
            args.difficulty_b,
            seeds,
            min_win_rate=args.min_win_rate,
            max_p99_ms=args.max_p99_ms,
            **eval_kwargs,
        )
    else:
        result = run_eval(
            args.games,
            args.difficulty_a,
            args.difficulty_b,
            min_win_rate=args.min_win_rate,
            max_p99_ms=args.max_p99_ms,
            seed=args.seed,
            **eval_kwargs,
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

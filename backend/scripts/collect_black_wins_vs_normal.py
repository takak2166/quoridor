#!/usr/bin/env python3
"""Measure Normal (black / first) vs Normal (white / second) win rates."""

from __future__ import annotations

import argparse
import collections
import multiprocessing as mp
import sys

from app.infrastructure.rl.white_demonstrations import (
    collect_black_wins_vs_normal,
    play_normal_vs_normal_game,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--collect-wins",
        type=int,
        default=0,
        help="If >0, also run sequential BC collection (fails if zero black wins)",
    )
    args = parser.parse_args()

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        results = pool.map(play_normal_vs_normal_game, range(args.games))
    counts: collections.Counter[str] = collections.Counter()
    for winner, plies in results:
        label = winner if winner is not None else "unfinished"
        counts[label] += 1
        print(f"winner={label} plies={plies}", flush=True)
    print("summary", dict(counts), flush=True)
    black_wins = int(counts.get("black", 0))
    print(f"black_wins={black_wins}/{args.games}", flush=True)

    if args.collect_wins > 0:
        demos = collect_black_wins_vs_normal(
            n_wins=args.collect_wins,
            max_games=args.games,
            seed=0,
        )
        print(f"transitions={len(demos)}", flush=True)
        if not demos:
            return 1
    return 0 if black_wins else 1


if __name__ == "__main__":
    sys.exit(main())

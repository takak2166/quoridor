#!/usr/bin/env python3
"""Probe Expert (black / first) vs node-limited Normal (white / second)."""

from __future__ import annotations

import argparse
import collections
import sys

from app.infrastructure.rl.white_demonstrations import (
    collect_black_wins_expert_vs_normal,
    iter_expert_black_vs_normal_games,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--budget-ms", type=int, default=450)
    parser.add_argument(
        "--collect-wins",
        type=int,
        default=0,
        help="If >0, also collect Black winning transitions (sequential)",
    )
    args = parser.parse_args()

    counts: collections.Counter[str] = collections.Counter()
    for i, (winner, plies, opening) in enumerate(
        iter_expert_black_vs_normal_games(args.games, budget_ms=args.budget_ms)
    ):
        label = winner if winner is not None else "unfinished"
        counts[label] += 1
        print(f"game={i} winner={label} plies={plies} opening={opening}", flush=True)
    print("summary", dict(counts), flush=True)
    black_wins = int(counts.get("black", 0))
    print(f"black_wins={black_wins}/{args.games}", flush=True)

    if args.collect_wins > 0:
        demos = collect_black_wins_expert_vs_normal(
            n_wins=args.collect_wins,
            max_games=args.games,
            budget_ms=args.budget_ms,
            stop_if_no_wins_after=args.games,
        )
        print(f"transitions={len(demos)}", flush=True)
        if not demos:
            return 1
    return 0 if black_wins else 1


if __name__ == "__main__":
    sys.exit(main())

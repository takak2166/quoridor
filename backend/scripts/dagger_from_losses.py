#!/usr/bin/env python3
"""Hunt teacher wins from Hard's first off-book / uncovered ply."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.infrastructure.rl.dagger_losses import (
    format_prefix_csv,
    load_hard_loss_texts,
    unique_divergences,
    write_hunt_scoresheet,
)
from app.infrastructure.rl.hunt_black_wins import play_opening_vs_normal
from app.infrastructure.rl.white_demonstrations import load_teacher_book
from quoridor.domain.state import Color


def _hunt(
    *,
    ranked: list,
    target: Color,
    black_kind: str,
    white_kind: str,
    repeats: int,
    max_moves: int,
    max_wins: int,
    out_dir: Path,
) -> int:
    wins = 0
    for found, count in ranked:
        prefix = found.prefix
        csv = format_prefix_csv(prefix)
        print(
            f"PREFIX {target} ply={found.ply} reason={found.reason} n={count} prefix={csv or '-'}",
            flush=True,
        )
        for repeat_i in range(repeats):
            tag = f"dagger-{target}-p{found.ply}-{found.reason}-n{count}#r{repeat_i}"
            result = play_opening_vs_normal((prefix, max_moves, black_kind, white_kind, tag))
            label = result.winner if result.winner is not None else "unfinished"
            print(
                f"  hunt winner={label} plies={result.plies} tag={result.tag}",
                flush=True,
            )
            if result.winner == target:
                wins += 1
                path = out_dir / f"{target}_win_{wins:03d}_{tag.replace('#', '_')}.txt"
                write_hunt_scoresheet(
                    path,
                    tag=result.tag,
                    winner=result.winner,
                    plies=result.plies,
                    opening=result.opening,
                    scoresheet=result.scoresheet,
                )
                print(f"  SAVED {path}", flush=True)
                if max_wins > 0 and wins >= max_wins:
                    return wins
    return wins


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss-dir", type=Path, required=True)
    parser.add_argument("--black-book", type=str, default="artifacts/black_wins_vs_400ms/pawn_first")
    parser.add_argument("--white-book", type=str, default="artifacts/white_wins_vs_400ms/pawn_first")
    parser.add_argument("--black-stem", type=str, default="M14_M15_M25")
    parser.add_argument("--white-stem", type=str, default="M_7_4_M_2_4_M_6_4")
    parser.add_argument("--white-out", type=Path, default=Path("artifacts/white_wins_vs_400ms/dagger"))
    parser.add_argument("--black-out", type=Path, default=Path("artifacts/black_wins_vs_400ms/dagger"))
    parser.add_argument("--max-white-prefixes", type=int, default=8)
    parser.add_argument("--max-black-prefixes", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-white-wins", type=int, default=12)
    parser.add_argument("--max-black-wins", type=int, default=8)
    parser.add_argument("--max-moves", type=int, default=200)
    args = parser.parse_args()

    book = load_teacher_book(
        black_source=args.black_book,
        white_source=args.white_book,
        black_prefer_stem=args.black_stem,
        white_prefer_stem=args.white_stem,
    )
    print(f"Teacher book positions={len(book.actions)}", flush=True)

    white_losses = load_hard_loss_texts(args.loss_dir, "white")
    black_losses = load_hard_loss_texts(args.loss_dir, "black")
    print(f"loss sheets white={len(white_losses)} black={len(black_losses)}", flush=True)

    white_ranked = unique_divergences(
        white_losses, book, "white", limit=args.max_white_prefixes
    )
    black_ranked = unique_divergences(
        black_losses, book, "black", limit=args.max_black_prefixes
    )
    print(f"unique prefixes white={len(white_ranked)} black={len(black_ranked)}", flush=True)

    white_wins = _hunt(
        ranked=white_ranked,
        target="white",
        black_kind="factory",
        white_kind="node-limited",
        repeats=args.repeats,
        max_moves=args.max_moves,
        max_wins=args.max_white_wins,
        out_dir=args.white_out,
    )
    black_wins = _hunt(
        ranked=black_ranked,
        target="black",
        black_kind="node-limited",
        white_kind="factory",
        repeats=args.repeats,
        max_moves=args.max_moves,
        max_wins=args.max_black_wins,
        out_dir=args.black_out,
    )
    print(f"dagger_wins white={white_wins} black={black_wins}", flush=True)
    return 0 if (white_wins + black_wins) else 1


if __name__ == "__main__":
    sys.exit(main())

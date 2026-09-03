#!/usr/bin/env python3
"""Search for Black wins vs Normal (second) via forced openings and asymmetric Black."""

from __future__ import annotations

import argparse
import collections
import multiprocessing as mp
import os
import re
import sys
from pathlib import Path

from app.infrastructure.rl.hunt_black_wins import (
    BLACK_KINDS,
    WHITE_KINDS,
    HuntResult,
    _select_fn,
    encode_prefix_action,
    format_numbered_scoresheet,
    format_prefix_spec,
    parse_scoresheet,
    play_opening_vs_normal,
)
from app.infrastructure.rl.white_demonstrations import _format_action
from quoridor.domain.actions import Move
from quoridor.domain.state import initial_state
from quoridor.rules import apply_action, get_legal_actions

Payload = tuple[tuple, int, str, str, str]

# Horizontal walls that block White's (8,4)->(7,4) first step.
FACE_WHITE_SPECS = (("H", 7, 3), ("H", 7, 4))


def _legal(state, pawns_only: bool):
    legal = get_legal_actions(state)
    if pawns_only:
        return [action for action in legal if isinstance(action, Move)]
    return legal


def _payload(
    specs: tuple,
    max_moves: int,
    black_kind: str,
    white_kind: str,
    tag: str,
) -> Payload:
    return (specs, max_moves, black_kind, white_kind, tag)


def _first_move_payloads(
    max_moves: int, pawns_only: bool, black_kind: str, white_kind: str
) -> list[Payload]:
    payloads: list[Payload] = []
    for action in _legal(initial_state(), pawns_only):
        spec = encode_prefix_action(action)
        payloads.append(
            _payload(
                (spec,),
                max_moves,
                black_kind,
                white_kind,
                f"first:{format_prefix_spec(spec)}",
            )
        )
    return payloads


def _after_face_wall_payloads(
    max_moves: int, pawns_only: bool, black_kind: str, white_kind: str
) -> list[Payload]:
    """Black (1,4), White's Normal reply, then every Black second move."""
    black_fwd = next(
        action
        for action in get_legal_actions(initial_state())
        if isinstance(action, Move) and action.to == (1, 4)
    )
    after_fwd = apply_action(initial_state(), black_fwd)
    white_reply = _select_fn(white_kind)(after_fwd, "white")
    after_white = apply_action(after_fwd, white_reply)
    replies = _legal(after_white, pawns_only)
    prefix_head = (encode_prefix_action(black_fwd), encode_prefix_action(white_reply))
    print(
        f"White reply to M(1, 4) is {_format_action(white_reply)}; {len(replies)} Black seconds",
        flush=True,
    )
    payloads: list[Payload] = []
    for action in replies:
        spec = encode_prefix_action(action)
        payloads.append(
            _payload(
                prefix_head + (spec,),
                max_moves,
                black_kind,
                white_kind,
                f"face-wall:{format_prefix_spec(spec)}",
            )
        )
    return payloads


def _pawn_second_payloads(max_moves: int, black_kind: str, white_kind: str) -> list[Payload]:
    """Every first pawn move, White's Normal reply, then every Black pawn second."""
    white_select = _select_fn(white_kind)
    payloads: list[Payload] = []
    for first in _legal(initial_state(), pawns_only=True):
        after_first = apply_action(initial_state(), first)
        white_reply = white_select(after_first, "white")
        after_white = apply_action(after_first, white_reply)
        head = (encode_prefix_action(first), encode_prefix_action(white_reply))
        for second in _legal(after_white, pawns_only=True):
            spec = encode_prefix_action(second)
            payloads.append(
                _payload(
                    head + (spec,),
                    max_moves,
                    black_kind,
                    white_kind,
                    f"pawn-second:{format_prefix_spec(encode_prefix_action(first))}"
                    f"+{_format_action(white_reply)}+{format_prefix_spec(spec)}",
                )
            )
    return payloads


def _face_white_payloads(max_moves: int, black_kind: str, white_kind: str) -> list[Payload]:
    payloads: list[Payload] = []
    for spec in FACE_WHITE_SPECS:
        payloads.append(
            _payload(
                (spec,),
                max_moves,
                black_kind,
                white_kind,
                f"face-white:{format_prefix_spec(spec)}",
            )
        )
    return payloads


def _asymmetric_payloads(max_moves: int, black_kind: str, white_kind: str) -> list[Payload]:
    return [_payload((), max_moves, black_kind, white_kind, f"asymmetric:{black_kind}")]


def _save_win(out_dir: Path, result: HuntResult, index: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", result.tag).strip("_")
    path = out_dir / f"black_win_{index:03d}_{safe_tag}.txt"
    numbered = format_numbered_scoresheet(parse_scoresheet(result.scoresheet))
    path.write_text(
        "\n".join(
            [
                f"tag={result.tag}",
                f"winner={result.winner}",
                f"plies={result.plies}",
                f"opening={result.opening}",
                f"scoresheet={result.scoresheet}",
                "",
                numbered,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["first", "face-wall", "pawn-second", "face-white", "asymmetric"],
        default="first",
    )
    parser.add_argument("--black-kind", choices=BLACK_KINDS, default="normal")
    parser.add_argument("--white-kind", choices=WHITE_KINDS, default="node-limited")
    parser.add_argument("--games", type=int, default=0, help="Cap (0 = all openings)")
    parser.add_argument("--repeats", type=int, default=1, help="Replay each opening (400ms jitter)")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 8))
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--pawns-only", action="store_true")
    parser.add_argument("--stop-on-win", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/black_wins_vs_normal"),
    )
    args = parser.parse_args()

    if args.white_kind == "factory" and args.workers > 1:
        print(
            f"factory 400ms Normal: forcing workers=1 (requested {args.workers}) "
            "to avoid time-budget starvation",
            flush=True,
        )
        args.workers = 1

    if args.mode == "first":
        payloads = _first_move_payloads(
            args.max_moves, args.pawns_only, args.black_kind, args.white_kind
        )
    elif args.mode == "face-wall":
        payloads = _after_face_wall_payloads(
            args.max_moves, args.pawns_only, args.black_kind, args.white_kind
        )
    elif args.mode == "pawn-second":
        payloads = _pawn_second_payloads(args.max_moves, args.black_kind, args.white_kind)
    elif args.mode == "face-white":
        payloads = _face_white_payloads(args.max_moves, args.black_kind, args.white_kind)
    else:
        payloads = _asymmetric_payloads(args.max_moves, args.black_kind, args.white_kind)

    if args.repeats > 1:
        repeated: list[Payload] = []
        for spec, max_moves, black_kind, white_kind, tag in payloads:
            for repeat_i in range(args.repeats):
                repeated.append(
                    (spec, max_moves, black_kind, white_kind, f"{tag}#r{repeat_i}")
                )
        payloads = repeated
    if args.games > 0:
        payloads = payloads[: args.games]
    print(
        f"mode={args.mode} black_kind={args.black_kind} white_kind={args.white_kind} "
        f"openings={len(payloads)} workers={args.workers}",
        flush=True,
    )

    counts: collections.Counter[str] = collections.Counter()
    black_wins = 0
    workers = max(1, min(args.workers, len(payloads)))
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        for result in pool.imap_unordered(play_opening_vs_normal, payloads):
            label = result.winner if result.winner is not None else "unfinished"
            counts[label] += 1
            print(
                f"winner={label} plies={result.plies} tag={result.tag} opening={result.opening}",
                flush=True,
            )
            if result.winner == "black":
                black_wins += 1
                saved = _save_win(args.out_dir, result, black_wins)
                print(f"SAVED {saved} scoresheet={result.scoresheet}", flush=True)
                if args.stop_on_win:
                    pool.terminate()
                    break

    print("summary", dict(counts), flush=True)
    print(f"black_wins={black_wins}/{len(payloads)}", flush=True)
    return 0 if black_wins else 1


if __name__ == "__main__":
    sys.exit(main())

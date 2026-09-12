#!/usr/bin/env python3
"""Hunt second-player wins vs factory/node-limited Normal (first)."""

from __future__ import annotations

import argparse
import collections
import multiprocessing as mp
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
from quoridor.domain.actions import Move
from quoridor.domain.state import initial_state
from quoridor.rules import apply_action, get_legal_actions

Payload = tuple[tuple, int, str, str, str]


def _legal(state, pawns_only: bool):
    legal = get_legal_actions(state)
    if pawns_only:
        return [action for action in legal if isinstance(action, Move)]
    return legal


def _payload(specs: tuple, max_moves: int, black_kind: str, white_kind: str, tag: str) -> Payload:
    return (specs, max_moves, black_kind, white_kind, tag)


def _asymmetric_payloads(max_moves: int, black_kind: str, white_kind: str) -> list[Payload]:
    return [_payload((), max_moves, black_kind, white_kind, f"asymmetric:{white_kind}")]


def _white_first_payloads(max_moves: int, black_kind: str, white_kind: str) -> list[Payload]:
    """Force Black M(1,4), then every White pawn first."""
    black_fwd = next(
        action
        for action in get_legal_actions(initial_state())
        if isinstance(action, Move) and action.to == (1, 4)
    )
    after = apply_action(initial_state(), black_fwd)
    head = (encode_prefix_action(black_fwd),)
    payloads: list[Payload] = []
    for action in _legal(after, pawns_only=True):
        spec = encode_prefix_action(action)
        payloads.append(
            _payload(
                head + (spec,),
                max_moves,
                black_kind,
                white_kind,
                f"white-first:{format_prefix_spec(spec)}",
            )
        )
    return payloads


def _white_pawn_second_payloads(max_moves: int, black_kind: str, white_kind: str) -> list[Payload]:
    """Black M(1,4), each White pawn, factory/teacher Black reply, each White pawn second."""
    black_select = _select_fn(black_kind)
    black_fwd = next(
        action
        for action in get_legal_actions(initial_state())
        if isinstance(action, Move) and action.to == (1, 4)
    )
    after_black = apply_action(initial_state(), black_fwd)
    payloads: list[Payload] = []
    for white1 in _legal(after_black, pawns_only=True):
        after_white1 = apply_action(after_black, white1)
        black2 = black_select(after_white1, "black")
        after_black2 = apply_action(after_white1, black2)
        head = (
            encode_prefix_action(black_fwd),
            encode_prefix_action(white1),
            encode_prefix_action(black2),
        )
        for white2 in _legal(after_black2, pawns_only=True):
            spec = encode_prefix_action(white2)
            payloads.append(
                _payload(
                    head + (spec,),
                    max_moves,
                    black_kind,
                    white_kind,
                    "white-pawn-second:"
                    f"{format_prefix_spec(head[1])}"
                    f"+{_format_black(black2)}"
                    f"+{format_prefix_spec(spec)}",
                )
            )
    print(f"white-pawn-second openings={len(payloads)}", flush=True)
    return payloads


def _format_black(action) -> str:
    from app.infrastructure.rl.white_demonstrations import _format_action

    return _format_action(action)


def _save_win(out_dir: Path, result: HuntResult, index: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", result.tag).strip("_")
    path = out_dir / f"white_win_{index:03d}_{safe_tag}.txt"
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
        choices=["asymmetric", "white-first", "white-pawn-second", "prefix"],
        default="asymmetric",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Forced opening scoresheet for --mode prefix",
    )
    parser.add_argument("--black-kind", choices=BLACK_KINDS, default="factory")
    parser.add_argument("--white-kind", choices=WHITE_KINDS, default="greedy")
    parser.add_argument("--games", type=int, default=0, help="Cap (0 = all openings)")
    parser.add_argument("--repeats", type=int, default=1, help="Replay each opening (400ms jitter)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--stop-on-win", action="store_true")
    parser.add_argument(
        "--max-wins",
        type=int,
        default=0,
        help="Stop after this many White wins (0 = no cap)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/white_wins_vs_400ms"),
    )
    args = parser.parse_args()

    if args.black_kind == "factory" and args.workers > 1:
        print(
            f"factory 400ms Normal: forcing workers=1 (requested {args.workers}) "
            "to avoid time-budget starvation",
            flush=True,
        )
        args.workers = 1

    if args.mode == "asymmetric":
        payloads = _asymmetric_payloads(args.max_moves, args.black_kind, args.white_kind)
    elif args.mode == "white-first":
        payloads = _white_first_payloads(args.max_moves, args.black_kind, args.white_kind)
    elif args.mode == "prefix":
        from app.infrastructure.rl.dagger_losses import prefix_from_csv

        if not args.prefix:
            raise SystemExit("--mode prefix requires --prefix")
        specs = prefix_from_csv(args.prefix)
        payloads = [_payload(specs, args.max_moves, args.black_kind, args.white_kind, "prefix")]
    else:
        payloads = _white_pawn_second_payloads(args.max_moves, args.black_kind, args.white_kind)

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
    white_wins = 0
    workers = max(1, min(args.workers, len(payloads)))
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        iterator = pool.imap if args.workers == 1 else pool.imap_unordered
        for result in iterator(play_opening_vs_normal, payloads):
            label = result.winner if result.winner is not None else "unfinished"
            counts[label] += 1
            print(
                f"winner={label} plies={result.plies} tag={result.tag} opening={result.opening}",
                flush=True,
            )
            if result.winner == "white":
                white_wins += 1
                saved = _save_win(args.out_dir, result, white_wins)
                print(f"SAVED {saved} scoresheet={result.scoresheet}", flush=True)
                if args.stop_on_win or (args.max_wins > 0 and white_wins >= args.max_wins):
                    pool.terminate()
                    break

    print("summary", dict(counts), flush=True)
    print(f"white_wins={white_wins}/{len(payloads)}", flush=True)
    return 0 if white_wins else 1


if __name__ == "__main__":
    sys.exit(main())

"""Hunt first-player wins against node-limited Normal (second)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.infrastructure.rl.white_demonstrations import (
    DEFAULT_WHITE_DEMO_MAX_MOVES,
    _format_action,
    _node_limited_normal_policy,
    greedy_race_action,
)
from quoridor.domain.actions import Action, Move, WallSlot
from quoridor.domain.game import Game
from quoridor.domain.state import QuoridorState
from quoridor.rules import get_legal_actions

PrefixSpec = tuple[str, int, int]

BLACK_KINDS = ("normal", "greedy", "deep")

_SCORESHEET_TOKEN = re.compile(r"([MHV])\((\d+),\s*(\d+)\)")


@dataclass(frozen=True)
class HuntResult:
    winner: str | None
    plies: int
    opening: str
    scoresheet: str
    tag: str


def encode_prefix_action(action: Action) -> PrefixSpec:
    if isinstance(action, Move) and action.to is not None:
        return ("M", action.to[0], action.to[1])
    if isinstance(action, WallSlot):
        tag = "H" if action.orientation == "horizontal" else "V"
        return (tag, action.row, action.col)
    raise ValueError(f"unsupported action {action!r}")


def format_prefix_spec(spec: PrefixSpec) -> str:
    kind, row, col = spec
    if kind == "M":
        return f"M({row}, {col})"
    return f"{kind}({row},{col})"


def resolve_prefix_action(state: QuoridorState, spec: PrefixSpec) -> Action | None:
    kind, row, col = spec
    legal = get_legal_actions(state)
    if kind == "M":
        return next(
            (act for act in legal if isinstance(act, Move) and act.to == (row, col)),
            None,
        )
    want = WallSlot(
        orientation="horizontal" if kind == "H" else "vertical",
        row=row,
        col=col,
    )
    return next((act for act in legal if act == want), None)


def parse_scoresheet(text: str) -> list[PrefixSpec]:
    for line in text.splitlines():
        if line.startswith("scoresheet="):
            text = line.split("=", 1)[1]
            break
    return [
        (match.group(1), int(match.group(2)), int(match.group(3)))
        for match in _SCORESHEET_TOKEN.finditer(text)
    ]


def format_numbered_scoresheet(specs: list[PrefixSpec] | tuple[PrefixSpec, ...]) -> str:
    lines: list[str] = []
    for ply, spec in enumerate(specs, start=1):
        side = "Black" if ply % 2 == 1 else "White"
        lines.append(f"{ply:3d}. {side} {format_prefix_spec(spec)}")
    return "\n".join(lines)


def replay_scoresheet(text: str) -> HuntResult:
    specs = tuple(parse_scoresheet(text))
    game = Game.from_initial()
    labels: list[str] = []
    for spec in specs:
        action = resolve_prefix_action(game.state, spec)
        if action is None:
            return HuntResult(
                winner="illegal",
                plies=len(labels),
                opening=",".join(labels) if labels else format_prefix_spec(spec),
                scoresheet=",".join(labels),
                tag="replay",
            )
        labels.append(_format_action(action))
        game.play(action)
        if game.is_finished:
            return _result_from_game(game, labels, "replay")
    return _result_from_game(game, labels, "replay")


def apply_prefix(specs: tuple[PrefixSpec, ...]) -> QuoridorState | None:
    game = Game.from_initial()
    for spec in specs:
        action = resolve_prefix_action(game.state, spec)
        if action is None:
            return None
        game.play(action)
        if game.is_finished:
            return game.state
    return game.state


def _deep_black_policy():
    from app.config import settings
    from app.infrastructure.ai.minimax import MinimaxConfig, NormalMinimaxPolicy

    return NormalMinimaxPolicy(
        config=MinimaxConfig(
            time_budget_ms=60_000,
            max_nodes=max(4000, settings.minimax_max_nodes_normal * 4),
            max_wall_candidates=12,
            two_phase_search=True,
            primary_depth=settings.minimax_depth_normal + 2,
            fallback_depth=settings.minimax_depth_normal,
        )
    )


def _black_action_fn(kind: str):
    if kind not in BLACK_KINDS:
        raise ValueError(f"unknown black kind {kind!r}")
    if kind == "greedy":
        return lambda state, color: greedy_race_action(state, color)
    if kind == "deep":
        policy = _deep_black_policy()
        return policy.select_move
    policy = _node_limited_normal_policy()
    return policy.select_move


def _play_specs_then_policies(
    specs: tuple[PrefixSpec, ...],
    *,
    max_moves: int,
    black_kind: str,
    tag: str,
) -> HuntResult:
    white_policy = _node_limited_normal_policy()
    black_select = _black_action_fn(black_kind)
    game = Game.from_initial()
    labels: list[str] = []

    for spec in specs:
        action = resolve_prefix_action(game.state, spec)
        if action is None:
            return HuntResult(
                winner="illegal",
                plies=len(labels),
                opening=",".join(labels) if labels else format_prefix_spec(spec),
                scoresheet=",".join(labels),
                tag=tag,
            )
        labels.append(_format_action(action))
        game.play(action)
        if game.is_finished:
            return _result_from_game(game, labels, tag)

    for _ in range(max_moves):
        if game.is_finished:
            break
        color = game.state.current_player
        if color == "black":
            action = black_select(game.state, color)
        else:
            action = white_policy.select_move(game.state, color)
        labels.append(_format_action(action))
        game.play(action)
    return _result_from_game(game, labels, tag)


def _result_from_game(game: Game, labels: list[str], tag: str) -> HuntResult:
    opening = ",".join(labels[:10])
    if len(labels) > 10:
        opening += "..."
    return HuntResult(
        winner=game.winner,
        plies=len(labels),
        opening=opening,
        scoresheet=",".join(labels),
        tag=tag,
    )


def play_opening_vs_normal(
    payload: tuple[tuple[PrefixSpec, ...], int, str, str],
) -> HuntResult:
    """Picklable worker: forced prefix, then Black policy vs node-limited Normal."""
    specs, max_moves, black_kind, tag = payload
    return _play_specs_then_policies(
        specs,
        max_moves=max_moves,
        black_kind=black_kind,
        tag=tag,
    )


def default_max_moves() -> int:
    return DEFAULT_WHITE_DEMO_MAX_MOVES

"""Find where Hard left the teacher book and hunt wins from that prefix."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.infrastructure.rl.hunt_black_wins import (
    PrefixSpec,
    format_numbered_scoresheet,
    format_prefix_spec,
    parse_scoresheet,
    resolve_prefix_action,
)
from app.infrastructure.rl.white_demonstrations import TeacherBook, actions_match
from quoridor.domain.game import Game
from quoridor.domain.state import Color


@dataclass(frozen=True)
class Divergence:
    hard_color: Color
    ply: int
    prefix: tuple[PrefixSpec, ...]
    reason: str


def first_divergence(text: str, book: TeacherBook, hard_color: Color) -> Divergence | None:
    """Prefix is every ply before Hard's first off-book (or uncovered) action."""
    specs = tuple(parse_scoresheet(text))
    if not specs:
        return None
    game = Game.from_initial()
    prefix: list[PrefixSpec] = []
    for ply, spec in enumerate(specs, start=1):
        action = resolve_prefix_action(game.state, spec)
        if action is None:
            return Divergence(hard_color, ply, tuple(prefix), "illegal")
        if game.state.current_player == hard_color:
            teacher = book.action_for(game.state, hard_color)
            if teacher is None:
                return Divergence(hard_color, ply, tuple(prefix), "uncovered")
            if not actions_match(teacher, action):
                return Divergence(hard_color, ply, tuple(prefix), "off_book")
        game.play(action)
        prefix.append(spec)
        if game.is_finished:
            break
    return None


def unique_divergences(
    texts: list[str],
    book: TeacherBook,
    hard_color: Color,
    *,
    limit: int,
) -> list[tuple[Divergence, int]]:
    counts: dict[tuple[PrefixSpec, ...], tuple[Divergence, int]] = {}
    for text in texts:
        found = first_divergence(text, book, hard_color)
        if found is None:
            continue
        prev = counts.get(found.prefix)
        if prev is None:
            counts[found.prefix] = (found, 1)
        else:
            counts[found.prefix] = (prev[0], prev[1] + 1)
    ranked = sorted(counts.values(), key=lambda item: (-item[1], item[0].ply, item[0].reason))
    return ranked[: max(0, limit)]


def format_prefix_csv(prefix: tuple[PrefixSpec, ...]) -> str:
    return ",".join(format_prefix_spec(spec) for spec in prefix)


def prefix_from_csv(text: str) -> tuple[PrefixSpec, ...]:
    raw = text.strip()
    if not raw:
        return ()
    if "scoresheet=" not in raw:
        raw = f"scoresheet={raw}"
    return tuple(parse_scoresheet(raw))


def write_hunt_scoresheet(
    path: Path,
    *,
    tag: str,
    winner: str | None,
    plies: int,
    opening: str,
    scoresheet: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    numbered = format_numbered_scoresheet(parse_scoresheet(f"scoresheet={scoresheet}"))
    path.write_text(
        "\n".join(
            [
                f"tag={tag}",
                f"winner={winner}",
                f"plies={plies}",
                f"opening={opening}",
                f"scoresheet={scoresheet}",
                "",
                numbered,
                "",
            ]
        ),
        encoding="utf-8",
    )

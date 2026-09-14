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
from app.infrastructure.rl.white_demonstrations import (
    DemoTransition,
    TeacherBook,
    actions_match,
    greedy_race_action,
    record_demo_transition,
)
from quoridor.domain.game import Game
from quoridor.domain.state import Color, position_key


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


def hard_color_from_sheet(path: Path, text: str) -> Color | None:
    for line in text.splitlines():
        if line.startswith("tag=eval-"):
            parts = line.split("=", 1)[1].split("-")
            if len(parts) >= 2 and parts[1] in ("black", "white"):
                return parts[1]  # type: ignore[return-value]
    name = path.name
    if "_white_" in name:
        return "white"
    if "_black_" in name:
        return "black"
    return None


def is_hard_loss(path: Path, text: str, hard: Color) -> bool:
    winner = None
    for line in text.splitlines():
        if line.startswith("winner="):
            winner = line.split("=", 1)[1].strip() or None
            break
    if winner is None:
        return "loss" in path.name
    return winner != hard


def _loss_roots(source: str | Path) -> list[Path]:
    if isinstance(source, Path) or "," not in str(source):
        return [Path(source)]
    return [Path(part.strip()) for part in str(source).split(",") if part.strip()]


def load_hard_loss_texts(loss_dir: Path | str, hard: Color) -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()
    for root in _loss_roots(loss_dir):
        if not root.exists():
            raise FileNotFoundError(f"Hard-loss scoresheets not found: {root}")
        paths = sorted(root.glob("*.txt")) if root.is_dir() else [root]
        for path in paths:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            color = hard_color_from_sheet(path, text)
            if color != hard:
                continue
            if not is_hard_loss(path, text, hard):
                continue
            if text in seen:
                continue
            seen.add(text)
            texts.append(text)
    return texts


def first_uncovered_race_error(
    text: str, book: TeacherBook, hard_color: Color
) -> Divergence | None:
    """Prefix is every ply before Hard's first uncovered move that is not greedy-race."""
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
                race = greedy_race_action(game.state, hard_color)
                if not actions_match(race, action):
                    return Divergence(hard_color, ply, tuple(prefix), "race_error")
        game.play(action)
        prefix.append(spec)
        if game.is_finished:
            break
    return None


def unique_race_errors(
    texts: list[str],
    book: TeacherBook,
    hard_color: Color,
    *,
    limit: int,
) -> list[tuple[Divergence, int]]:
    counts: dict[tuple[PrefixSpec, ...], tuple[Divergence, int]] = {}
    for text in texts:
        found = first_uncovered_race_error(text, book, hard_color)
        if found is None:
            continue
        prev = counts.get(found.prefix)
        if prev is None:
            counts[found.prefix] = (found, 1)
        else:
            counts[found.prefix] = (prev[0], prev[1] + 1)
    ranked = sorted(counts.values(), key=lambda item: (-item[1], item[0].ply, item[0].reason))
    return ranked[: max(0, limit)]


def uncovered_hold_focus_transitions(
    texts: list[str],
    book: TeacherBook,
    hard_color: Color,
    *,
    repeat: int,
) -> list[DemoTransition]:
    """Repeat greedy-race at the first uncovered ply (even if the loss matched)."""
    if repeat <= 0:
        return []
    seen: set[tuple] = set()
    out: list[DemoTransition] = []
    for text in texts:
        found = first_divergence(text, book, hard_color)
        if found is None or found.reason != "uncovered":
            continue
        game = Game.from_initial()
        for spec in found.prefix:
            action = resolve_prefix_action(game.state, spec)
            if action is None:
                break
            game.play(action)
        else:
            race = greedy_race_action(game.state, hard_color)
            key = (hard_color, position_key(game.state))
            if key in seen:
                continue
            seen.add(key)
            transition = record_demo_transition(game.state, race, hard_color)
            out.extend([transition] * repeat)
    return out


def load_sheet_texts(source: Path | str, *, stem_contains: str | None = None) -> list[str]:
    texts: list[str] = []
    for root in _loss_roots(source):
        if not root.exists():
            raise FileNotFoundError(f"scoresheets not found: {root}")
        paths = sorted(root.glob("*.txt")) if root.is_dir() else [root]
        for path in paths:
            if not path.is_file():
                continue
            if stem_contains and stem_contains not in path.name:
                continue
            texts.append(path.read_text(encoding="utf-8"))
    return texts


def uncovered_sheet_follow_transitions(
    texts: list[str],
    book: TeacherBook,
    hard_color: Color,
    *,
    max_actions: int,
    repeat: int,
) -> list[DemoTransition]:
    """Label the sheet's own Hard moves after the first uncovered ply."""
    if repeat <= 0 or max_actions <= 0:
        return []
    seen: set[tuple] = set()
    out: list[DemoTransition] = []
    for text in texts:
        found = first_divergence(text, book, hard_color)
        if found is None or found.reason != "uncovered":
            continue
        game = Game.from_initial()
        for spec in found.prefix:
            action = resolve_prefix_action(game.state, spec)
            if action is None:
                break
            game.play(action)
        else:
            remaining = parse_scoresheet(text)[len(found.prefix) :]
            recorded = 0
            for spec in remaining:
                action = resolve_prefix_action(game.state, spec)
                if action is None:
                    break
                if game.state.current_player == hard_color:
                    if recorded >= max_actions:
                        break
                    key = (hard_color, position_key(game.state))
                    if key not in seen:
                        seen.add(key)
                        out.extend(
                            [record_demo_transition(game.state, action, hard_color)]
                            * repeat
                        )
                    recorded += 1
                game.play(action)
                if game.is_finished:
                    break
    return out


def uncovered_race_focus_transitions(
    texts: list[str],
    book: TeacherBook,
    hard_color: Color,
    *,
    repeat: int,
) -> list[DemoTransition]:
    """Repeat the greedy-race action at each unique uncovered race error."""
    if repeat <= 0:
        return []
    seen: set[tuple] = set()
    out: list[DemoTransition] = []
    for text in texts:
        found = first_uncovered_race_error(text, book, hard_color)
        if found is None or found.reason != "race_error":
            continue
        game = Game.from_initial()
        for spec in found.prefix:
            action = resolve_prefix_action(game.state, spec)
            if action is None:
                break
            game.play(action)
        else:
            race = greedy_race_action(game.state, hard_color)
            key = (hard_color, position_key(game.state))
            if key in seen:
                continue
            seen.add(key)
            transition = record_demo_transition(game.state, race, hard_color)
            out.extend([transition] * repeat)
    return out


def teacher_focus_transitions(
    texts: list[str],
    book: TeacherBook,
    hard_color: Color,
    *,
    repeat: int,
) -> list[DemoTransition]:
    """Repeat the teacher action at each unique off-book divergence."""
    if repeat <= 0:
        return []
    seen: set[tuple] = set()
    out: list[DemoTransition] = []
    for text in texts:
        found = first_divergence(text, book, hard_color)
        if found is None or found.reason != "off_book":
            continue
        game = Game.from_initial()
        for spec in found.prefix:
            action = resolve_prefix_action(game.state, spec)
            if action is None:
                break
            game.play(action)
        else:
            teacher = book.action_for(game.state, hard_color)
            if teacher is None:
                continue
            key = (hard_color, position_key(game.state))
            if key in seen:
                continue
            seen.add(key)
            transition = record_demo_transition(game.state, teacher, hard_color)
            out.extend([transition] * repeat)
    return out


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

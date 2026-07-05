"""Debug helpers for self-play: repetition (千日手) detection and move logging."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TextIO

from app.infrastructure.ai.evaluation import EASY_EVAL_CONFIG, NORMAL_EVAL_CONFIG, StateEvaluator
from quoridor.domain.actions import Action, Move
from quoridor.domain.state import GOAL_ROW, Color, QuoridorState, position_key
from quoridor.pathfinding import _bfs_distance, distances
from quoridor.pawn_moves import step_destinations_from
from quoridor.rules import apply_action, get_legal_actions


def format_action(action: Action) -> str:
    if isinstance(action, Move):
        return f"move {action.direction} -> {action.to}"
    return f"wall {action.orientation} @ (r{action.row},c{action.col})"


def format_state(state: QuoridorState) -> str:
    return (
        f"white={state.white} black={state.black} turn={state.current_player} "
        f"walls_rem(w/b)={state.white_walls_remaining}/{state.black_walls_remaining}"
    )


def wall_coords(state: QuoridorState) -> str:
    h = [(r, c) for r in range(8) for c in range(8) if state.horizontal_walls[r][c]]
    v = [(r, c) for r in range(8) for c in range(8) if state.vertical_walls[r][c]]
    return f"H={h} V={v}"


def format_board_ascii(state: QuoridorState) -> str:
    """9x9 board with walls. W=white pawn, B=black pawn. '-' horizontal wall, '|' vertical wall."""
    from quoridor.board_topology import is_horizontal_wall, is_vertical_wall

    lines: list[str] = []
    header = "    " + " ".join(f"c{c}" for c in range(9))
    lines.append(header)
    for r in range(9):
        cells: list[str] = []
        for c in range(9):
            if state.white == (r, c):
                mark = "W"
            elif state.black == (r, c):
                mark = "B"
            else:
                mark = "."
            cells.append(mark)
            if c < 8:
                cells.append("|" if is_vertical_wall(state, r, c) or is_vertical_wall(state, r - 1, c) else " ")
        lines.append(f"r{r}  " + "".join(cells))
        if r < 8:
            under: list[str] = ["    "]
            for c in range(9):
                blocked = is_horizontal_wall(state, r, c) or is_horizontal_wall(state, r, c - 1)
                under.append("--" if blocked else "  ")
            lines.append("".join(under))
    return "\n".join(lines)


def move_eval_breakdown(state: QuoridorState, difficulty: str) -> list[str]:
    """For the side to move, list every legal pawn move with resulting eval + distances."""
    color = state.current_player
    goal_row = GOAL_ROW[color]
    evaluator = _evaluator_for(difficulty)
    cur_dist = _bfs_distance(state, color, for_evaluation=True)

    rows: list[tuple[float, str]] = []
    legal = get_legal_actions(state)
    for action in legal:
        if not isinstance(action, Move) or action.to is None:
            continue
        child = apply_action(state, action)
        score = evaluator.evaluate(child, color, before=state, remaining_depth=0)
        mine_strict = _bfs_distance(child, color, for_evaluation=False)
        mine_ghost = _bfs_distance(child, color, for_evaluation=True)
        toward = "" if action.to[0] == state.pawn(color)[0] else (
            " [toward-goal]" if abs(action.to[0] - goal_row) < abs(state.pawn(color)[0] - goal_row) else " [away-goal]"
        )
        rows.append(
            (
                score,
                f"    {format_action(action):24s} eval={score:+.4f} "
                f"dist_ghost={mine_ghost} dist_strict={mine_strict}{toward}",
            )
        )
    rows.sort(key=lambda x: -x[0])

    out = [f"  Side-to-move {color}({difficulty}) current dist_ghost={cur_dist}; legal pawn moves by eval:"]
    out.extend(line for _, line in rows)

    # What does shortest-path want? best neighbor that reduces strict distance.
    best_step = None
    best_d = None
    for dest in step_destinations_from(state, color, state.pawn(color)):
        d = _bfs_distance(state.with_pawn(color, dest), color, for_evaluation=False)
        if d is not None and (best_d is None or d < best_d):
            best_d = d
            best_step = dest
    out.append(f"  Shortest-path first step -> {best_step} (resulting strict dist={best_d})")
    return out


_EVALUATORS: dict[str, StateEvaluator] = {
    "easy": StateEvaluator(EASY_EVAL_CONFIG),
    "very_easy": StateEvaluator(EASY_EVAL_CONFIG),
    "normal": StateEvaluator(NORMAL_EVAL_CONFIG),
}


def _evaluator_for(difficulty: str) -> StateEvaluator:
    return _EVALUATORS.get(difficulty, StateEvaluator(NORMAL_EVAL_CONFIG))


def evaluate_for_player(state: QuoridorState, color: Color, difficulty: str) -> float:
    return _evaluator_for(difficulty).evaluate(state, color)


def distance_summary(state: QuoridorState, color: Color) -> str:
    dw, db = distances(state)
    mine = dw if color == "white" else db
    enemy = db if color == "white" else dw
    return f"dist(mine={mine},enemy={enemy})"


@dataclass(frozen=True)
class MoveRecord:
    move_num: int
    color: Color
    difficulty: str
    action: Action
    eval_score: float
    dist_summary: str
    state_summary: str


class RepetitionDebugger:
    def __init__(
        self,
        game_id: int,
        *,
        difficulty_a: str,
        difficulty_b: str,
        colors: tuple[Color, Color],
        stream: TextIO | None = None,
    ) -> None:
        self.game_id = game_id
        self.difficulty_a = difficulty_a
        self.difficulty_b = difficulty_b
        self.colors = colors
        self._stream = stream or sys.stderr
        self._move_log: list[MoveRecord] = []
        self._position_visits: dict[tuple[object, ...], list[int]] = {}
        self.repetition_events = 0
        self._analysis_done = False

    def _difficulty_for(self, color: Color) -> str:
        return self.difficulty_a if color == self.colors[0] else self.difficulty_b

    def on_before_move(self, state: QuoridorState, move_num: int) -> None:
        key = position_key(state)
        visits = self._position_visits.get(key, [])
        if visits:
            self._log_repetition(state, move_num, visits[-1])
        self._position_visits.setdefault(key, []).append(move_num)

    def on_after_move(
        self,
        state_before: QuoridorState,
        color: Color,
        action: Action,
        move_num: int,
    ) -> None:
        difficulty = self._difficulty_for(color)
        record = MoveRecord(
            move_num=move_num,
            color=color,
            difficulty=difficulty,
            action=action,
            eval_score=evaluate_for_player(state_before, color, difficulty),
            dist_summary=distance_summary(state_before, color),
            state_summary=format_state(state_before),
        )
        self._move_log.append(record)

    def _log_repetition(self, state: QuoridorState, move_num: int, first_move_num: int) -> None:
        self.repetition_events += 1
        color = state.current_player
        difficulty = self._difficulty_for(color)
        current_eval = evaluate_for_player(state, color, difficulty)
        cycle = [r for r in self._move_log if first_move_num <= r.move_num < move_num]

        lines = [
            "",
            f"=== Game {self.game_id}: 千日手 (position repeat) at move {move_num} ===",
            f"Position: {format_state(state)}",
            f"Current player: {color} ({difficulty}) "
            f"eval={current_eval:+.4f} {distance_summary(state, color)}",
            f"Same position last seen before move {first_move_num} "
            f"(cycle length: {len(cycle)} half-moves)",
        ]
        if cycle:
            lines.append("Cycle moves:")
            for rec in cycle:
                lines.append(
                    f"  #{rec.move_num:3d} {rec.color}({rec.difficulty:8s}) "
                    f"eval={rec.eval_score:+.4f} {rec.dist_summary}  {format_action(rec.action)}"
                )
        else:
            lines.append("(no intervening moves recorded)")

        if not self._analysis_done:
            self._analysis_done = True
            lines.append("")
            lines.append("Walls: " + wall_coords(state))
            lines.append("Board:")
            lines.append(format_board_ascii(state))
            lines.append("")
            lines.extend(move_eval_breakdown(state, difficulty))
        lines.append("")
        self._stream.write("\n".join(lines) + "\n")
        self._stream.flush()

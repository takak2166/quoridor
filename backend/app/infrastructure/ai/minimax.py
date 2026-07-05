from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.infrastructure.ai.evaluation import EASY_EVAL_CONFIG, NORMAL_EVAL_CONFIG, EvalConfig, StateEvaluator
from app.infrastructure.ai.search_actions import (
    opponent,
    prioritize_wall_actions,
    search_actions,
    split_legal_actions,
)
from quoridor.domain.actions import Action, Move, encode
from quoridor.domain.state import Color, QuoridorState, position_key
from quoridor.pathfinding import SimpleDistanceCache, distances
from quoridor.rules import apply_action, get_legal_actions


@dataclass(frozen=True)
class MinimaxConfig:
    time_budget_ms: int = 400
    max_nodes: int = 500
    primary_depth: int = 3
    fallback_depth: int = 2
    max_wall_candidates: int = 12
    two_phase_search: bool = True
    moves_only: bool = False


class MinimaxEngine:
    def __init__(
        self,
        config: MinimaxConfig,
        eval_config: EvalConfig | None = None,
        *,
        cache: SimpleDistanceCache | None = None,
        evaluator: StateEvaluator | None = None,
    ) -> None:
        self.config = config
        self._eval_config = eval_config or NORMAL_EVAL_CONFIG
        self._cache = cache if cache is not None else SimpleDistanceCache()
        self._evaluator = evaluator if evaluator is not None else StateEvaluator(self._eval_config)
        self._visited = 0
        self._deadline = 0.0
        self._aborted = False
        self._anti_loop_enabled = (
            self._eval_config.elapsed_move_penalty > 0 or self._eval_config.revisit_penalty > 0
        )

    def select_move(self, state: QuoridorState, color: Color) -> Action:
        legal = get_legal_actions(state, dist_cache=self._cache)
        if not legal:
            raise RuntimeError("no legal moves")
        if len(legal) == 1:
            return legal[0]

        self._deadline = time.perf_counter() + self.config.time_budget_ms / 1000
        best_action, aborted = self._search_depth(state, color, self.config.primary_depth)
        if aborted:
            remaining = self._deadline - time.perf_counter()
            if remaining > 0 and self.config.fallback_depth < self.config.primary_depth:
                self._deadline = time.perf_counter() + remaining
                best_action, aborted = self._search_depth(state, color, self.config.fallback_depth)
        return best_action

    def _search_actions_for_node(self, state: QuoridorState) -> list[Action]:
        legal = get_legal_actions(state, dist_cache=self._cache)
        if self.config.moves_only:
            moves, _ = split_legal_actions(legal)
            return list(moves)
        return search_actions(state, legal, self._cache, self.config.max_wall_candidates)

    def _search_depth(self, state: QuoridorState, color: Color, depth: int) -> tuple[Action, bool]:
        self._visited = 0
        self._aborted = False
        legal = get_legal_actions(state, dist_cache=self._cache)
        moves, walls = split_legal_actions(legal)
        move_actions: list[Action] = list(moves)
        if self.config.moves_only:
            action, _, aborted = self._best_root_action(state, color, depth, move_actions)
            return action or move_actions[0], aborted

        top_walls = prioritize_wall_actions(state, walls, self._cache, self.config.max_wall_candidates)
        top_wall_actions: list[Action] = list(top_walls)

        sidestep = self._lane_match_move(state, color)
        if sidestep is not None and self._needs_lane_match_sidestep(state, color):
            return sidestep, self._aborted

        if self.config.two_phase_search:
            move_action, move_score, move_aborted = self._best_root_action(
                state, color, depth, move_actions
            )
            wall_action, wall_score, wall_aborted = self._best_root_action(
                state, color, depth, top_wall_actions
            )

            chosen = self._choose_move_or_wall(
                state,
                color,
                move_action,
                move_score,
                wall_action,
                wall_score,
            )
            if chosen is not None:
                return chosen, self._aborted or move_aborted or wall_aborted
            fallback = top_wall_actions[0] if top_wall_actions else legal[0]
            return fallback, self._aborted

        combined: list[Action] = [*move_actions, *top_wall_actions]
        action, _, aborted = self._best_root_action(state, color, depth, combined)
        return action or combined[0], aborted

    def _best_root_action(
        self,
        state: QuoridorState,
        color: Color,
        depth: int,
        actions: list[Action],
    ) -> tuple[Action | None, float, bool]:
        if not actions:
            return None, -math.inf, self._aborted
        best_score = -math.inf
        best_action = actions[0]
        alpha, beta = -math.inf, math.inf
        root_path = None
        if self._anti_loop_enabled:
            root_path = {position_key(state): 1}
        for action in actions:
            if self._timed_out() or self._visited >= self.config.max_nodes:
                self._aborted = True
                return best_action, best_score, True
            child = apply_action(state, action)
            score = self._minimax(
                child,
                depth - 1,
                color,
                alpha,
                beta,
                before=state,
                path_counts=root_path,
                plies_from_root=1,
            )
            if score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, score)
        return best_action, best_score, self._aborted

    def _path_distances(self, state: QuoridorState, color: Color) -> tuple[int | None, int | None]:
        dw, db = distances(state, self._cache)
        enemy = opponent(color)
        enemy_dist = dw if enemy == "white" else db
        my_dist = db if color == "black" else dw
        return my_dist, enemy_dist

    def _is_behind_in_race(self, state: QuoridorState, color: Color) -> bool:
        my_dist, enemy_dist = self._path_distances(state, color)
        if my_dist is None or enemy_dist is None:
            return False
        return my_dist > enemy_dist

    def _choose_move_or_wall(
        self,
        state: QuoridorState,
        color: Color,
        move_action: Action | None,
        move_score: float,
        wall_action: Action | None,
        wall_score: float,
    ) -> Action | None:
        if wall_action is None:
            return move_action
        if move_action is None:
            return wall_action
        if self._is_behind_in_race(state, color):
            return wall_action
        if wall_score > move_score:
            return wall_action
        if move_score > wall_score:
            return move_action
        return random.choice([move_action, wall_action])

    def _needs_lane_match_sidestep(self, state: QuoridorState, color: Color) -> bool:
        walls_used = 10 - (
            state.white_walls_remaining if color == "white" else state.black_walls_remaining
        )
        if walls_used != 1:
            return False
        enemy = opponent(color)
        return state.pawn(color)[1] != state.pawn(enemy)[1]

    def _lane_match_move(self, state: QuoridorState, color: Color) -> Move | None:
        moves, _ = split_legal_actions(get_legal_actions(state, dist_cache=self._cache))
        enemy_col = state.pawn(opponent(color))[1]
        lane_moves = [move for move in moves if move.to is not None and move.to[1] == enemy_col]
        if not lane_moves:
            return None
        return max(
            lane_moves,
            key=lambda move: self._evaluate(
                apply_action(state, move),
                color,
                before=state,
                remaining_depth=0,
            ),
        )

    def _minimax(
        self,
        state: QuoridorState,
        depth: int,
        root_color: Color,
        alpha: float,
        beta: float,
        *,
        before: QuoridorState | None = None,
        path_counts: dict[tuple[object, ...], int] | None = None,
        plies_from_root: int = 0,
    ) -> float:
        self._visited += 1
        position_revisits = 0
        extended_path = path_counts
        if self._anti_loop_enabled:
            key = position_key(state)
            position_revisits = (path_counts or {}).get(key, 0)
            extended_path = dict(path_counts or {})
            extended_path[key] = extended_path.get(key, 0) + 1

        if self._visited >= self.config.max_nodes or self._timed_out():
            self._aborted = True
            return self._evaluate(
                state,
                root_color,
                before=before,
                remaining_depth=depth,
                plies_from_root=plies_from_root,
                position_revisits=position_revisits,
            )
        if depth == 0:
            return self._evaluate(
                state,
                root_color,
                before=before,
                remaining_depth=0,
                plies_from_root=plies_from_root,
                position_revisits=position_revisits,
            )

        legal = self._search_actions_for_node(state)
        if not legal:
            return self._evaluate(
                state,
                root_color,
                before=before,
                remaining_depth=depth,
                plies_from_root=plies_from_root,
                position_revisits=position_revisits,
            )

        child_plies = plies_from_root + 1 if self._anti_loop_enabled else 0
        maximizing = state.current_player == root_color
        if maximizing:
            value = -math.inf
            for action in legal:
                child = apply_action(state, action)
                value = max(
                    value,
                    self._minimax(
                        child,
                        depth - 1,
                        root_color,
                        alpha,
                        beta,
                        before=state,
                        path_counts=extended_path,
                        plies_from_root=child_plies,
                    ),
                )
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value
        value = math.inf
        for action in legal:
            child = apply_action(state, action)
            value = min(
                value,
                self._minimax(
                    child,
                    depth - 1,
                    root_color,
                    alpha,
                    beta,
                    before=state,
                    path_counts=extended_path,
                    plies_from_root=child_plies,
                ),
            )
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value

    def _evaluate(
        self,
        state: QuoridorState,
        color: Color,
        *,
        before: QuoridorState | None = None,
        remaining_depth: int = 0,
        plies_from_root: int = 0,
        position_revisits: int = 0,
    ) -> float:
        return self._evaluator.evaluate(
            state,
            color,
            before=before,
            remaining_depth=remaining_depth,
            plies_from_root=plies_from_root,
            position_revisits=position_revisits,
        )

    def _timed_out(self) -> bool:
        return time.perf_counter() >= self._deadline


@dataclass
class EasyMinimaxPolicy:
    config: MinimaxConfig

    def __post_init__(self) -> None:
        self._engine_config = MinimaxConfig(
            time_budget_ms=self.config.time_budget_ms,
            max_nodes=self.config.max_nodes,
            max_wall_candidates=self.config.max_wall_candidates,
            two_phase_search=self.config.two_phase_search,
            primary_depth=1,
            fallback_depth=1,
        )
        self._evaluator = StateEvaluator(EASY_EVAL_CONFIG)

    def _new_engine(self) -> MinimaxEngine:
        return MinimaxEngine(self._engine_config, eval_config=EASY_EVAL_CONFIG)

    def select_move(self, state: QuoridorState, color: Color) -> Action:
        return self._new_engine().select_move(state, color)

    def action_prior(self, state: QuoridorState, color: Color) -> NDArray[np.floating]:
        from quoridor.domain.actions import NUM_ACTIONS

        legal = get_legal_actions(state)
        prior = np.zeros(NUM_ACTIONS, dtype=np.float64)
        for a in legal:
            prior[encode(a)] = 1.0
        if prior.sum() > 0:
            prior /= prior.sum()
        return prior

    def value(self, state: QuoridorState, color: Color) -> float:
        return self._evaluator.evaluate(state, color)


@dataclass
class VeryEasyMinimaxPolicy:
    config: MinimaxConfig

    def __post_init__(self) -> None:
        self._engine_config = MinimaxConfig(
            time_budget_ms=self.config.time_budget_ms,
            max_nodes=self.config.max_nodes,
            max_wall_candidates=0,
            two_phase_search=False,
            moves_only=True,
            primary_depth=1,
            fallback_depth=1,
        )
        self._evaluator = StateEvaluator(EASY_EVAL_CONFIG)

    def _new_engine(self) -> MinimaxEngine:
        return MinimaxEngine(self._engine_config, eval_config=EASY_EVAL_CONFIG)

    def select_move(self, state: QuoridorState, color: Color) -> Action:
        return self._new_engine().select_move(state, color)

    def action_prior(self, state: QuoridorState, color: Color) -> NDArray[np.floating]:
        from quoridor.domain.actions import NUM_ACTIONS

        legal = get_legal_actions(state)
        prior = np.zeros(NUM_ACTIONS, dtype=np.float64)
        for a in legal:
            prior[encode(a)] = 1.0
        if prior.sum() > 0:
            prior /= prior.sum()
        return prior

    def value(self, state: QuoridorState, color: Color) -> float:
        return self._evaluator.evaluate(state, color)


@dataclass
class NormalMinimaxPolicy:
    config: MinimaxConfig

    def __post_init__(self) -> None:
        self._evaluator = StateEvaluator(NORMAL_EVAL_CONFIG)
        # Reuse distance/eval caches across per-move engine instances so depth
        # search stays fast; search counters remain isolated per select_move.
        self._cache = SimpleDistanceCache()

    def select_move(self, state: QuoridorState, color: Color) -> Action:
        engine = MinimaxEngine(
            self.config,
            eval_config=NORMAL_EVAL_CONFIG,
            cache=self._cache,
            evaluator=self._evaluator,
        )
        legal = get_legal_actions(state, dist_cache=self._cache)
        if not legal:
            raise RuntimeError("no legal moves")
        if len(legal) == 1:
            return legal[0]

        global_deadline = time.perf_counter() + self.config.time_budget_ms / 1000
        primary_ms = max(1, int(self.config.time_budget_ms * 0.65))
        engine._deadline = time.perf_counter() + primary_ms / 1000
        best_action, aborted = engine._search_depth(state, color, self.config.primary_depth)
        if aborted:
            remaining = engine._deadline - time.perf_counter()
            if remaining > 0 and self.config.fallback_depth < self.config.primary_depth:
                engine._deadline = global_deadline
                best_action, aborted = engine._search_depth(state, color, self.config.fallback_depth)

        if aborted and self.config.primary_depth > 1:
            remaining = global_deadline - time.perf_counter()
            if remaining <= 0:
                return best_action
            behind = engine._is_behind_in_race(state, color)
            shallow_engine = MinimaxEngine(
                MinimaxConfig(
                    time_budget_ms=max(1, int(remaining * 1000)),
                    max_nodes=self.config.max_nodes,
                    max_wall_candidates=self.config.max_wall_candidates,
                    two_phase_search=self.config.two_phase_search,
                    moves_only=not behind,
                    primary_depth=1,
                    fallback_depth=1,
                ),
                NORMAL_EVAL_CONFIG,
                cache=self._cache,
                evaluator=self._evaluator,
            )
            shallow_engine._deadline = global_deadline
            best_action, _ = shallow_engine._search_depth(state, color, 1)
        return best_action

    def action_prior(self, state: QuoridorState, color: Color) -> NDArray[np.floating]:
        from quoridor.domain.actions import NUM_ACTIONS

        legal = get_legal_actions(state)
        prior = np.zeros(NUM_ACTIONS, dtype=np.float64)
        for a in legal:
            prior[encode(a)] = 1.0
        if prior.sum() > 0:
            prior /= prior.sum()
        return prior

    def value(self, state: QuoridorState, color: Color) -> float:
        return self._evaluator.evaluate(state, color)

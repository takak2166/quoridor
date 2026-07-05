from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quoridor.domain.state import Color, QuoridorState, empty_walls
from quoridor.pathfinding import SimpleDistanceCache, distances

if TYPE_CHECKING:
    from quoridor.domain.actions import WallSlot

# Backward-compatible defaults (Normal preset).
ENEMY_PATH_BLOCK_WEIGHT = 2.5
SELF_PATH_BLOCK_WEIGHT = 0.75


@dataclass(frozen=True)
class EvalConfig:
    enemy_path_block_weight: float = ENEMY_PATH_BLOCK_WEIGHT
    self_path_block_weight: float = SELF_PATH_BLOCK_WEIGHT
    wall_remaining_weight: float = 0.5
    strategic_path_block_weight: float = 0.4
    # Indexed by remaining minimax depth; last entry is reused for deeper nodes.
    wall_bonus_depth_scales: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)
    # Raw score units (before /16 normalization). Normal uses these to break search loops.
    elapsed_move_penalty: float = 0.0
    revisit_penalty: float = 0.0


EASY_EVAL_CONFIG = EvalConfig(
    enemy_path_block_weight=1.0,
    self_path_block_weight=0.75,
    wall_remaining_weight=0.35,
    strategic_path_block_weight=0.5,
    wall_bonus_depth_scales=(0.35,),
)

NORMAL_EVAL_CONFIG = EvalConfig(
    enemy_path_block_weight=2.5,
    self_path_block_weight=0.5,
    wall_remaining_weight=0.25,
    strategic_path_block_weight=0.6,
    wall_bonus_depth_scales=(1.0, 0.75, 0.5, 0.25),
    elapsed_move_penalty=0.08,
    revisit_penalty=3.0,
)


class StateEvaluator:
    def __init__(self, config: EvalConfig | None = None) -> None:
        self._config = config or NORMAL_EVAL_CONFIG
        self._cache = SimpleDistanceCache()
        self._baseline_cache: dict[tuple[tuple[int, int], tuple[int, int]], tuple[int | None, int | None]] = {}
        self._baseline_cache_limit = 2_048

    def _depth_scale(self, remaining_depth: int) -> float:
        scales = self._config.wall_bonus_depth_scales
        if not scales:
            return 1.0
        if remaining_depth < len(scales):
            return scales[remaining_depth]
        return scales[-1]

    @staticmethod
    def _wall_was_placed(before: QuoridorState, state: QuoridorState) -> bool:
        return (
            before.white == state.white
            and before.black == state.black
            and (
                before.horizontal_walls != state.horizontal_walls
                or before.vertical_walls != state.vertical_walls
            )
        )

    def _baseline_distances(self, state: QuoridorState) -> tuple[int | None, int | None]:
        key = (state.white, state.black)
        cached = self._baseline_cache.get(key)
        if cached is not None:
            return cached
        baseline = QuoridorState(
            white=state.white,
            black=state.black,
            white_walls_remaining=state.white_walls_remaining,
            black_walls_remaining=state.black_walls_remaining,
            horizontal_walls=empty_walls(),
            vertical_walls=empty_walls(),
            current_player=state.current_player,
        )
        result = distances(baseline)
        if len(self._baseline_cache) >= self._baseline_cache_limit:
            for stale in list(self._baseline_cache.keys())[: self._baseline_cache_limit // 2]:
                self._baseline_cache.pop(stale, None)
        self._baseline_cache[key] = result
        return result

    @staticmethod
    def _new_wall(before: QuoridorState, state: QuoridorState) -> WallSlot | None:
        from quoridor.domain.actions import WallSlot

        for row in range(8):
            for col in range(8):
                if (
                    not before.horizontal_walls[row][col]
                    and state.horizontal_walls[row][col]
                ):
                    return WallSlot(orientation="horizontal", row=row, col=col)
                if not before.vertical_walls[row][col] and state.vertical_walls[row][col]:
                    return WallSlot(orientation="vertical", row=row, col=col)
        return None

    def _path_block_adjustment(
        self,
        *,
        enemy: int,
        mine: int,
        enemy_reference: int | None,
        mine_reference: int | None,
        depth_scale: float,
    ) -> float:
        adjustment = 0.0
        if enemy_reference is not None:
            adjustment += (
                depth_scale
                * self._config.enemy_path_block_weight
                * max(0, enemy - enemy_reference)
            )
        if mine_reference is not None:
            adjustment -= (
                depth_scale
                * self._config.self_path_block_weight
                * max(0, mine - mine_reference)
            )
        return adjustment

    def evaluate(
        self,
        state: QuoridorState,
        color: Color,
        *,
        before: QuoridorState | None = None,
        remaining_depth: int = 0,
        plies_from_root: int = 0,
        position_revisits: int = 0,
    ) -> float:
        dw, db = distances(state, self._cache)
        if dw is None:
            return -1.0 if color == "white" else 1.0
        if db is None:
            return 1.0 if color == "white" else -1.0

        enemy = db if color == "white" else dw
        mine = dw if color == "white" else db
        depth_scale = self._depth_scale(remaining_depth)

        score = float(enemy - mine)
        if before is not None and self._wall_was_placed(before, state):
            dw_before, db_before = distances(before, self._cache)
            enemy_before = db_before if color == "white" else dw_before
            mine_before = dw_before if color == "white" else db_before
            score += self._path_block_adjustment(
                enemy=enemy,
                mine=mine,
                enemy_reference=enemy_before,
                mine_reference=mine_before,
                depth_scale=depth_scale,
            )
            if enemy == enemy_before:
                new_wall = self._new_wall(before, state)
                if new_wall is not None:
                    from app.infrastructure.ai.search_actions import shortest_path_edges_blocked

                    blocked = shortest_path_edges_blocked(before, new_wall, self._cache)
                    score += (
                        depth_scale
                        * self._config.strategic_path_block_weight
                        * blocked
                    )
        elif before is None:
            dw0, db0 = self._baseline_distances(state)
            enemy_baseline = db0 if color == "white" else dw0
            mine_baseline = dw0 if color == "white" else db0
            score += self._path_block_adjustment(
                enemy=enemy,
                mine=mine,
                enemy_reference=enemy_baseline,
                mine_reference=mine_baseline,
                depth_scale=depth_scale,
            )

        wr = state.white_walls_remaining if color == "white" else state.black_walls_remaining
        er = state.black_walls_remaining if color == "white" else state.white_walls_remaining
        score += self._config.wall_remaining_weight * (wr - er)

        if plies_from_root > 0 and self._config.elapsed_move_penalty > 0:
            score -= self._config.elapsed_move_penalty * plies_from_root
        if position_revisits > 0 and self._config.revisit_penalty > 0:
            score -= self._config.revisit_penalty * position_revisits

        return max(-1.0, min(1.0, score / 16.0))

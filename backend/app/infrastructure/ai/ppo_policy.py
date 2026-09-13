from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from app.config import settings
from app.infrastructure.ai.action_mask import (
    estimated_agent_plies,
    exclude_previous_action,
    filter_repeat_pawn_cells,
    legal_action_mask_agent_frame,
    legal_actions_for_policy,
    policy_wall_candidate_limit,
)
from app.infrastructure.ai.evaluation import StateEvaluator
from app.infrastructure.ai.ppo_loader import ppo_model_store
from app.infrastructure.rl.action_resolution import resolve_agent_index_to_action
from app.mappers.observation_mapper import to_observation
from quoridor.agent_frame import encode_for_viewer
from quoridor.domain.actions import NUM_ACTIONS, Action, Move, WallSlot, encode
from quoridor.domain.state import Color, QuoridorState, position_key
from quoridor.pathfinding import SimpleDistanceCache

logger = logging.getLogger(__name__)


@dataclass
class PPOPolicy:
    model_path: str
    _evaluator: StateEvaluator = field(default_factory=StateEvaluator)
    _warned_missing: bool = False
    _dist_cache: SimpleDistanceCache = field(default_factory=SimpleDistanceCache)
    _pawn_path: dict[Color, list[tuple[int, int]]] = field(default_factory=dict)
    _select_count: dict[Color, int] = field(default_factory=dict)
    _last_action_by_pos: dict[tuple[Color, tuple], Action] = field(default_factory=dict)

    def is_available(self) -> bool:
        return ppo_model_store.is_available(self.model_path)

    def select_move(self, state: QuoridorState, color: Color) -> Action:
        legal = legal_actions_for_policy(
            state,
            self._dist_cache,
            policy_wall_candidate_limit(settings.ppo_max_wall_candidates),
            color=color,
            opening_wall_free_plies=settings.ppo_opening_wall_free_plies,
        )
        legal = self._apply_loop_filters(state, color, legal)
        if not legal:
            raise RuntimeError("no legal moves")
        if len(legal) == 1:
            chosen = legal[0]
            self._remember_action(state, color, chosen)
            return chosen

        if self.is_available():
            try:
                chosen = self._select_with_model(state, color, legal)
                chosen = self._break_stall(state, color, legal, chosen)
                self._remember_action(state, color, chosen)
                return chosen
            except Exception:
                logger.exception("PPO inference failed for %s", self.model_path)
                self._record_fallback()
        else:
            self._warn_missing_once()
            self._record_fallback()

        from_pos = state.pawn(color)
        chosen = self._select_with_prior(self._uniform_prior(legal, from_pos), legal, from_pos)
        chosen = self._break_stall(state, color, legal, chosen)
        self._remember_action(state, color, chosen)
        return chosen

    def action_prior(self, state: QuoridorState, color: Color) -> NDArray[np.floating]:
        legal = legal_actions_for_policy(
            state,
            self._dist_cache,
            policy_wall_candidate_limit(settings.ppo_max_wall_candidates),
            color=color,
            opening_wall_free_plies=settings.ppo_opening_wall_free_plies,
        )
        legal = self._apply_loop_filters(state, color, legal)
        prior = np.zeros(NUM_ACTIONS, dtype=np.float64)
        if not legal:
            return prior

        from_pos = state.pawn(color)
        if self.is_available():
            try:
                return self._prior_with_model(state, color, legal)
            except Exception:
                logger.exception("PPO prior failed for %s", self.model_path)
                self._record_fallback()
        else:
            self._warn_missing_once()
            self._record_fallback()

        for action in legal:
            prior[encode(action, from_pos=from_pos)] = 1.0
        prior /= prior.sum()
        return prior

    def value(self, state: QuoridorState, color: Color) -> float:
        if self.is_available():
            try:
                return self._value_with_model(state, color)
            except Exception:
                logger.exception("PPO value failed for %s", self.model_path)
                self._record_fallback()
        else:
            self._warn_missing_once()
            self._record_fallback()
        return self._evaluator.evaluate(state, color)

    def _select_with_model(self, state: QuoridorState, color: Color, legal: list[Action]) -> Action:
        model = ppo_model_store.get(self.model_path)
        from_pos = state.pawn(color)
        obs = to_observation(state, color)
        mask = legal_action_mask_agent_frame(legal, color, from_pos=from_pos)
        action_idx, _ = model.predict(obs, action_masks=mask, deterministic=True)
        try:
            return resolve_agent_index_to_action(
                int(action_idx),
                legal,
                color,
                from_pos=from_pos,
            )
        except ValueError:
            return self._select_with_prior(
                self._prior_with_model(state, color, legal),
                legal,
                from_pos,
            )

    def _prior_with_model(
        self,
        state: QuoridorState,
        color: Color,
        legal: list[Action],
    ) -> NDArray[np.floating]:
        import torch

        model = ppo_model_store.get(self.model_path)
        from_pos = state.pawn(color)
        obs = to_observation(state, color)
        mask = legal_action_mask_agent_frame(legal, color, from_pos=from_pos)
        obs_tensor = torch.as_tensor(obs, device=model.device).unsqueeze(0)
        mask_tensor = torch.as_tensor(mask, device=model.device).unsqueeze(0)
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_tensor, action_masks=mask_tensor)
            probs = dist.distribution.probs.detach().cpu().numpy().reshape(-1)
        # Expose prior in absolute-delta action space for search consumers.
        prior = np.zeros(NUM_ACTIONS, dtype=np.float64)
        for action in legal:
            framed_idx = encode_for_viewer(action, from_pos, color)
            prior[encode(action, from_pos=from_pos)] = float(probs[framed_idx])
        if prior.sum() <= 0:
            for action in legal:
                prior[encode(action, from_pos=from_pos)] = 1.0
        prior /= prior.sum()
        return prior

    def _value_with_model(self, state: QuoridorState, color: Color) -> float:
        import torch

        model = ppo_model_store.get(self.model_path)
        obs = to_observation(state, color)
        obs_tensor = torch.as_tensor(obs, device=model.device).unsqueeze(0)
        with torch.no_grad():
            value = model.policy.predict_values(obs_tensor)
        return float(value.detach().cpu().numpy().reshape(-1)[0])

    def _reset_color_loop_state(self, color: Color, pawn: tuple[int, int]) -> None:
        self._pawn_path[color] = [pawn]
        self._select_count[color] = 0
        self._last_action_by_pos = {
            key: action for key, action in self._last_action_by_pos.items() if key[0] != color
        }

    def _apply_loop_filters(
        self,
        state: QuoridorState,
        color: Color,
        legal: list[Action],
    ) -> list[Action]:
        if estimated_agent_plies(state, color) == 0:
            self._reset_color_loop_state(color, state.pawn(color))
        path = self._pawn_path.setdefault(color, [state.pawn(color)])
        if self._select_count.get(color, 0) < settings.ppo_loop_filter_plies:
            return legal
        legal = filter_repeat_pawn_cells(
            legal,
            path,
            max_visits=settings.ppo_repeat_pawn_max_visits,
        )
        pos_key = (color, position_key(state))
        return exclude_previous_action(legal, self._last_action_by_pos.get(pos_key))

    def _break_stall(
        self,
        state: QuoridorState,
        color: Color,
        legal: list[Action],
        chosen: Action,
    ) -> Action:
        if not isinstance(chosen, WallSlot):
            return chosen
        if self._select_count.get(color, 0) < settings.ppo_stall_plies:
            return chosen
        path = self._pawn_path.get(color, [])
        recent = path[-8:] if path else []
        if recent and len(set(recent)) > 2:
            return chosen
        from app.infrastructure.rl.white_demonstrations import greedy_race_action

        race = greedy_race_action(state, color, self._dist_cache)
        if race in legal and isinstance(race, Move):
            return race
        return chosen

    def _remember_action(self, state: QuoridorState, color: Color, action: Action) -> None:
        self._last_action_by_pos[(color, position_key(state))] = action
        self._select_count[color] = self._select_count.get(color, 0) + 1
        if isinstance(action, Move) and action.to is not None:
            self._pawn_path.setdefault(color, [state.pawn(color)]).append(action.to)

    def _select_with_prior(
        self,
        prior: NDArray[np.floating],
        legal: list[Action],
        from_pos: tuple[int, int],
    ) -> Action:
        best_action: Action | None = None
        best_p = -1.0
        for action in legal:
            p = float(prior[encode(action, from_pos=from_pos)])
            if p > best_p:
                best_p = p
                best_action = action
        if best_action is not None:
            return best_action
        return random.choice(legal)

    def _uniform_prior(
        self,
        legal: list[Action],
        from_pos: tuple[int, int],
    ) -> NDArray[np.floating]:
        prior = np.zeros(NUM_ACTIONS, dtype=np.float64)
        for action in legal:
            prior[encode(action, from_pos=from_pos)] = 1.0
        prior /= prior.sum()
        return prior

    def _warn_missing_once(self) -> None:
        if self._warned_missing:
            return
        self._warned_missing = True
        logger.warning("PPO model unavailable (%s); using heuristic fallback", self.model_path)

    def _record_fallback(self) -> None:
        from app.middleware.metrics import metrics_store

        metrics_store.record_ai_fallback()

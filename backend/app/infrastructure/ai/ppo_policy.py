from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from app.config import settings
from app.infrastructure.ai.action_mask import legal_action_mask_agent_frame, legal_actions_for_policy
from app.infrastructure.ai.evaluation import StateEvaluator
from app.infrastructure.ai.ppo_loader import ppo_model_store
from app.infrastructure.rl.action_resolution import resolve_agent_index_to_action
from app.mappers.observation_mapper import to_observation
from quoridor.agent_frame import action_to_agent_frame
from quoridor.domain.actions import NUM_ACTIONS, Action, encode
from quoridor.domain.state import Color, QuoridorState
from quoridor.pathfinding import SimpleDistanceCache

logger = logging.getLogger(__name__)


@dataclass
class PPOPolicy:
    model_path: str
    _evaluator: StateEvaluator = field(default_factory=StateEvaluator)
    _warned_missing: bool = False
    _dist_cache: SimpleDistanceCache = field(default_factory=SimpleDistanceCache)

    def is_available(self) -> bool:
        return ppo_model_store.is_available(self.model_path)

    def select_move(self, state: QuoridorState, color: Color) -> Action:
        legal = legal_actions_for_policy(
            state,
            self._dist_cache,
            settings.ppo_max_wall_candidates,
        )
        if not legal:
            raise RuntimeError("no legal moves")
        if len(legal) == 1:
            return legal[0]

        if self.is_available():
            try:
                return self._select_with_model(state, color, legal)
            except Exception:
                logger.exception("PPO inference failed for %s", self.model_path)
                self._record_fallback()
        else:
            self._warn_missing_once()
            self._record_fallback()

        return self._select_with_prior(self._uniform_prior(legal), legal)

    def action_prior(self, state: QuoridorState, color: Color) -> NDArray[np.floating]:
        legal = legal_actions_for_policy(
            state,
            self._dist_cache,
            settings.ppo_max_wall_candidates,
        )
        prior = np.zeros(NUM_ACTIONS, dtype=np.float64)
        if not legal:
            return prior

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
            prior[encode(action)] = 1.0
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
        obs = to_observation(state, color)
        mask = legal_action_mask_agent_frame(legal, color)
        action_idx, _ = model.predict(obs, action_masks=mask, deterministic=True)
        try:
            return resolve_agent_index_to_action(int(action_idx), legal, color)
        except ValueError:
            return self._select_with_prior(self._prior_with_model(state, color, legal), legal)

    def _prior_with_model(
        self,
        state: QuoridorState,
        color: Color,
        legal: list[Action],
    ) -> NDArray[np.floating]:
        import torch

        model = ppo_model_store.get(self.model_path)
        obs = to_observation(state, color)
        mask = legal_action_mask_agent_frame(legal, color)
        obs_tensor = torch.as_tensor(obs, device=model.device).unsqueeze(0)
        mask_tensor = torch.as_tensor(mask, device=model.device).unsqueeze(0)
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_tensor, action_masks=mask_tensor)
            probs = dist.distribution.probs.detach().cpu().numpy().reshape(-1)
        # Expose prior in absolute action space for search consumers.
        prior = np.zeros(NUM_ACTIONS, dtype=np.float64)
        for action in legal:
            framed_idx = encode(action_to_agent_frame(action, color))
            prior[encode(action)] = float(probs[framed_idx])
        if prior.sum() <= 0:
            for action in legal:
                prior[encode(action)] = 1.0
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

    def _select_with_prior(self, prior: NDArray[np.floating], legal: list[Action]) -> Action:
        best_action: Action | None = None
        best_p = -1.0
        for action in legal:
            p = float(prior[encode(action)])
            if p > best_p:
                best_p = p
                best_action = action
        if best_action is not None:
            return best_action
        return random.choice(legal)

    def _uniform_prior(self, legal: list[Action]) -> NDArray[np.floating]:
        prior = np.zeros(NUM_ACTIONS, dtype=np.float64)
        for action in legal:
            prior[encode(action)] = 1.0
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

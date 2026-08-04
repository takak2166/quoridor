"""VecEnv wrapper that logs predict-time mask vs sampled action mismatches."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

from quoridor.domain.actions import decode

logger = logging.getLogger(__name__)


class MaskDiagnosticVecEnv(VecEnvWrapper):
    """Compare action masks at sample time with actions about to be stepped."""

    def __init__(self, venv: VecEnv) -> None:
        super().__init__(venv)
        self._last_predict_masks: np.ndarray | None = None
        self._mismatch_count = 0

    def reset(self) -> VecEnvObs:
        return self.venv.reset()

    def step_async(self, actions: np.ndarray) -> None:
        masks = get_action_masks(self.venv)
        self._last_predict_masks = masks
        acts = np.asarray(actions).reshape(-1)
        for env_idx, action in enumerate(acts):
            action_i = int(action)
            in_predict = bool(masks[env_idx, action_i])
            if in_predict:
                continue
            self._mismatch_count += 1
            decoded = decode(action_i)
            logger.error(
                "PREDICT_MASK_MISS env=%s action=%s decoded=%s mask_sum=%s "
                "mismatch_count=%s (sampled outside get_action_masks)",
                env_idx,
                action_i,
                decoded,
                int(masks[env_idx].sum()),
                self._mismatch_count,
            )
        self.venv.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        return self.venv.step_wait()

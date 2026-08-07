"""Smoke eval must serialize MaskablePPO.predict across worker threads."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from app.infrastructure.rl.train_ppo import _predict_action


class _CountingModel:
    """Minimal stand-in that detects overlapping predict calls."""

    def __init__(self) -> None:
        self._active = 0
        self.max_active = 0
        self._guard = threading.Lock()

    def predict(self, obs, action_masks=None, deterministic=True):  # noqa: ANN001
        with self._guard:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        time.sleep(0.05)
        with self._guard:
            self._active -= 1
        # Always return a legal opening-style move index.
        return np.array([67]), None


def test_predict_action_lock_serializes_shared_model() -> None:
    model = _CountingModel()
    lock = threading.Lock()
    obs = np.zeros(4, dtype=np.float32)
    mask = np.zeros(209, dtype=bool)
    mask[67] = True

    def worker() -> int:
        return _predict_action(model, obs, mask, lock=lock)  # type: ignore[arg-type]

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: worker(), range(8)))

    assert results == [67] * 8
    assert model.max_active == 1


def test_predict_action_without_lock_can_overlap() -> None:
    model = _CountingModel()
    obs = np.zeros(4, dtype=np.float32)
    mask = np.zeros(209, dtype=bool)
    mask[67] = True

    def worker() -> int:
        return _predict_action(model, obs, mask)  # type: ignore[arg-type]

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda _: worker(), range(8)))

    assert model.max_active > 1

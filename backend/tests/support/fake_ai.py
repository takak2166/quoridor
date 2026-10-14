from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from app.schemas import Color
from quoridor.domain.actions import Action, Move
from quoridor.domain.state import QuoridorState
from quoridor.rules import get_legal_actions


@dataclass
class AiConcurrencyTracker:
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    current: int = 0
    peak: int = 0

    def enter(self) -> None:
        with self._lock:
            self.current += 1
            if self.current > self.peak:
                self.peak = self.current

    def leave(self) -> None:
        with self._lock:
            self.current -= 1


@dataclass
class FakeAiProvider:
    delay_ms: float = 0
    fail: bool = False
    exception_to_raise: Exception | None = None
    illegal_move: bool = False
    concurrency_tracker: AiConcurrencyTracker | None = None

    def select_move(self, state: QuoridorState, color: Color) -> Action:
        if self.concurrency_tracker is not None:
            self.concurrency_tracker.enter()
        try:
            return self._select_move_body(state, color)
        finally:
            if self.concurrency_tracker is not None:
                self.concurrency_tracker.leave()

    def _select_move_body(self, state: QuoridorState, color: Color) -> Action:
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000)
        if self.fail:
            raise TimeoutError("AI failure")
        if self.exception_to_raise is not None:
            raise self.exception_to_raise
        if self.illegal_move:
            return Move(direction="up", to=(99, 99))
        legal = get_legal_actions(state)
        if not legal:
            raise RuntimeError("no legal moves")
        return random.choice(legal)

    def action_prior(self, state: QuoridorState, color: Color) -> NDArray[np.floating]:
        from quoridor.domain.actions import NUM_ACTIONS, encode

        legal = get_legal_actions(state)
        prior = np.zeros(NUM_ACTIONS, dtype=np.float64)
        if not legal:
            return prior
        from_pos = state.pawn(color)
        for action in legal:
            prior[encode(action, from_pos=from_pos)] = 1.0
        prior /= prior.sum()
        return prior

    def value(self, state: QuoridorState, color: Color) -> float:
        return 0.0

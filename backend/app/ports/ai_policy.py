from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from quoridor.domain.actions import Action
from quoridor.domain.state import Color, QuoridorState


class AiPolicy(Protocol):
    def select_move(self, state: QuoridorState, color: Color) -> Action: ...

    def action_prior(self, state: QuoridorState, color: Color) -> NDArray[np.floating]:
        """Return action priors for ``color`` to move.

        When used as MCTS ``prior_fn``, must return the same vector for the same
        ``(state, color)`` throughout one ``mcts_search`` run. See ``PriorFn`` in
        ``app.infrastructure.ai.mcts`` for the full contract (session-scoped policy
        state is fine if priors stay stable for unchanged ``(state, color)``).
        """

    def value(self, state: QuoridorState, color: Color) -> float: ...

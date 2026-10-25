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

        Implementations used as MCTS ``prior_fn`` must return the same vector for the
        same ``(state, color)`` while a search tree is built (no hidden state).
        """

    def value(self, state: QuoridorState, color: Color) -> float: ...

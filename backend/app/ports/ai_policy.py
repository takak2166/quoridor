from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from quoridor.domain.actions import Action
from quoridor.domain.state import Color, QuoridorState


class AiPolicy(Protocol):
    def select_move(self, state: QuoridorState, color: Color) -> Action: ...

    def action_prior(self, state: QuoridorState, color: Color) -> NDArray[np.floating]: ...

    def value(self, state: QuoridorState, color: Color) -> float: ...

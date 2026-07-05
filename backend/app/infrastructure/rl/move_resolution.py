from __future__ import annotations

import random
from typing import Protocol

from quoridor.domain.actions import Direction, Move
from quoridor.domain.state import QuoridorState
from quoridor.rules import move_destinations, resolve_move


class RandomSource(Protocol):
    def choice(self, seq: list[Move]) -> Move: ...


def resolve_ambiguous_move(
    state: QuoridorState,
    direction: Direction,
    rng: RandomSource | None = None,
) -> Move:
    dests = move_destinations(state, direction)
    if len(dests) <= 1:
        return resolve_move(state, direction, None)
    candidates = [Move(direction=direction, to=dest) for dest in sorted(dests, key=lambda p: (p[0], p[1]))]
    source: RandomSource = rng if rng is not None else random
    return source.choice(candidates)

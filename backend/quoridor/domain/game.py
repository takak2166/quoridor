from __future__ import annotations

from dataclasses import dataclass

from quoridor.domain.actions import Action
from quoridor.domain.state import Color, QuoridorState
from quoridor.rules import PlayResult, apply_action, check_winner, is_action_legal


@dataclass(frozen=True)
class GameSnapshot:
    state: QuoridorState
    winner: Color | None


@dataclass
class Game:
    state: QuoridorState
    winner: Color | None = None

    @classmethod
    def from_initial(cls) -> Game:
        from quoridor.domain.state import initial_state

        return cls(state=initial_state())

    def snapshot(self) -> GameSnapshot:
        return GameSnapshot(state=self.state.copy(), winner=self.winner)

    def restore(self, snap: GameSnapshot) -> None:
        self.state = snap.state.copy()
        self.winner = snap.winner

    def play(self, action: Action) -> PlayResult:
        if self.winner is not None:
            raise ValueError("game over")
        if not is_action_legal(self.state, action):
            raise ValueError("illegal move")
        new_state = apply_action(self.state, action)
        winner = check_winner(new_state)
        self.state = new_state
        self.winner = winner
        return PlayResult(state=new_state, winner=winner)

    @property
    def is_finished(self) -> bool:
        return self.winner is not None

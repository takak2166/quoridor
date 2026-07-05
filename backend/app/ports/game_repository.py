from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import datetime
from typing import Protocol

from app.schemas import Difficulty
from quoridor.domain.game import Game
from quoridor.domain.state import Color


class SessionRecordLike(Protocol):
    game_id: str
    expires_at: datetime
    human_color: Color
    difficulty: Difficulty
    game: Game
    lock: AbstractContextManager[None]


class GameRepositoryPort(Protocol):
    def create(self, human_color: Color, difficulty: Difficulty) -> tuple[SessionRecordLike, str]: ...

    def get(self, game_id: str) -> SessionRecordLike | None: ...

    def verify(self, game_id: str, token: str) -> SessionRecordLike | None: ...

    def delete(self, game_id: str) -> bool: ...

    def active_count(self) -> int: ...

from __future__ import annotations

import hashlib
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from app.config import settings
from app.schemas import Difficulty
from quoridor.domain.game import Game
from quoridor.domain.state import Color


@dataclass
class GameSession:
    session_token_hash: str
    expires_at: datetime
    human_color: Color
    difficulty: Difficulty
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class GameSessionRecord:
    game_id: str
    session: GameSession
    game: Game
    lock: threading.Lock = field(default_factory=threading.Lock, compare=False)

    @property
    def session_token_hash(self) -> str:
        return self.session.session_token_hash

    @property
    def expires_at(self) -> datetime:
        return self.session.expires_at

    @property
    def human_color(self) -> Color:
        return self.session.human_color

    @property
    def difficulty(self) -> Difficulty:
        return self.session.difficulty

    @property
    def created_at(self) -> datetime:
        return self.session.created_at


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class GameRepository:
    def __init__(self) -> None:
        self._sessions: dict[str, GameSessionRecord] = {}
        self._lock = threading.Lock()
        self._gc_thread = threading.Thread(target=self._gc_loop, daemon=True)
        self._gc_thread.start()

    def _gc_loop(self) -> None:
        while True:
            time.sleep(60)
            self.gc_expired()

    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def gc_expired(self) -> None:
        now = datetime.now(UTC)
        with self._lock:
            expired = [gid for gid, rec in self._sessions.items() if rec.expires_at < now]
            for gid in expired:
                del self._sessions[gid]

    def create(
        self,
        human_color: Color,
        difficulty: Difficulty,
    ) -> tuple[GameSessionRecord, str]:
        with self._lock:
            if len(self._sessions) >= settings.max_concurrent_games:
                raise CapacityExceededError()
            token = secrets.token_urlsafe(32)
            game_id = str(uuid.uuid4())
            now = datetime.now(UTC)
            record = GameSessionRecord(
                game_id=game_id,
                session=GameSession(
                    session_token_hash=hash_token(token),
                    expires_at=now + timedelta(minutes=settings.session_ttl_minutes),
                    human_color=human_color,
                    difficulty=difficulty,
                    created_at=now,
                ),
                game=Game.from_initial(),
            )
            self._sessions[game_id] = record
            return record, token

    def get(self, game_id: str) -> GameSessionRecord | None:
        with self._lock:
            rec = self._sessions.get(game_id)
            if rec is None:
                return None
            if rec.expires_at < datetime.now(UTC):
                del self._sessions[game_id]
                return None
            return rec

    def verify(self, game_id: str, token: str) -> GameSessionRecord | None:
        rec = self.get(game_id)
        if rec is None:
            return None
        if not secrets.compare_digest(rec.session_token_hash, hash_token(token)):
            return None
        return rec

    def delete(self, game_id: str) -> bool:
        with self._lock:
            if game_id in self._sessions:
                del self._sessions[game_id]
                return True
            return False


class CapacityExceededError(Exception):
    pass


game_repository = GameRepository()

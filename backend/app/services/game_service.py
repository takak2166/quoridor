from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from app.config import settings
from app.infrastructure.persistence.game_repository import (
    CapacityExceededError,
    game_repository,
)
from app.mappers.game_mapper import action_to_dto, dto_to_action, state_to_dto
from app.ports.ai_policy import AiPolicy
from app.ports.game_repository import GameRepositoryPort, SessionRecordLike
from app.ports.metrics import MetricsPort
from app.schemas import (
    Color,
    CreateGameRequest,
    CreateGameResponse,
    Difficulty,
    ErrorCode,
    GameDetailResponse,
    GameStatus,
    MoveResponseDTO,
    PlayMoveRequest,
    PlayMoveResponse,
    Turn,
)
from quoridor.domain.actions import Move
from quoridor.rules import AmbiguousMoveError, get_legal_actions, resolve_move


class GameServiceError(Exception):
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        http_status: int,
        details: dict[str, object] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.http_status = http_status
        self.details = details
        super().__init__(message)


AiFactory = Callable[[Difficulty], AiPolicy]


@dataclass
class _NullMetrics:
    def record_ai_inference(self, duration_s: float) -> None:
        _ = duration_s

    def record_ai_failure(self) -> None:
        return None

    def record_ai_fallback(self) -> None:
        return None

    def record_ai_rollback(self) -> None:
        return None


class GameService:
    def __init__(
        self,
        ai_factory: AiFactory | None = None,
        repository: GameRepositoryPort | None = None,
        metrics: MetricsPort | None = None,
        ai_limiter: threading.Semaphore | None = None,
    ) -> None:
        self._ai_factory = ai_factory
        self._repository = cast(
            GameRepositoryPort,
            repository if repository is not None else game_repository,
        )
        self._metrics = cast(MetricsPort, metrics if metrics is not None else _NullMetrics())
        self._ai_limiter = ai_limiter if ai_limiter is not None else threading.Semaphore(settings.ai_concurrent_limit)

    def _cpu_color(self, human_color: Color) -> Color:
        return "black" if human_color == "white" else "white"

    def _turn(self, record: SessionRecordLike) -> Turn | None:
        if record.game.is_finished:
            return None
        if record.game.state.current_player == record.human_color:
            return "human"
        return "cpu"

    def _status(self, record: SessionRecordLike) -> GameStatus:
        return "finished" if record.game.is_finished else "active"

    def create_game(self, req: CreateGameRequest) -> tuple[CreateGameResponse, str]:
        try:
            record = self._repository.create(req.human_color, req.difficulty)
            session_record, token = record
        except CapacityExceededError as e:
            raise GameServiceError(
                "GAME_CAPACITY_EXCEEDED",
                "Server at capacity",
                503,
            ) from e

        cpu_move: MoveResponseDTO | None = None
        if req.human_color == "white":
            with session_record.lock:
                snap = session_record.game.snapshot()
                try:
                    cpu_move = self._play_cpu(session_record)
                except GameServiceError as e:
                    if e.code == "AI_FAILURE":
                        self._metrics.record_ai_rollback()
                        session_record.game.restore(snap)
                    raise

        return (
            CreateGameResponse(
                game_id=session_record.game_id,
                session_token=token,
                human_color=session_record.human_color,
                difficulty=session_record.difficulty,
                status=self._status(session_record),
                turn=self._turn(session_record) or "human",
                winner=session_record.game.winner,
                cpu_move=cpu_move,
                state=state_to_dto(session_record.game.state),
            ),
            token,
        )

    def get_game(self, game_id: str) -> GameDetailResponse:
        record = self._repository.get(game_id)
        if record is None:
            raise GameServiceError("SESSION_EXPIRED", "Game not found", 403)
        return GameDetailResponse(
            game_id=record.game_id,
            human_color=record.human_color,
            difficulty=record.difficulty,
            status=self._status(record),
            turn=self._turn(record),
            winner=record.game.winner,
            state=state_to_dto(record.game.state),
        )

    def play_move(self, game_id: str, req: PlayMoveRequest) -> PlayMoveResponse:
        record = self._repository.get(game_id)
        if record is None:
            raise GameServiceError("SESSION_EXPIRED", "Game not found", 403)
        with record.lock:
            if record.game.is_finished:
                raise GameServiceError("GAME_OVER", "Game is finished", 409)
            if record.game.state.current_player != record.human_color:
                raise GameServiceError("WRONG_TURN", "Not human turn", 409)

            human_action = dto_to_action(req.action)
            if isinstance(human_action, Move):
                try:
                    human_action = resolve_move(
                        record.game.state,
                        human_action.direction,
                        human_action.to,
                    )
                except ValueError as e:
                    reason = "ambiguous" if isinstance(e, AmbiguousMoveError) else "illegal"
                    raise GameServiceError("ILLEGAL_MOVE", str(e), 400, {"reason": reason}) from e
            snap = record.game.snapshot()
            try:
                record.game.play(human_action)
            except ValueError as e:
                raise GameServiceError("ILLEGAL_MOVE", str(e), 400, {"reason": "illegal"}) from e

            human_move_dto = action_to_dto(human_action)
            cpu_move: MoveResponseDTO | None = None
            if not record.game.is_finished and record.game.state.current_player != record.human_color:
                try:
                    cpu_move = self._play_cpu(record)
                except GameServiceError as e:
                    if e.code == "AI_FAILURE":
                        self._metrics.record_ai_rollback()
                    record.game.restore(snap)
                    raise

            return PlayMoveResponse(
                human_move=human_move_dto,
                cpu_move=cpu_move,
                status=self._status(record),
                turn=self._turn(record),
                winner=record.game.winner,
                state=state_to_dto(record.game.state),
            )

    def list_legal_actions(self, game_id: str) -> list[MoveResponseDTO]:
        record = self._repository.get(game_id)
        if record is None:
            raise GameServiceError("SESSION_EXPIRED", "Game not found", 403)
        with record.lock:
            legal = get_legal_actions(record.game.state)
            return [action_to_dto(action) for action in legal]

    def delete_game(self, game_id: str) -> None:
        if not self._repository.delete(game_id):
            raise GameServiceError("SESSION_EXPIRED", "Game not found", 403)

    def _play_cpu(self, record: SessionRecordLike) -> MoveResponseDTO | None:
        if record.game.is_finished:
            return None
        cpu_color = self._cpu_color(record.human_color)
        if record.game.state.current_player != cpu_color:
            return None
        ai = self._get_ai(record.difficulty)
        acquired = self._ai_limiter.acquire(timeout=5.0)
        if not acquired:
            raise GameServiceError(
                "AI_FAILURE",
                "AI concurrency limit reached",
                503,
                {"reason": "concurrency_limit", "difficulty": record.difficulty},
            )
        try:
            start = time.perf_counter()
            action = ai.select_move(record.game.state, cpu_color)
            if isinstance(action, Move):
                if action.to is None:
                    from app.infrastructure.rl.move_resolution import resolve_ambiguous_move

                    action = resolve_ambiguous_move(record.game.state, action.direction)
                else:
                    action = resolve_move(record.game.state, action.direction, action.to)
            self._metrics.record_ai_inference(time.perf_counter() - start)
            record.game.play(action)
            return action_to_dto(action)
        except GameServiceError:
            raise
        except Exception as e:
            self._metrics.record_ai_failure()
            raise GameServiceError(
                "AI_FAILURE",
                "AI inference failed",
                503,
                {"reason": type(e).__name__, "difficulty": record.difficulty},
            ) from e
        finally:
            self._ai_limiter.release()

    def _get_ai(self, difficulty: Difficulty) -> AiPolicy:
        if self._ai_factory is not None:
            return self._ai_factory(difficulty)
        from app.infrastructure.ai.factory import ai_for_difficulty

        return ai_for_difficulty(difficulty)

    def verify_session(self, game_id: str, token: str) -> None:
        if self._repository.verify(game_id, token) is None:
            raise GameServiceError("SESSION_EXPIRED", "Invalid session", 403)


def get_game_service() -> GameService:
    from app.middleware.metrics import metrics_store

    return GameService(
        repository=cast(GameRepositoryPort, game_repository),
        metrics=metrics_store,
        ai_limiter=threading.Semaphore(settings.ai_concurrent_limit),
    )

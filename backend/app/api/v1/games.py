from __future__ import annotations

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import Response

from app.schemas import (
    CreateGameRequest,
    CreateGameResponse,
    ErrorCode,
    ErrorDetail,
    ErrorResponse,
    GameDetailResponse,
    LegalActionsResponse,
    PlayMoveRequest,
    PlayMoveResponse,
)
from app.services.game_service import GameService, GameServiceError, get_game_service

router = APIRouter(tags=["games"])


def verify_session(
    game_id: str,
    x_quoridor_session: str = Header(..., alias="X-Quoridor-Session"),
    svc: GameService = Depends(get_game_service),
) -> str:
    try:
        svc.verify_session(game_id, x_quoridor_session)
    except GameServiceError as e:
        raise _http_error(e.code, e.message, e.http_status, e.details) from e
    return x_quoridor_session


def _http_error(
    code: ErrorCode,
    message: str,
    status: int,
    details: dict[str, object] | None = None,
) -> Exception:
    from fastapi import HTTPException

    body = ErrorResponse(error=ErrorDetail(code=code, message=message, details=details))
    return HTTPException(status_code=status, detail=body.model_dump())


@router.post("/games", status_code=201, response_model=CreateGameResponse)
def create_game(
    req: CreateGameRequest,
    request: Request,
    svc: GameService = Depends(get_game_service),
) -> CreateGameResponse:
    request.state.difficulty = req.difficulty
    try:
        resp, _ = svc.create_game(req)
        request.state.game_id = resp.game_id
        return resp
    except GameServiceError as e:
        request.state.error_code = e.code
        raise _http_error(e.code, e.message, e.http_status, e.details) from e


@router.get("/games/{game_id}", response_model=GameDetailResponse)
def get_game(
    game_id: str,
    _session: str = Depends(verify_session),
    svc: GameService = Depends(get_game_service),
) -> GameDetailResponse:
    try:
        return svc.get_game(game_id)
    except GameServiceError as e:
        raise _http_error(e.code, e.message, e.http_status, e.details) from e


@router.post("/games/{game_id}/moves", response_model=PlayMoveResponse)
def play_move(
    game_id: str,
    req: PlayMoveRequest,
    request: Request,
    _session: str = Depends(verify_session),
    svc: GameService = Depends(get_game_service),
) -> PlayMoveResponse:
    request.state.game_id = game_id
    try:
        return svc.play_move(game_id, req)
    except GameServiceError as e:
        request.state.error_code = e.code
        raise _http_error(e.code, e.message, e.http_status, e.details) from e


@router.get("/games/{game_id}/legal-actions", response_model=LegalActionsResponse)
def get_legal_actions(
    game_id: str,
    _session: str = Depends(verify_session),
    svc: GameService = Depends(get_game_service),
) -> LegalActionsResponse:
    try:
        return LegalActionsResponse(actions=svc.list_legal_actions(game_id))
    except GameServiceError as e:
        raise _http_error(e.code, e.message, e.http_status, e.details) from e


@router.delete("/games/{game_id}", status_code=204)
def delete_game(
    game_id: str,
    _session: str = Depends(verify_session),
    svc: GameService = Depends(get_game_service),
) -> Response:
    try:
        svc.delete_game(game_id)
        return Response(status_code=204)
    except GameServiceError as e:
        raise _http_error(e.code, e.message, e.http_status, e.details) from e

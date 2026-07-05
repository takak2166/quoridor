from __future__ import annotations

from app.schemas import (
    BoardPositionDTO,
    GameStateDTO,
    MoveActionResponse,
    MoveDTO,
    PlayerStateDTO,
    WallActionDTO,
    WallPositionDTO,
)
from quoridor.domain.actions import Action, Move, WallSlot
from quoridor.domain.state import QuoridorState


def state_to_dto(state: QuoridorState) -> GameStateDTO:
    return GameStateDTO(
        white=PlayerStateDTO(
            row=state.white[0],
            col=state.white[1],
            walls_remaining=state.white_walls_remaining,
        ),
        black=PlayerStateDTO(
            row=state.black[0],
            col=state.black[1],
            walls_remaining=state.black_walls_remaining,
        ),
        horizontal_walls=[list(row) for row in state.horizontal_walls],
        vertical_walls=[list(row) for row in state.vertical_walls],
        current_player=state.current_player,
    )


def action_to_dto(action: Action) -> MoveActionResponse | WallActionDTO:
    if isinstance(action, Move):
        if action.to is None:
            raise ValueError("move response requires destination")
        return MoveActionResponse(
            type="move",
            direction=action.direction,
            to=BoardPositionDTO(row=action.to[0], col=action.to[1]),
        )
    return WallActionDTO(
        type="wall",
        orientation=action.orientation,
        position=WallPositionDTO(row=action.row, col=action.col),
    )


def dto_to_action(dto: MoveDTO) -> Action:
    if dto.type == "move":
        to: tuple[int, int] | None = None
        if dto.to is not None:
            to = (dto.to.row, dto.to.col)
        return Move(direction=dto.direction, to=to)
    return WallSlot(
        orientation=dto.orientation,
        row=dto.position.row,
        col=dto.position.col,
    )

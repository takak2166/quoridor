from typing import Literal

from pydantic import BaseModel, Field

Color = Literal["white", "black"]
Difficulty = Literal["very_easy", "easy", "normal", "hard", "expert"]
Direction = Literal["up", "down", "left", "right"]
WallOrientation = Literal["horizontal", "vertical"]
GameStatus = Literal["active", "finished"]
Turn = Literal["human", "cpu"]
ErrorCode = Literal[
    "ILLEGAL_MOVE",
    "WRONG_TURN",
    "GAME_OVER",
    "SESSION_EXPIRED",
    "AI_FAILURE",
    "RATE_LIMITED",
    "GAME_CAPACITY_EXCEEDED",
]

class PlayerStateDTO(BaseModel):
    row: int = Field(ge=0, le=8)
    col: int = Field(ge=0, le=8)
    walls_remaining: int = Field(ge=0, le=10)


class GameStateDTO(BaseModel):
    white: PlayerStateDTO
    black: PlayerStateDTO
    horizontal_walls: list[list[bool]]
    vertical_walls: list[list[bool]]
    current_player: Color


class BoardPositionDTO(BaseModel):
    row: int = Field(ge=0, le=8)
    col: int = Field(ge=0, le=8)


class MoveActionRequest(BaseModel):
    type: Literal["move"] = "move"
    direction: Direction
    to: BoardPositionDTO | None = None


class MoveActionResponse(BaseModel):
    type: Literal["move"] = "move"
    direction: Direction
    to: BoardPositionDTO


MoveActionDTO = MoveActionRequest


class WallPositionDTO(BaseModel):
    row: int = Field(ge=0, le=7)
    col: int = Field(ge=0, le=7)


class WallActionDTO(BaseModel):
    type: Literal["wall"] = "wall"
    orientation: WallOrientation
    position: WallPositionDTO


MoveDTO = MoveActionRequest | WallActionDTO
MoveResponseDTO = MoveActionResponse | WallActionDTO


class CreateGameRequest(BaseModel):
    human_color: Color
    difficulty: Difficulty


class CreateGameResponse(BaseModel):
    game_id: str
    session_token: str
    human_color: Color
    difficulty: Difficulty
    status: GameStatus
    turn: Turn
    winner: Color | None
    cpu_move: MoveResponseDTO | None
    state: GameStateDTO


class GameDetailResponse(BaseModel):
    game_id: str
    human_color: Color
    difficulty: Difficulty
    status: GameStatus
    turn: Turn | None
    winner: Color | None
    state: GameStateDTO


class PlayMoveRequest(BaseModel):
    action: MoveDTO


class PlayMoveResponse(BaseModel):
    human_move: MoveResponseDTO
    cpu_move: MoveResponseDTO | None
    status: GameStatus
    turn: Turn | None
    winner: Color | None
    state: GameStateDTO


class LegalActionsResponse(BaseModel):
    actions: list[MoveResponseDTO]


class ErrorDetail(BaseModel):
    code: ErrorCode
    message: str
    details: dict[str, object] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class ModelHealthDetail(BaseModel):
    file_present: bool
    dependencies_ok: bool
    loadable: bool
    loaded: bool


class EffectiveAiStatus(BaseModel):
    very_easy: str
    easy: str
    normal: str
    hard: str
    expert: str


class HealthReadyResponse(BaseModel):
    status: str
    models: dict[str, ModelHealthDetail]
    effective_ai: EffectiveAiStatus

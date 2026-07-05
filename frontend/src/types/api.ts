export type Color = "white" | "black";
export type Difficulty = "very_easy" | "easy" | "normal" | "hard" | "expert";
export type Direction = "up" | "down" | "left" | "right";
export type WallOrientation = "horizontal" | "vertical";
export type GameStatus = "active" | "finished";
export type Turn = "human" | "cpu";
export type ErrorCode =
  | "ILLEGAL_MOVE"
  | "WRONG_TURN"
  | "GAME_OVER"
  | "SESSION_EXPIRED"
  | "AI_FAILURE"
  | "RATE_LIMITED"
  | "GAME_CAPACITY_EXCEEDED";

export interface PlayerStateDTO {
  row: number;
  col: number;
  walls_remaining: number;
}

export interface GameStateDTO {
  white: PlayerStateDTO;
  black: PlayerStateDTO;
  horizontal_walls: boolean[][];
  vertical_walls: boolean[][];
  current_player: Color;
}

export interface BoardPositionDTO {
  row: number;
  col: number;
}

export interface MoveActionDTO {
  type: "move";
  direction: Direction;
  to?: BoardPositionDTO;
}

export interface MoveActionResponseDTO {
  type: "move";
  direction: Direction;
  to: BoardPositionDTO;
}

export interface WallPositionDTO {
  row: number;
  col: number;
}

export interface WallActionDTO {
  type: "wall";
  orientation: WallOrientation;
  position: WallPositionDTO;
}

export type MoveDTO = MoveActionDTO | WallActionDTO;
export type MoveResponseDTO = MoveActionResponseDTO | WallActionDTO;

export interface CreateGameRequest {
  human_color: Color;
  difficulty: Difficulty;
}

export interface CreateGameResponse {
  game_id: string;
  session_token: string;
  human_color: Color;
  difficulty: Difficulty;
  status: GameStatus;
  turn: Turn;
  winner: Color | null;
  cpu_move: MoveResponseDTO | null;
  state: GameStateDTO;
}

export interface GameDetailResponse {
  game_id: string;
  human_color: Color;
  difficulty: Difficulty;
  status: GameStatus;
  turn: Turn | null;
  winner: Color | null;
  state: GameStateDTO;
}

export interface PlayMoveRequest {
  action: MoveDTO;
}

export interface PlayMoveResponse {
  human_move: MoveResponseDTO;
  cpu_move: MoveResponseDTO | null;
  status: GameStatus;
  turn: Turn | null;
  winner: Color | null;
  state: GameStateDTO;
}

export interface LegalActionsResponse {
  actions: MoveResponseDTO[];
}

export interface ErrorDetail {
  code: ErrorCode;
  message: string;
  details?: Record<string, unknown>;
}

export interface ErrorResponse {
  error: ErrorDetail;
}

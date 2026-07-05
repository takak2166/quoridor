import type { CreateGameResponse, GameStateDTO, PlayMoveResponse } from "../src/types/api";

export function emptyWalls(): boolean[][] {
  return Array.from({ length: 8 }, () => Array(8).fill(false));
}

export function initialState(
  overrides: Partial<{
    blackRow: number;
    blackCol: number;
    whiteRow: number;
    whiteCol: number;
    currentPlayer: "white" | "black";
    blackWalls: number;
    whiteWalls: number;
    horizontalWalls: boolean[][];
    verticalWalls: boolean[][];
  }> = {},
): GameStateDTO {
  return {
    white: {
      row: overrides.whiteRow ?? 8,
      col: overrides.whiteCol ?? 4,
      walls_remaining: overrides.whiteWalls ?? 10,
    },
    black: {
      row: overrides.blackRow ?? 0,
      col: overrides.blackCol ?? 4,
      walls_remaining: overrides.blackWalls ?? 10,
    },
    horizontal_walls: overrides.horizontalWalls ?? emptyWalls(),
    vertical_walls: overrides.verticalWalls ?? emptyWalls(),
    current_player: overrides.currentPlayer ?? "white",
  };
}

export function b4BlockState(): GameStateDTO {
  const horizontalWalls = emptyWalls();
  for (let col = 0; col < 8; col += 1) {
    horizontalWalls[0][col] = true;
  }
  return initialState({
    blackRow: 0,
    blackCol: 4,
    whiteRow: 8,
    whiteCol: 4,
    currentPlayer: "black",
    horizontalWalls,
  });
}

export function createGameResponse(
  overrides: Partial<CreateGameResponse> & { state: GameStateDTO },
): CreateGameResponse {
  return {
    game_id: "e2e-game-id",
    session_token: "e2e-session-token",
    human_color: "black",
    difficulty: "easy",
    status: "active",
    turn: "human",
    winner: null,
    cpu_move: null,
    ...overrides,
  };
}

export function playMoveResponse(
  overrides: Partial<PlayMoveResponse> & { state: GameStateDTO },
): PlayMoveResponse {
  return {
    human_move: { type: "move", direction: "up" },
    cpu_move: null,
    status: "active",
    turn: "human",
    winner: null,
    ...overrides,
  };
}

export function errorBody(code: string, message: string) {
  return JSON.stringify({
    detail: { error: { code, message } },
  });
}

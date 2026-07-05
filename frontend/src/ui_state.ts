import type { Color, GameStateDTO, MoveDTO } from "./types/api";

export type InteractionMode = "move" | "wall-h" | "wall-v";

export interface UiState {
  gameId: string | null;
  sessionToken: string | null;
  humanColor: Color;
  state: GameStateDTO | null;
  status: string;
  turn: string | null;
  winner: Color | null;
  mode: InteractionMode;
  error: string | null;
}

export function initialUiState(humanColor: Color): UiState {
  return {
    gameId: null,
    sessionToken: null,
    humanColor,
    state: null,
    status: "idle",
    turn: null,
    winner: null,
    mode: "move",
    error: null,
  };
}

export function isHumanTurn(ui: UiState): boolean {
  return ui.turn === "human" && ui.status === "active";
}

export function directionFromClick(
  humanColor: Color,
  fromRow: number,
  fromCol: number,
  toRow: number,
  toCol: number,
): MoveDTO | null {
  const dr = toRow - fromRow;
  const dc = toCol - fromCol;
  if (Math.abs(dr) + Math.abs(dc) !== 1) return null;
  if (dc === -1) return { type: "move", direction: "left" };
  if (dc === 1) return { type: "move", direction: "right" };
  if (humanColor === "white") {
    if (dr === -1) return { type: "move", direction: "up" };
    if (dr === 1) return { type: "move", direction: "down" };
  } else {
    if (dr === 1) return { type: "move", direction: "up" };
    if (dr === -1) return { type: "move", direction: "down" };
  }
  return null;
}

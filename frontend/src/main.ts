import { ApiError, createGame, playMove } from "./api";
import { renderBoard } from "./board";
import { actionForTarget, applyHumanAction, isLegalWall, legalMoveTargets } from "./rules";
import type { Color, Difficulty, GameStateDTO, MoveDTO } from "./types/api";
import { initialUiState, isHumanTurn, type UiState } from "./ui_state";

let ui: UiState = initialUiState("black");
let selectedCell: { row: number; col: number } | null = null;
let hoveredWall: { orientation: "horizontal" | "vertical"; row: number; col: number } | null = null;
let pendingRequest = false;

const boardEl = document.getElementById("board") as unknown as SVGSVGElement;
const statusEl = document.getElementById("status")!;
const errorEl = document.getElementById("error")!;
const newGameBtn = document.getElementById("new-game") as HTMLButtonElement;
const difficultyEl = document.getElementById("difficulty") as HTMLSelectElement;
const humanColorEl = document.getElementById("human-color") as HTMLSelectElement;

const PAWN_STYLE: Record<Color, { fill: string; stroke: string }> = {
  white: { fill: "#f8f8f8", stroke: "#333" },
  black: { fill: "#2d2d2d", stroke: "#fff" },
};

function setError(msg: string | null): void {
  errorEl.textContent = msg ?? "";
}

function updateStatus(): void {
  if (!ui.state) {
    statusEl.textContent = "Start a new game";
    return;
  }
  const parts = [
    `Turn: ${ui.turn ?? "—"}`,
    `Status: ${ui.status}`,
    ui.winner ? `Winner: ${ui.winner}` : "",
    `White walls: ${ui.state.white.walls_remaining}`,
    `Black walls: ${ui.state.black.walls_remaining}`,
  ].filter(Boolean);
  statusEl.textContent = parts.join(" | ");
}

function isPawnSelecting(): boolean {
  return selectedCell !== null;
}

function buildHints() {
  if (!ui.state || !isHumanTurn(ui) || pendingRequest) return null;

  const style = PAWN_STYLE[ui.humanColor];
  const moveTargets = isPawnSelecting()
    ? legalMoveTargets(ui.state, ui.humanColor).map(([row, col]) => ({ row, col }))
    : [];

  const hovered =
    !isPawnSelecting() &&
    hoveredWall &&
    isLegalWall(ui.state, ui.humanColor, hoveredWall.orientation, hoveredWall.row, hoveredWall.col)
      ? hoveredWall
      : null;

  if (moveTargets.length === 0 && !hovered) return null;

  return {
    moveTargets,
    pawnFill: style.fill,
    pawnStroke: style.stroke,
    hoveredWall: hovered,
  };
}

function refresh(): void {
  if (!ui.state) return;
  const hints = buildHints();
  renderBoard(
    boardEl,
    ui.state,
    ui.humanColor,
    onCellClick,
    onWallClick,
    {
      hints,
      wallsEnabled: isHumanTurn(ui) && !pendingRequest && !isPawnSelecting(),
      humanColor: ui.humanColor,
    },
    onWallHover,
    onWallLeave,
    (orientation, row, col) =>
      ui.state !== null && isLegalWall(ui.state, ui.humanColor, orientation, row, col),
  );
  updateStatus();
}

async function startGame(): Promise<void> {
  setError(null);
  selectedCell = null;
  hoveredWall = null;
  pendingRequest = false;
  const humanColor = humanColorEl.value as Color;
  const difficulty = difficultyEl.value as Difficulty;
  ui = initialUiState(humanColor);
  try {
    const resp = await createGame({ human_color: humanColor, difficulty });
    ui = {
      ...ui,
      gameId: resp.game_id,
      sessionToken: resp.session_token,
      state: resp.state,
      status: resp.status,
      turn: resp.turn,
      winner: resp.winner,
    };
    refresh();
  } catch (e) {
    setError(e instanceof ApiError ? `${e.code}: ${e.message}` : String(e));
  }
}

async function submitMove(action: MoveDTO): Promise<void> {
  if (!ui.gameId || !ui.sessionToken || !ui.state || !isHumanTurn(ui) || pendingRequest) return;

  const gameId = ui.gameId;
  const sessionToken = ui.sessionToken;
  const snapshot: GameStateDTO = structuredClone(ui.state);
  const optimistic = applyHumanAction(ui.state, ui.humanColor, action);
  if (!optimistic) return;

  setError(null);
  pendingRequest = true;
  selectedCell = null;
  hoveredWall = null;
  ui = { ...ui, state: optimistic, turn: "cpu" };
  refresh();

  try {
    const resp = await playMove(gameId, sessionToken, action);
    ui = {
      ...ui,
      state: resp.state,
      status: resp.status,
      turn: resp.turn,
      winner: resp.winner,
    };
  } catch (e) {
    ui = { ...ui, state: snapshot, turn: "human" };
    setError(e instanceof ApiError ? `${e.code}: ${e.message}` : String(e));
  } finally {
    pendingRequest = false;
    refresh();
  }
}

function isMoveTarget(row: number, col: number): boolean {
  if (!ui.state || !selectedCell) return false;
  return legalMoveTargets(ui.state, ui.humanColor).some(([r, c]) => r === row && c === col);
}

function onCellClick(row: number, col: number): void {
  if (!ui.state || !isHumanTurn(ui) || pendingRequest) return;
  const human = ui.humanColor === "white" ? ui.state.white : ui.state.black;

  if (!selectedCell) {
    if (human.row === row && human.col === col) {
      selectedCell = { row, col };
      hoveredWall = null;
      refresh();
    }
    return;
  }

  if (isMoveTarget(row, col)) {
    const action = actionForTarget(ui.state, ui.humanColor, row, col);
    if (action) void submitMove(action);
    return;
  }

  selectedCell = null;
  hoveredWall = null;
  refresh();
}

function onWallClick(orientation: "horizontal" | "vertical", row: number, col: number): void {
  if (!isHumanTurn(ui) || pendingRequest || isPawnSelecting() || !ui.state) return;
  if (!isLegalWall(ui.state, ui.humanColor, orientation, row, col)) return;
  void submitMove({
    type: "wall",
    orientation,
    position: { row, col },
  });
}

function onWallHover(orientation: "horizontal" | "vertical", row: number, col: number): void {
  if (!ui.state || !isHumanTurn(ui) || pendingRequest || isPawnSelecting()) return;
  if (!isLegalWall(ui.state, ui.humanColor, orientation, row, col)) return;
  hoveredWall = { orientation, row, col };
  refresh();
}

function onWallLeave(): void {
  if (!hoveredWall) return;
  hoveredWall = null;
  refresh();
}

newGameBtn.addEventListener("click", () => void startGame());
updateStatus();

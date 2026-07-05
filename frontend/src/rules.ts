import type { BoardPositionDTO, Color, Direction, GameStateDTO, MoveActionDTO, MoveDTO } from "./types/api";

const GOAL_ROW: Record<Color, number> = { white: 0, black: 8 };

type Pos = readonly [number, number];

function pawn(state: GameStateDTO, color: Color): Pos {
  const p = color === "white" ? state.white : state.black;
  return [p.row, p.col];
}

function absoluteDelta(color: Color, direction: Direction): Pos {
  if (direction === "left") return [0, -1];
  if (direction === "right") return [0, 1];
  if (direction === "up") return color === "white" ? [-1, 0] : [1, 0];
  return color === "white" ? [1, 0] : [-1, 0];
}

function isHorizontalWall(state: GameStateDTO, row: number, col: number): boolean {
  return row >= 0 && row <= 7 && col >= 0 && col <= 7 && state.horizontal_walls[row][col];
}

function isVerticalWall(state: GameStateDTO, row: number, col: number): boolean {
  return row >= 0 && row <= 7 && col >= 0 && col <= 7 && state.vertical_walls[row][col];
}

function canStep(state: GameStateDTO, from: Pos, to: Pos): boolean {
  const [fr, fc] = from;
  const [tr, tc] = to;
  if (tr < 0 || tr > 8 || tc < 0 || tc > 8) return false;
  const dr = tr - fr;
  const dc = tc - fc;
  if (Math.abs(dr) + Math.abs(dc) !== 1) return false;
  if (dc !== 0) {
    if (dc === 1) {
      return !isVerticalWall(state, fr, fc) && !isVerticalWall(state, fr - 1, fc);
    }
    return !isVerticalWall(state, fr, fc - 1) && !isVerticalWall(state, fr - 1, fc - 1);
  }
  if (dr === 1) {
    return !isHorizontalWall(state, fr, fc) && !isHorizontalWall(state, fr, fc - 1);
  }
  return !isHorizontalWall(state, fr - 1, fc) && !isHorizontalWall(state, fr - 1, fc - 1);
}

function pawnAdjacent(a: Pos, b: Pos): boolean {
  return Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]) === 1;
}

function occupiedCells(state: GameStateDTO): Set<string> {
  return new Set([
    `${state.white.row},${state.white.col}`,
    `${state.black.row},${state.black.col}`,
  ]);
}

function stepDestinationsInDirection(
  state: GameStateDTO,
  color: Color,
  pos: Pos,
  direction: Direction,
): Pos[] {
  const occupied = occupiedCells(state);
  const [dr, dc] = absoluteDelta(color, direction);
  const adj: Pos = [pos[0] + dr, pos[1] + dc];

  if (adj[0] < 0 || adj[0] > 8 || adj[1] < 0 || adj[1] > 8) return [];
  if (!canStep(state, pos, adj)) return [];

  const adjKey = `${adj[0]},${adj[1]}`;
  if (!occupied.has(adjKey)) {
    return [adj];
  }

  const jump: Pos = [adj[0] + dr, adj[1] + dc];
  const allowStraightJump = adj[1] !== 0 && adj[1] !== 8;
  if (
    allowStraightJump &&
    jump[0] >= 0 &&
    jump[0] <= 8 &&
    jump[1] >= 0 &&
    jump[1] <= 8 &&
    !occupied.has(`${jump[0]},${jump[1]}`) &&
    canStep(state, adj, jump)
  ) {
    return [jump];
  }

  const perpendicular = (direction === "up" || direction === "down"
    ? (["left", "right"] as const)
    : (["up", "down"] as const));
  const diags: Pos[] = [];
  for (const perp of perpendicular) {
    const [pdr, pdc] = absoluteDelta(color, perp);
    const diag: Pos = [adj[0] + pdr, adj[1] + pdc];
    if (diag[0] < 0 || diag[0] > 8 || diag[1] < 0 || diag[1] > 8) continue;
    if (occupied.has(`${diag[0]},${diag[1]}`)) continue;
    if (!canStep(state, adj, diag)) continue;
    if (
      diag[1] < adj[1] &&
      dr !== 0 &&
      isHorizontalWall(state, adj[0] + dr, adj[1]) &&
      (isVerticalWall(state, adj[0], adj[1]) || isVerticalWall(state, adj[0] - 1, adj[1]))
    ) {
      continue;
    }
    diags.push(diag);
  }
  return diags;
}

function stepDestinationsFrom(state: GameStateDTO, color: Color, pos: Pos): Pos[] {
  const dests: Pos[] = [];
  for (const direction of ["up", "down", "left", "right"] as const) {
    dests.push(...stepDestinationsInDirection(state, color, pos, direction));
  }
  return dests;
}

function moveDestinations(state: GameStateDTO, color: Color, direction: Direction): Pos[] {
  return stepDestinationsInDirection(state, color, pawn(state, color), direction);
}

function expandBfsNeighbors(state: GameStateDTO, color: Color, pos: Pos): Pos[] {
  const opponent = pawn(state, color === "white" ? "black" : "white");
  if (pawnAdjacent(pos, opponent)) {
    return stepDestinationsFrom(state, color, pos);
  }
  const neighbors: Pos[] = [];
  for (const [dr, dc] of [[0, 1], [0, -1], [1, 0], [-1, 0]] as const) {
    const nxt: Pos = [pos[0] + dr, pos[1] + dc];
    if (nxt[0] === opponent[0] && nxt[1] === opponent[1]) continue;
    if (nxt[0] < 0 || nxt[0] > 8 || nxt[1] < 0 || nxt[1] > 8) continue;
    if (!canStep(state, pos, nxt)) continue;
    neighbors.push(nxt);
  }
  return neighbors;
}

export function legalMoveTargets(state: GameStateDTO, color: Color): Pos[] {
  const seen = new Set<string>();
  const targets: Pos[] = [];
  for (const direction of ["up", "down", "left", "right"] as const) {
    for (const [row, col] of moveDestinations(state, color, direction)) {
      const key = `${row},${col}`;
      if (!seen.has(key)) {
        seen.add(key);
        targets.push([row, col]);
      }
    }
  }
  return targets;
}

function withWall(
  state: GameStateDTO,
  orientation: "horizontal" | "vertical",
  row: number,
  col: number,
): GameStateDTO {
  const horizontal = state.horizontal_walls.map((r) => [...r]);
  const vertical = state.vertical_walls.map((r) => [...r]);
  if (orientation === "horizontal") horizontal[row][col] = true;
  else vertical[row][col] = true;
  return { ...state, horizontal_walls: horizontal, vertical_walls: vertical };
}

function bfsDistance(state: GameStateDTO, color: Color): number | null {
  const start = pawn(state, color);
  const goalRow = GOAL_ROW[color];
  const visited = new Set<string>([`${start[0]},${start[1]}`]);
  const queue: Array<{ pos: Pos; dist: number }> = [{ pos: start, dist: 0 }];

  while (queue.length > 0) {
    const { pos, dist } = queue.shift()!;
    if (pos[0] === goalRow) return dist;
    for (const nxt of expandBfsNeighbors(state, color, pos)) {
      const key = `${nxt[0]},${nxt[1]}`;
      if (visited.has(key)) continue;
      visited.add(key);
      queue.push({ pos: nxt, dist: dist + 1 });
    }
  }
  return null;
}

function canReachGoal(state: GameStateDTO, color: Color): boolean {
  return bfsDistance(state, color) !== null;
}

function wallPassesFilters(
  state: GameStateDTO,
  color: Color,
  orientation: "horizontal" | "vertical",
  row: number,
  col: number,
): boolean {
  const wallsRemaining = color === "white" ? state.white.walls_remaining : state.black.walls_remaining;
  if (wallsRemaining <= 0) return false;
  if (orientation === "horizontal") {
    if (state.horizontal_walls[row][col]) return false;
    if (state.vertical_walls[row][col]) return false;
    if (col > 0 && state.horizontal_walls[row][col - 1]) return false;
    if (col < 7 && state.horizontal_walls[row][col + 1]) return false;
  } else {
    if (state.vertical_walls[row][col]) return false;
    if (state.horizontal_walls[row][col]) return false;
    if (row > 0 && state.vertical_walls[row - 1][col]) return false;
    if (row < 7 && state.vertical_walls[row + 1][col]) return false;
  }
  return true;
}

export function isLegalWall(
  state: GameStateDTO,
  color: Color,
  orientation: "horizontal" | "vertical",
  row: number,
  col: number,
): boolean {
  if (!wallPassesFilters(state, color, orientation, row, col)) return false;
  const next = withWall(state, orientation, row, col);
  return canReachGoal(next, "white") && canReachGoal(next, "black");
}

export function resolveMoveDest(
  state: GameStateDTO,
  color: Color,
  direction: Direction,
  to?: BoardPositionDTO,
): BoardPositionDTO | null {
  const dests = moveDestinations(state, color, direction);
  if (dests.length === 0) return null;
  if (dests.length === 1) {
    const only = dests[0];
    if (to && (to.row !== only[0] || to.col !== only[1])) return null;
    return { row: only[0], col: only[1] };
  }
  if (!to) return null;
  for (const [row, col] of dests) {
    if (row === to.row && col === to.col) return to;
  }
  return null;
}

export function actionForTarget(
  state: GameStateDTO,
  color: Color,
  toRow: number,
  toCol: number,
): MoveDTO | null {
  for (const direction of ["up", "down", "left", "right"] as const) {
    for (const [row, col] of moveDestinations(state, color, direction)) {
      if (row === toRow && col === toCol) {
        return { type: "move", direction, to: { row, col } };
      }
    }
  }
  return null;
}

function destinationForMove(
  state: GameStateDTO,
  color: Color,
  action: MoveActionDTO,
): Pos | null {
  if (action.to) {
    const resolved = resolveMoveDest(state, color, action.direction, action.to);
    if (!resolved) return null;
    return [resolved.row, resolved.col];
  }
  const dests = moveDestinations(state, color, action.direction);
  return dests.length === 1 ? dests[0] : null;
}

export function applyHumanAction(
  state: GameStateDTO,
  color: Color,
  action: MoveDTO,
): GameStateDTO | null {
  if (action.type === "wall") {
    const { orientation, position } = action;
    if (!isLegalWall(state, color, orientation, position.row, position.col)) return null;
    const next = withWall(state, orientation, position.row, position.col);
    const cpu: Color = color === "white" ? "black" : "white";
    if (color === "white") {
      return {
        ...next,
        white: { ...next.white, walls_remaining: next.white.walls_remaining - 1 },
        current_player: cpu,
      };
    }
    return {
      ...next,
      black: { ...next.black, walls_remaining: next.black.walls_remaining - 1 },
      current_player: cpu,
    };
  }

  const dest = destinationForMove(state, color, action);
  if (!dest) return null;
  const cpu: Color = color === "white" ? "black" : "white";
  if (color === "white") {
    return {
      ...state,
      white: { ...state.white, row: dest[0], col: dest[1] },
      current_player: cpu,
    };
  }
  return {
    ...state,
    black: { ...state.black, row: dest[0], col: dest[1] },
    current_player: cpu,
  };
}

import type { Color, GameStateDTO } from "./types/api";

const CELL = 45;
const OFFSET = 22;
const PAWN_SIZE = 36;
const HALF_PAWN = PAWN_SIZE / 2;
const GROOVE = (CELL - PAWN_SIZE) / 2;

export interface WallSlot {
  orientation: "horizontal" | "vertical";
  row: number;
  col: number;
}

export interface BoardHints {
  moveTargets: Array<{ row: number; col: number }>;
  pawnFill: string;
  pawnStroke: string;
  hoveredWall: WallSlot | null;
}

/** Black's absolute orientation is inverted; flip so the human sits at visual bottom. */
function needsFlip(humanColor: Color): boolean {
  return humanColor === "black";
}

function displayPawnRow(modelRow: number, flip: boolean): number {
  return flip ? 8 - modelRow : modelRow;
}

function displayWallRow(modelRow: number, flip: boolean): number {
  return flip ? 7 - modelRow : modelRow;
}

function pos(row: number, col: number): [number, number] {
  return [OFFSET + col * CELL, OFFSET + row * CELL];
}

function horizontalGrooveY(row: number): number {
  return OFFSET + row * CELL + HALF_PAWN + GROOVE;
}

function verticalGrooveX(col: number): number {
  return OFFSET + col * CELL + HALF_PAWN + GROOVE;
}

function horizontalWallSpan(row: number, col: number): { x1: number; x2: number; y: number } {
  const [xLeft] = pos(row, col);
  const [xRight] = pos(row, col + 1);
  return {
    x1: xLeft - HALF_PAWN,
    x2: xRight + HALF_PAWN,
    y: horizontalGrooveY(row),
  };
}

function verticalWallSpan(row: number, col: number): { x: number; y1: number; y2: number } {
  const [, yTop] = pos(row, col);
  const [, yBottom] = pos(row + 1, col);
  return {
    x: verticalGrooveX(col),
    y1: yTop - HALF_PAWN,
    y2: yBottom + HALF_PAWN,
  };
}

export interface BoardRenderOptions {
  hints: BoardHints | null;
  wallsEnabled: boolean;
  humanColor: Color;
}

export function renderBoard(
  svg: SVGSVGElement,
  state: GameStateDTO,
  humanColor: Color,
  onCellClick: (row: number, col: number) => void,
  onWallClick: (orientation: "horizontal" | "vertical", row: number, col: number) => void,
  options: BoardRenderOptions | null = null,
  onWallHover: ((orientation: "horizontal" | "vertical", row: number, col: number) => void) | null = null,
  onWallLeave: (() => void) | null = null,
  isLegalWallAt: (
    orientation: "horizontal" | "vertical",
    row: number,
    col: number,
  ) => boolean = () => true,
): void {
  const hints = options?.hints ?? null;
  const wallsEnabled = options?.wallsEnabled ?? true;
  const flip = needsFlip(humanColor);
  const renderKey = JSON.stringify({
    state,
    humanColor,
    wallsEnabled,
    hints,
  });
  if (svg.dataset.renderKey === renderKey) {
    return;
  }
  svg.dataset.renderKey = renderKey;
  svg.innerHTML = "";

  for (let r = 0; r < 9; r++) {
    for (let c = 0; c < 9; c++) {
      const [x, y] = pos(displayPawnRow(r, flip), c);
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", String(x - HALF_PAWN));
      rect.setAttribute("y", String(y - HALF_PAWN));
      rect.setAttribute("width", String(PAWN_SIZE));
      rect.setAttribute("height", String(PAWN_SIZE));
      rect.setAttribute("fill", (r + c) % 2 === 0 ? "#e8d4a8" : "#d4bc88");
      rect.setAttribute("stroke", "#8b7355");
      rect.setAttribute("data-testid", "board-cell");
      // Model coordinates stay absolute for API / e2e; only SVG placement flips.
      rect.setAttribute("data-row", String(r));
      rect.setAttribute("data-col", String(c));
      rect.style.cursor = "pointer";
      rect.addEventListener("click", () => onCellClick(r, c));
      svg.appendChild(rect);
    }
  }

  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const visRow = displayWallRow(r, flip);
      if (state.horizontal_walls[r][c]) {
        drawWall(svg, "horizontal", visRow, c);
      }
      if (state.vertical_walls[r][c]) {
        drawWall(svg, "vertical", visRow, c);
      }
      if (wallsEnabled) {
        if (isLegalWallAt("horizontal", r, c)) {
          drawWallHotspot(svg, "horizontal", r, c, visRow, onWallClick, onWallHover, onWallLeave);
        }
        if (isLegalWallAt("vertical", r, c)) {
          drawWallHotspot(svg, "vertical", r, c, visRow, onWallClick, onWallHover, onWallLeave);
        }
      }
    }
  }

  if (hints?.hoveredWall) {
    drawWallPreview(
      svg,
      hints.hoveredWall.orientation,
      displayWallRow(hints.hoveredWall.row, flip),
      hints.hoveredWall.col,
    );
  }

  if (hints) {
    for (const target of hints.moveTargets) {
      drawGhostPawn(
        svg,
        displayPawnRow(target.row, flip),
        target.col,
        target.row,
        target.col,
        hints.pawnFill,
        hints.pawnStroke,
        onCellClick,
      );
    }
  }

  drawPawn(
    svg,
    displayPawnRow(state.white.row, flip),
    state.white.col,
    state.white.row,
    state.white.col,
    "#f8f8f8",
    "#333",
    onCellClick,
  );
  drawPawn(
    svg,
    displayPawnRow(state.black.row, flip),
    state.black.col,
    state.black.row,
    state.black.col,
    "#2d2d2d",
    "#fff",
    onCellClick,
  );

  const human = humanColor === "white" ? state.white : state.black;
  const [hx, hy] = pos(displayPawnRow(human.row, flip), human.col);
  const ring = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  ring.setAttribute("cx", String(hx));
  ring.setAttribute("cy", String(hy));
  ring.setAttribute("r", "20");
  ring.setAttribute("fill", "none");
  ring.setAttribute("stroke", "#4ecdc4");
  ring.setAttribute("stroke-width", "3");
  ring.setAttribute("data-testid", "human-pawn-ring");
  svg.appendChild(ring);
}

function drawGhostPawn(
  svg: SVGSVGElement,
  displayRow: number,
  displayCol: number,
  modelRow: number,
  modelCol: number,
  fill: string,
  stroke: string,
  onCellClick: (row: number, col: number) => void,
): void {
  const [x, y] = pos(displayRow, displayCol);
  const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  c.setAttribute("cx", String(x));
  c.setAttribute("cy", String(y));
  c.setAttribute("r", "14");
  c.setAttribute("fill", fill);
  c.setAttribute("fill-opacity", "0.35");
  c.setAttribute("stroke", stroke);
  c.setAttribute("stroke-opacity", "0.5");
  c.setAttribute("stroke-width", "2");
  c.setAttribute("data-testid", "pawn-ghost");
  c.setAttribute("data-row", String(modelRow));
  c.setAttribute("data-col", String(modelCol));
  c.style.cursor = "pointer";
  c.style.pointerEvents = "none";
  c.addEventListener("click", (e) => {
    e.stopPropagation();
    onCellClick(modelRow, modelCol);
  });
  svg.appendChild(c);
}

function drawPawn(
  svg: SVGSVGElement,
  displayRow: number,
  displayCol: number,
  modelRow: number,
  modelCol: number,
  fill: string,
  stroke: string,
  onCellClick: (row: number, col: number) => void,
): void {
  const [x, y] = pos(displayRow, displayCol);
  const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  c.setAttribute("cx", String(x));
  c.setAttribute("cy", String(y));
  c.setAttribute("r", "14");
  c.setAttribute("fill", fill);
  c.setAttribute("stroke", stroke);
  c.setAttribute("stroke-width", "2");
  c.setAttribute("data-testid", fill === "#f8f8f8" ? "pawn-white" : "pawn-black");
  c.setAttribute("data-row", String(modelRow));
  c.setAttribute("data-col", String(modelCol));
  c.style.cursor = "pointer";
  c.addEventListener("click", (e) => {
    e.stopPropagation();
    onCellClick(modelRow, modelCol);
  });
  svg.appendChild(c);
}

function drawWall(svg: SVGSVGElement, orientation: "horizontal" | "vertical", row: number, col: number): void {
  const line = createWallLine("#5c4033", 1);
  line.setAttribute("data-testid", "wall-line");
  setWallLineCoords(line, orientation, row, col);
  svg.appendChild(line);
}

function drawWallPreview(
  svg: SVGSVGElement,
  orientation: "horizontal" | "vertical",
  row: number,
  col: number,
): void {
  const line = createWallLine("#5c4033", 0.4);
  line.setAttribute("data-testid", "wall-preview");
  setWallLineCoords(line, orientation, row, col);
  svg.appendChild(line);
}

function createWallLine(stroke: string, opacity: number): SVGLineElement {
  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("stroke", stroke);
  line.setAttribute("stroke-width", "6");
  line.setAttribute("stroke-linecap", "round");
  line.setAttribute("stroke-opacity", String(opacity));
  line.style.pointerEvents = "none";
  return line as SVGLineElement;
}

function setWallLineCoords(
  line: SVGLineElement,
  orientation: "horizontal" | "vertical",
  row: number,
  col: number,
): void {
  if (orientation === "horizontal") {
    const { x1, x2, y } = horizontalWallSpan(row, col);
    line.setAttribute("x1", String(x1));
    line.setAttribute("y1", String(y));
    line.setAttribute("x2", String(x2));
    line.setAttribute("y2", String(y));
  } else {
    const { x, y1, y2 } = verticalWallSpan(row, col);
    line.setAttribute("x1", String(x));
    line.setAttribute("y1", String(y1));
    line.setAttribute("x2", String(x));
    line.setAttribute("y2", String(y2));
  }
}

function drawWallHotspot(
  svg: SVGSVGElement,
  orientation: "horizontal" | "vertical",
  modelRow: number,
  modelCol: number,
  displayRow: number,
  onWallClick: (orientation: "horizontal" | "vertical", row: number, col: number) => void,
  onWallHover: ((orientation: "horizontal" | "vertical", row: number, col: number) => void) | null,
  onWallLeave: (() => void) | null,
): void {
  const hitPad = 4;
  const hitThickness = 16;
  const gap = hitThickness / 2 + 1;

  const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
  group.setAttribute("data-testid", orientation === "horizontal" ? "wall-h" : "wall-v");
  group.setAttribute("data-row", String(modelRow));
  group.setAttribute("data-col", String(modelCol));
  group.style.cursor = "crosshair";
  group.addEventListener("click", (e) => {
    e.stopPropagation();
    onWallClick(orientation, modelRow, modelCol);
  });
  if (onWallHover) {
    group.addEventListener("mouseenter", () => onWallHover(orientation, modelRow, modelCol));
  }
  if (onWallLeave) {
    group.addEventListener("mouseleave", () => onWallLeave());
  }

  const addHotspot = (x: number, y: number, width: number, height: number) => {
    if (width <= 0 || height <= 0) return;
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", String(x));
    rect.setAttribute("y", String(y));
    rect.setAttribute("width", String(width));
    rect.setAttribute("height", String(height));
    rect.setAttribute("fill", "transparent");
    rect.style.pointerEvents = "all";
    group.appendChild(rect);
  };

  if (orientation === "horizontal") {
    const { x1, x2, y } = horizontalWallSpan(displayRow, modelCol);
    const hy = y - hitThickness / 2;
    const h = hitThickness;
    const splitX = verticalGrooveX(modelCol);
    addHotspot(x1 - hitPad, hy, splitX - gap - (x1 - hitPad), h);
    addHotspot(splitX + gap, hy, x2 + hitPad - (splitX + gap), h);
  } else {
    const { x, y1, y2 } = verticalWallSpan(displayRow, modelCol);
    const hx = x - hitThickness / 2;
    const w = hitThickness;
    const splitY = horizontalGrooveY(displayRow);
    addHotspot(hx, y1 - hitPad, w, splitY - gap - (y1 - hitPad));
    addHotspot(hx, splitY + gap, w, y2 + hitPad - (splitY + gap));
  }

  svg.appendChild(group);
}

import type { GameStateDTO } from "../types/api";

function emptyWalls(): boolean[][] {
  return Array.from({ length: 8 }, () => Array.from({ length: 8 }, () => false));
}

function buildState(params: {
  white?: [number, number];
  black?: [number, number];
  current?: "white" | "black";
  horizontal?: Array<[number, number]>;
  vertical?: Array<[number, number]>;
}): GameStateDTO {
  const horizontal = emptyWalls();
  const vertical = emptyWalls();
  for (const [row, col] of params.horizontal ?? []) {
    horizontal[row][col] = true;
  }
  for (const [row, col] of params.vertical ?? []) {
    vertical[row][col] = true;
  }
  const white = params.white ?? [8, 4];
  const black = params.black ?? [0, 4];
  return {
    white: { row: white[0], col: white[1], walls_remaining: 10 },
    black: { row: black[0], col: black[1], walls_remaining: 10 },
    horizontal_walls: horizontal,
    vertical_walls: vertical,
    current_player: params.current ?? "white",
  };
}

export const JUMP_CASES = [
  {
    id: "J.1-STRAIGHT",
    state: buildState({ white: [5, 4], black: [4, 4], current: "white" }),
    direction: "up" as const,
    expected: [
      [3, 4],
    ],
  },
  {
    id: "J.2-DIAG-BOTH",
    state: buildState({
      white: [5, 4],
      black: [4, 4],
      current: "white",
      horizontal: [[3, 4]],
    }),
    direction: "up" as const,
    expected: [
      [4, 3],
      [4, 5],
    ],
  },
] as const;

export const WALL_CASES = [
  {
    id: "B.1-L",
    state: buildState({ horizontal: [[0, 2]] }),
    orientation: "vertical" as const,
    row: 0,
    col: 3,
    expectedLegal: true,
  },
  {
    id: "B.1-S",
    state: buildState({ horizontal: [[0, 3]] }),
    orientation: "horizontal" as const,
    row: 0,
    col: 3,
    expectedLegal: false,
  },
] as const;

export const WALL_PATH_CASES = [
  {
    id: "W.TAK-101-TOP-RIGHT",
    state: buildState({
      white: [4, 4],
      black: [1, 4],
      current: "black",
      horizontal: [
        [0, 0],
        [0, 2],
        [0, 5],
      ],
    }),
    orientation: "horizontal" as const,
    row: 0,
    col: 7,
    expectedLegal: true,
  },
] as const;

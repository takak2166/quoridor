import type { Direction, GameStateDTO } from "../types/api";

import generated from "./planFixtures.generated.json";

type JumpCase = {
  id: string;
  state: GameStateDTO;
  direction: Direction;
  expected: readonly [number, number][];
};

type WallCase = {
  id: string;
  state: GameStateDTO;
  orientation: "horizontal" | "vertical";
  row: number;
  col: number;
  expectedLegal: boolean;
};

export const JUMP_CASES = generated.jumpCases as unknown as JumpCase[];
export const WALL_CASES = generated.wallCases as unknown as WallCase[];
export const WALL_PATH_CASES = generated.wallPathCases as unknown as WallCase[];

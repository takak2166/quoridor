import { describe, expect, test } from "vitest";

import { JUMP_CASES, WALL_CASES, WALL_PATH_CASES } from "./__fixtures__/backendPlanFixtures";
import { isLegalWall, resolveMoveDest } from "./rules";

describe("rules fixtures parity", () => {
  test.each(JUMP_CASES)("$id", ({ state, direction, expected }) => {
    if (expected.length === 1) {
      const resolved = resolveMoveDest(state, state.current_player, direction);
      expect(resolved).toEqual({ row: expected[0][0], col: expected[0][1] });
      return;
    }
    expect(resolveMoveDest(state, state.current_player, direction)).toBeNull();
    for (const [row, col] of expected) {
      const resolved = resolveMoveDest(state, state.current_player, direction, { row, col });
      expect(resolved).toEqual({ row, col });
    }
  });

  test.each(WALL_CASES)("$id", ({ state, orientation, row, col, expectedLegal }) => {
    expect(isLegalWall(state, state.current_player, orientation, row, col)).toBe(expectedLegal);
  });

  test.each(WALL_PATH_CASES)("$id", ({ state, orientation, row, col, expectedLegal }) => {
    expect(isLegalWall(state, state.current_player, orientation, row, col)).toBe(expectedLegal);
  });
});

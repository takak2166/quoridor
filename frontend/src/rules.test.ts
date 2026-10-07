import { describe, expect, test } from "vitest";

import { JUMP_CASES, WALL_CASES, WALL_PATH_CASES } from "./__fixtures__/backendPlanFixtures";
import { isLegalWall, moveDestinations, resolveMoveDest } from "./rules";

function sortPositions(positions: readonly [number, number][]): [number, number][] {
  return [...positions].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
}

describe("rules fixtures parity", () => {
  test.each(JUMP_CASES)("$id", ({ state, direction, expected }) => {
    const actual = sortPositions(
      moveDestinations(state, state.current_player, direction).map(
        ([row, col]) => [row, col] as [number, number],
      ),
    );
    expect(actual).toEqual(sortPositions(expected));
  });

  test.each(WALL_CASES)("$id", ({ state, orientation, row, col, expectedLegal }) => {
    expect(isLegalWall(state, state.current_player, orientation, row, col)).toBe(expectedLegal);
  });

  test.each(WALL_PATH_CASES)("$id", ({ state, orientation, row, col, expectedLegal }) => {
    expect(isLegalWall(state, state.current_player, orientation, row, col)).toBe(expectedLegal);
  });

  test.each(JUMP_CASES)("$id resolveMoveDest contract", ({ state, direction, expected }) => {
    if (expected.length === 1) {
      const [row, col] = expected[0];
      expect(resolveMoveDest(state, state.current_player, direction)).toEqual({ row, col });
      expect(resolveMoveDest(state, state.current_player, direction, { row, col })).toEqual({
        row,
        col,
      });
      return;
    }
    expect(resolveMoveDest(state, state.current_player, direction)).toBeNull();
    for (const [row, col] of expected) {
      expect(resolveMoveDest(state, state.current_player, direction, { row, col })).toEqual({
        row,
        col,
      });
    }
  });
});

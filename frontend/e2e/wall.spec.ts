import { expect, test } from "@playwright/test";

import {
  expectCpuTurn,
  expectNoWallHotspot,
  placeWall,
  startGame,
  waitForHumanTurn,
} from "./helpers";

test.describe("Wall placement", () => {
  test("E2E-030: can place a horizontal wall", async ({ page }) => {
    await page.goto("/");
    await startGame(page, { humanColor: "black" });

    await placeWall(page, "horizontal", 1, 1);
    await expect(page.getByTestId("game-error")).toBeEmpty();
    await expect(page.getByTestId("game-status")).toContainText("Black walls: 9");
    await waitForHumanTurn(page);
  });

  test("E2E-031-032: vertical wall placement reduces remaining walls", async ({ page }) => {
    await page.goto("/");
    await startGame(page, { humanColor: "black" });

    await placeWall(page, "vertical", 2, 2);
    await expect(page.getByTestId("game-error")).toBeEmpty();
    await expect(page.getByTestId("game-status")).toContainText("Black walls: 9");
    await waitForHumanTurn(page);
    await expect
      .poll(async () => page.getByTestId("board").locator('[data-testid="wall-line"]').count())
      .toBeGreaterThan(0);
  });

  test("E2E-033: duplicate wall at same slot is not available in UI", async ({ page }) => {
    await page.goto("/");
    await startGame(page, { humanColor: "black" });

    await placeWall(page, "horizontal", 1, 1);
    await expect(page.getByTestId("game-error")).toBeEmpty();
    await waitForHumanTurn(page);

    await expectNoWallHotspot(page, "horizontal", 1, 1);
    await expect(page.getByTestId("game-status")).toContainText("Black walls: 9");
  });

  test("E2E-035: adjacent same-orientation walls hidden in UI", async ({ page }) => {
    await page.goto("/");
    await startGame(page, { humanColor: "black" });

    await placeWall(page, "vertical", 4, 1);
    await expect(page.getByTestId("game-error")).toBeEmpty();
    await waitForHumanTurn(page);

    await expectNoWallHotspot(page, "vertical", 5, 1);
    await expectNoWallHotspot(page, "vertical", 3, 1);
  });

  test("E2E-034: cannot place wall during CPU turn", async ({ page }) => {
    await page.goto("/");
    await startGame(page, { humanColor: "black" });

    await page.route("**/api/v1/games/*/moves", async (route) => {
      const body = route.request().postDataJSON() as { action?: { type?: string } };
      if (body?.action?.type === "wall") {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            human_move: body.action,
            cpu_move: null,
            status: "active",
            turn: "cpu",
            winner: null,
            state: {
              white: { row: 8, col: 4, walls_remaining: 10 },
              black: { row: 0, col: 4, walls_remaining: 9 },
              horizontal_walls: Array.from({ length: 8 }, (_, r) =>
                Array.from({ length: 8 }, (_, c) => r === 1 && c === 1),
              ),
              vertical_walls: Array.from({ length: 8 }, () => Array(8).fill(false)),
              current_player: "white",
            },
          }),
        });
        return;
      }
      await route.continue();
    });

    await placeWall(page, "horizontal", 1, 1);
    await expectCpuTurn(page);

    const wallsBefore = await page.getByTestId("board").locator('[data-testid="wall-line"]').count();
    await expectNoWallHotspot(page, "horizontal", 2, 2);
    await expect(page.getByTestId("board").locator('[data-testid="wall-line"]')).toHaveCount(wallsBefore);
    await expect(page.getByTestId("game-status")).toContainText("Black walls: 9");
  });
});

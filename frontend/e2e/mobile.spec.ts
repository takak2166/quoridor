import { expect, test } from "@playwright/test";

import { movePawn, startGame, waitForHumanTurn } from "./helpers";

test.beforeEach(({ }, testInfo) => {
  test.skip(testInfo.project.name !== "mobile-chrome", "mobile viewport only");
});

test("mobile: tap to move pawn", async ({ page }) => {
  await page.goto("/");
  await startGame(page, { humanColor: "black", difficulty: "easy" });

  const board = page.getByTestId("board");
  await expect(board).toBeVisible();
  const box = await board.boundingBox();
  expect(box).not.toBeNull();
  if (box && box.width > 450) {
    expect(box.width).toBeLessThanOrEqual(page.viewportSize()?.width ?? box.width);
  }

  await movePawn(page, 0, 4, 1, 4);
  await waitForHumanTurn(page);
});

test("mobile: tap to place wall", async ({ page }) => {
  await page.goto("/");
  await startGame(page, { humanColor: "black", difficulty: "easy" });

  const { placeWall } = await import("./helpers");
  await placeWall(page, "horizontal", 1, 1);
  await expect(page.getByTestId("game-status")).toContainText("Black walls: 9");
});

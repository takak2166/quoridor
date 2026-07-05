import { expect, test } from "@playwright/test";

import { boardCell, pawnAt, placeWall, startGame, wallHotspot } from "./helpers";

test.describe("Rendering regression", () => {
  test("E2E-070: white and black pawns are visually distinct", async ({ page }) => {
    await page.goto("/");
    await startGame(page, { humanColor: "black" });

    await expect(page.getByTestId("pawn-white")).toHaveAttribute("fill", "#f8f8f8");
    await expect(page.getByTestId("pawn-black")).toHaveAttribute("fill", "#2d2d2d");
  });

  test("E2E-071: human pawn highlight ring is visible", async ({ page }) => {
    await page.goto("/");
    await startGame(page, { humanColor: "black" });

    await expect(page.getByTestId("human-pawn-ring")).toBeVisible();
    await expect(page.getByTestId("human-pawn-ring")).toHaveAttribute("stroke", "#4ecdc4");
  });

  test("E2E-072: wall line is drawn after placement", async ({ page }) => {
    await page.goto("/");
    await startGame(page, { humanColor: "black" });

    await expect(page.getByTestId("board").locator('[data-testid="wall-line"]')).toHaveCount(0);
    await placeWall(page, "horizontal", 1, 1);
    await expect
      .poll(async () => page.getByTestId("board").locator('[data-testid="wall-line"]').count())
      .toBeGreaterThan(0);
  });

  test.skip("E2E-073: initial board screenshot regression", async () => {
    // Optional: requires dedicated CI and baseline management for visual regression.
  });

  test("E2E-074: pawn selection shows ghost pawns and disables wall hotspots", async ({ page }) => {
    await page.goto("/");
    await startGame(page, { humanColor: "black" });

    await pawnAt(page, 0, 4).click();
    await expect(page.getByTestId("pawn-ghost")).toHaveCount(3);
    await expect(wallHotspot(page, "horizontal", 1, 1)).toHaveCount(0);

    await boardCell(page, 2, 4).click();
    await expect(page.getByTestId("pawn-ghost")).toHaveCount(0);
  });
});

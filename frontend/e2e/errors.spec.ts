import { expect, test } from "@playwright/test";

import { errorBody } from "./fixtures";
import { expectPawnAt, movePawn, placeWall, startGame } from "./helpers";

test.describe("Error display (extended)", () => {
  test("E2E-041: board unchanged after illegal move", async ({ page }) => {
    await page.goto("/");
    await startGame(page, { humanColor: "black" });

    await page.route("**/api/v1/games/*/moves", async (route) => {
      await route.fulfill({
        status: 400,
        contentType: "application/json",
        body: errorBody("ILLEGAL_MOVE", "blocked"),
      });
    });

    await placeWall(page, "horizontal", 1, 1);
    await expect(page.getByTestId("game-error")).toContainText("ILLEGAL_MOVE");
    await expect
      .poll(async () => page.getByTestId("board").locator('[data-testid="wall-line"]').count())
      .toBe(0);
    await expectPawnAt(page, "black", 0, 4);
  });

  test("E2E-042: AI failure shows AI_FAILURE", async ({ page }) => {
    await page.goto("/");
    await startGame(page, { humanColor: "black" });

    await page.route("**/api/v1/games/*/moves", async (route) => {
      await route.fulfill({
        status: 503,
        contentType: "application/json",
        body: errorBody("AI_FAILURE", "AI inference failed"),
      });
    });

    await movePawn(page, 0, 4, 1, 4);
    await expect(page.getByTestId("game-error")).toContainText("AI_FAILURE");
  });

  test("E2E-043: rate limit shows RATE_LIMITED", async ({ page }) => {
    await page.goto("/");

    await page.route("**/api/v1/games", async (route) => {
      if (route.request().method() === "POST") {
        await route.fulfill({
          status: 429,
          contentType: "application/json",
          body: errorBody("RATE_LIMITED", "Too many requests"),
        });
        return;
      }
      await route.continue();
    });

    await page.getByTestId("new-game").click();
    await expect(page.getByTestId("game-error")).toContainText("RATE_LIMITED");
  });

  test("E2E-044: expired session shows SESSION_EXPIRED", async ({ page }) => {
    await page.goto("/");
    await startGame(page, { humanColor: "black" });

    await page.route("**/api/v1/games/*/moves", async (route) => {
      await route.fulfill({
        status: 403,
        contentType: "application/json",
        body: errorBody("SESSION_EXPIRED", "Game not found"),
      });
    });

    await movePawn(page, 0, 4, 1, 4);
    await expect(page.getByTestId("game-error")).toContainText("SESSION_EXPIRED");
  });

  test("E2E-045: move after game over shows GAME_OVER", async ({ page }) => {
    await page.goto("/");
    await startGame(page, { humanColor: "black" });

    await page.route("**/api/v1/games/*/moves", async (route) => {
      await route.fulfill({
        status: 409,
        contentType: "application/json",
        body: errorBody("GAME_OVER", "Game is finished"),
      });
    });

    await movePawn(page, 0, 4, 1, 4);
    await expect(page.getByTestId("game-error")).toContainText("GAME_OVER");
  });
});

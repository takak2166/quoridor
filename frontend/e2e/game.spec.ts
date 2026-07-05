import { expect, test } from "@playwright/test";

import {
  boardCell,
  clickNewGame,
  expectPawnAt,
  movePawn,
  pawnAt,
  startGame,
  waitForHumanTurn,
} from "./helpers";

test.describe("New game", () => {
  test("E2E-010-012: start game as black (first player)", async ({ page }) => {
    await page.goto("/");
    await startGame(page, { humanColor: "black", difficulty: "easy" });

    await expect(page.getByTestId("game-status")).toContainText("Turn: human");
    await expect(page.getByTestId("game-status")).toContainText("White walls: 10");
    await expect(page.getByTestId("game-status")).toContainText("Black walls: 10");
    await expectPawnAt(page, "white", 8, 4);
    await expectPawnAt(page, "black", 0, 4);
  });

  test("E2E-013-014: white (second) gets human turn after CPU opening", async ({ page }) => {
    await page.goto("/");
    await startGame(page, { humanColor: "white", difficulty: "easy" });

    await expect(page.getByTestId("game-status")).toContainText("Turn: human");
    await expectPawnAt(page, "white", 8, 4);

    const status = await page.getByTestId("game-status").textContent();
    const blackRow = await page.getByTestId("pawn-black").getAttribute("data-row");
    const cpuMovedPawn = blackRow !== "0";
    const cpuPlacedWall = status?.includes("Black walls: 9") ?? false;
    expect(cpuMovedPawn || cpuPlacedWall).toBeTruthy();
  });

  test("E2E-016-017: error clears on consecutive new games", async ({ page }) => {
    await page.goto("/");
    await startGame(page);

    await page.route("**/api/v1/games/*/moves", async (route) => {
      await route.fulfill({
        status: 400,
        contentType: "application/json",
        body: JSON.stringify({
          detail: { error: { code: "ILLEGAL_MOVE", message: "illegal" } },
        }),
      });
    });

    await movePawn(page, 0, 4, 1, 4);
    await expect(page.getByTestId("game-error")).toContainText("ILLEGAL_MOVE");

    await page.unroute("**/api/v1/games/*/moves");
    await clickNewGame(page);
    await expect(page.getByTestId("game-error")).toBeEmpty();
    await expect(page.getByTestId("game-status")).toContainText("Status: active");
  });
});

test.describe("Pawn movement", () => {
  test("E2E-020-021: legal black move returns to human turn after CPU reply", async ({ page }) => {
    await page.goto("/");
    await startGame(page, { humanColor: "black" });

    await movePawn(page, 0, 4, 1, 4);
    await expect(page.getByTestId("game-error")).toBeEmpty();
    await expectPawnAt(page, "black", 1, 4);
    await waitForHumanTurn(page);
  });

  test("E2E-022: clicking opponent pawn does not send a move", async ({ page }) => {
    await page.goto("/");
    await startGame(page, { humanColor: "black" });

    await pawnAt(page, 8, 4).click();
    await expectPawnAt(page, "black", 0, 4);
    await expectPawnAt(page, "white", 8, 4);
  });

  test("E2E-023: clicking non-adjacent cell does not send a move", async ({ page }) => {
    await page.goto("/");
    await startGame(page, { humanColor: "black" });

    await pawnAt(page, 0, 4).click();
    await boardCell(page, 2, 4).click();
    await expectPawnAt(page, "black", 0, 4);
    await expect(page.getByTestId("game-error")).toBeEmpty();
  });
});

test.describe("Error display", () => {
  test("E2E-040: illegal move shows ILLEGAL_MOVE", async ({ page }) => {
    await page.goto("/");
    await startGame(page, { humanColor: "black" });

    await page.route("**/api/v1/games/*/moves", async (route) => {
      await route.fulfill({
        status: 400,
        contentType: "application/json",
        body: JSON.stringify({
          detail: { error: { code: "ILLEGAL_MOVE", message: "illegal move" } },
        }),
      });
    });

    await movePawn(page, 0, 4, 1, 4);
    await expect(page.getByTestId("game-error")).toContainText("ILLEGAL_MOVE");
    await expectPawnAt(page, "black", 0, 4);
  });
});

test.describe("API integration", () => {
  test("E2E-060: new game API succeeds", async ({ page }) => {
    const responsePromise = page.waitForResponse(
      (resp) => resp.url().includes("/api/v1/games") && resp.request().method() === "POST",
    );

    await page.goto("/");
    await startGame(page);

    const response = await responsePromise;
    expect(response.status()).toBe(201);
  });
});

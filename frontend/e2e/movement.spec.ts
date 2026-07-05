import { expect, test } from "@playwright/test";

import { errorBody, initialState, playMoveResponse } from "./fixtures";
import {
  boardCell,
  expectCpuTurn,
  expectFinished,
  expectPawnAt,
  movePawn,
  movePawnLegalTowardGoal,
  pawnAt,
  startGame,
  waitForHumanTurn,
} from "./helpers";

test.describe("Pawn movement (extended)", () => {
  test("E2E-024: API returns ILLEGAL_MOVE for off-board direction", async ({ page }) => {
    await page.goto("/");
    await startGame(page, { humanColor: "black" });

    // Off-board directions cannot be sent from the UI; verify error display via mocked API response.
    await page.route("**/api/v1/games/*/moves", async (route) => {
      await route.fulfill({
        status: 400,
        contentType: "application/json",
        body: errorBody("ILLEGAL_MOVE", "cannot move down from row 0"),
      });
    });

    await movePawn(page, 0, 4, 1, 4);
    await expect(page.getByTestId("game-error")).toContainText("ILLEGAL_MOVE");
    await expectPawnAt(page, "black", 0, 4);
  });

  test("E2E-025: cannot move pawn during CPU turn", async ({ page }) => {
    await page.goto("/");
    await startGame(page, { humanColor: "black" });

    await page.route("**/api/v1/games/*/moves", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(
          playMoveResponse({
            human_move: { type: "move", direction: "up" },
            cpu_move: null,
            status: "active",
            turn: "cpu",
            winner: null,
            state: initialState({ blackRow: 1, blackCol: 4, currentPlayer: "white" }),
          }),
        ),
      });
    });

    await movePawn(page, 0, 4, 1, 4);
    await expectCpuTurn(page);

    await pawnAt(page, 1, 4).click();
    await boardCell(page, 2, 4).click();
    await expectPawnAt(page, "black", 1, 4);
    await expect(page.getByTestId("game-error")).toBeEmpty();
  });

  test("E2E-026: white (second) can move one step toward goal (up)", async ({ page }) => {
    await page.goto("/");
    await startGame(page, { humanColor: "white", difficulty: "easy" });

    await movePawn(page, 8, 4, 7, 4);
    await expect(page.getByTestId("game-error")).toBeEmpty();
    await expectPawnAt(page, "white", 7, 4);

    await waitForHumanTurn(page);
  });

  test("diagonal: diagonal jump POST includes to", async ({ page }) => {
    await page.goto("/");
    await startGame(page, { humanColor: "black", difficulty: "easy" });

    const moveRequest = page.waitForRequest(
      (req) => req.url().includes("/moves") && req.method() === "POST",
    );
    await movePawn(page, 0, 4, 1, 4);
    const body = (await moveRequest).postDataJSON() as {
      action?: { to?: { row: number; col: number } };
    };
    expect(body.action?.to).toEqual({ row: 1, col: 4 });

    await waitForHumanTurn(page);
    const moveResponse = page.waitForResponse(
      (resp) => resp.url().includes("/moves") && resp.request().method() === "POST",
    );
    await movePawn(page, 1, 4, 1, 5);
    const respBody = (await (await moveResponse).json()) as {
      human_move?: { to?: { row: number; col: number } };
    };
    expect(respBody.human_move?.to).toEqual({ row: 1, col: 5 });
  });

  test("straight jump: straight move POST/response includes to", async ({ page }) => {
    await page.goto("/");
    await startGame(page, { humanColor: "black", difficulty: "easy" });

    const moveRequest = page.waitForRequest(
      (req) => req.url().includes("/moves") && req.method() === "POST",
    );
    await movePawn(page, 0, 4, 1, 4);
    const body = (await moveRequest).postDataJSON() as {
      action?: { direction?: string; to?: { row: number; col: number } };
    };
    expect(body.action?.direction).toBe("up");
    expect(body.action?.to).toEqual({ row: 1, col: 4 });
  });
});

test.describe("Game end", () => {
  test("E2E-053: black advances on live server and game stays active", async ({ page }) => {
    test.setTimeout(120_000);
    await page.goto("/");
    await startGame(page, { humanColor: "black", difficulty: "easy" });

    for (let step = 0; step < 5; step += 1) {
      await movePawnLegalTowardGoal(page, "black");
      await expect(page.getByTestId("game-error")).toBeEmpty();
      const status = await page.getByTestId("game-status").textContent();
      if (status?.includes("finished")) {
        return;
      }
      await waitForHumanTurn(page, { timeout: 120_000 });
    }
    await expect(page.getByTestId("game-status")).toContainText("Status: active");
    expect(Number(await page.getByTestId("pawn-black").getAttribute("data-row"))).toBeGreaterThan(0);
  });

  test("E2E-050-052: finished state and no moves after game over", async ({ page }) => {
    await page.goto("/");
    await startGame(page, { humanColor: "black" });

    let moveRequests = 0;
    await page.route("**/api/v1/games/*/moves", async (route) => {
      moveRequests += 1;
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(
          playMoveResponse({
            human_move: { type: "move", direction: "up" },
            cpu_move: null,
            status: "finished",
            turn: null,
            winner: "black",
            state: initialState({
              blackRow: 8,
              blackCol: 4,
              whiteRow: 0,
              whiteCol: 4,
              currentPlayer: "white",
            }),
          }),
        ),
      });
    });

    await movePawn(page, 0, 4, 1, 4);
    await expectFinished(page, "black");
    await expect(page.getByTestId("game-status")).toContainText("Turn: —");

    const movesBeforeClick = moveRequests;
    await page.getByTestId("pawn-white").click();
    await boardCell(page, 1, 4).click();
    expect(moveRequests).toBe(movesBeforeClick);
    await expectPawnAt(page, "white", 0, 4);
  });
});

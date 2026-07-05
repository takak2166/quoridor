import { expect, test } from "@playwright/test";

import { clickNewGame, startGame } from "./helpers";

test.describe("Difficulty selection", () => {
  test("E2E-015: can start games on Normal and Hard", async ({ page }) => {
    await page.goto("/");

    await startGame(page, { humanColor: "black", difficulty: "normal" });
    await expect(page.getByTestId("game-status")).toContainText("Status: active");

    await page.getByTestId("difficulty").selectOption("hard");
    await clickNewGame(page);
    await expect(page.getByTestId("game-status")).toContainText("Status: active");
    await expect(page.getByTestId("game-status")).toContainText("Turn: human");
  });

  test("E2E: Expert is out of v0.2 GA scope (not in select)", async ({ page }) => {
    await page.goto("/");
    const options = page.getByTestId("difficulty").locator("option");
    await expect(options).toHaveCount(4);
    await expect(page.getByTestId("difficulty").locator('option[value="expert"]')).toHaveCount(0);
  });

  test("health ready reports expert model availability", async ({ page, request }) => {
    const backendPort = process.env.BACKEND_PORT ?? "8000";
    await page.goto("/");
    const resp = await request.get(`http://127.0.0.1:${backendPort}/health/ready`);
    expect(resp.ok()).toBeTruthy();
    const body = await resp.json();
    expect(["mcts", "unavailable"]).toContain(body.effective_ai.expert);
    if (body.models?.expert?.loadable) {
      expect(body.effective_ai.expert).toBe("mcts");
      expect(body.status).toBe("ready");
    }
  });
});

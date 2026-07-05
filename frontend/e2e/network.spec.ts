import { expect, test } from "@playwright/test";

import { startGame } from "./helpers";

const backendPort = process.env.BACKEND_PORT ?? "8000";
const frontendPort = process.env.FRONTEND_PORT ?? "5173";

test.describe("Network & infrastructure", () => {
  test("E2E-061: API is called via same-origin /api proxy", async ({ page }) => {
    const responsePromise = page.waitForResponse(
      (resp) => resp.url().includes("/api/v1/games") && resp.request().method() === "POST",
    );

    await page.goto("/");
    await startGame(page);

    const response = await responsePromise;
    expect(response.url()).toMatch(new RegExp(`^http://127\\.0\\.0\\.1:${frontendPort}/api/v1/games`));
  });

  test("E2E-062: error is shown when backend is unavailable", async ({ page }) => {
    await page.goto("/");

    await page.route("**/api/v1/games", async (route) => {
      if (route.request().method() === "POST") {
        await route.abort("failed");
        return;
      }
      await route.continue();
    });

    await page.getByTestId("new-game").click();
    await expect(page.getByTestId("game-error")).not.toBeEmpty();
  });

  test("E2E-063: health checks return 200", async ({ request }) => {
    const live = await request.get(`http://127.0.0.1:${backendPort}/health/live`);
    expect(live.status()).toBe(200);

    const ready = await request.get(`http://127.0.0.1:${backendPort}/health/ready`);
    expect(ready.status()).toBe(200);
    const body = (await ready.json()) as { status: string; models?: Record<string, boolean> };
    expect(body.status).toBe("ready");
    expect(body.models).toBeDefined();
  });
});

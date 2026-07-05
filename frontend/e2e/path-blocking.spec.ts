import { expect, test } from "@playwright/test";

import { b4BlockState, createGameResponse } from "./fixtures";
import { expectNoWallHotspot, startGame } from "./helpers";

test.describe("Path-blocking walls", () => {
  test("E2E-036: UI does not offer walls that fully block opponent path", async ({ page }) => {
    await page.goto("/");

    await page.route("**/api/v1/games", async (route) => {
      if (route.request().method() !== "POST") {
        await route.continue();
        return;
      }
      await route.fulfill({
        status: 201,
        contentType: "application/json",
        body: JSON.stringify(
          createGameResponse({
            state: b4BlockState(),
            human_color: "black",
            turn: "human",
          }),
        ),
      });
    });

    await startGame(page, { humanColor: "black", difficulty: "easy" });
    await expectNoWallHotspot(page, "vertical", 0, 3);
    await expect(page.getByTestId("game-error")).toBeEmpty();
  });
});

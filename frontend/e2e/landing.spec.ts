import { expect, test } from "@playwright/test";

test.describe("Initial view", () => {
  test("E2E-001: title and new game button are visible", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByRole("heading", { name: "Quoridor" })).toBeVisible();
    await expect(page.getByTestId("new-game")).toBeVisible();
  });

  test("E2E-002: status message before game start", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByTestId("game-status")).toHaveText("Start a new game");
  });

  test("E2E-003: no pawns rendered before game start", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByTestId("pawn-white")).toHaveCount(0);
    await expect(page.getByTestId("pawn-black")).toHaveCount(0);
  });

  test("E2E-004: error area is empty on first load", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByTestId("game-error")).toBeEmpty();
  });
});

import { expect, test } from "@playwright/test";

test.describe("Accessibility & UX", () => {
  test("E2E-080: Tab focuses the new game button", async ({ page }) => {
    await page.goto("/");
    await page.keyboard.press("Tab");
    await page.keyboard.press("Tab");
    await page.keyboard.press("Tab");

    await expect(page.getByTestId("new-game")).toBeFocused();
  });

  test("E2E-081: board and controls visible on mobile viewport", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto("/");

    await expect(page.getByRole("heading", { name: "Quoridor" })).toBeVisible();
    await expect(page.getByTestId("new-game")).toBeVisible();
    await expect(page.getByTestId("board")).toBeVisible();
  });
});

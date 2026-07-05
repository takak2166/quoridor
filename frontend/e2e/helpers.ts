import { expect, type Page } from "@playwright/test";

export async function clickNewGame(page: Page): Promise<void> {
  const responsePromise = page.waitForResponse(
    (resp) => resp.url().includes("/api/v1/games") && resp.request().method() === "POST",
  );
  await page.getByTestId("new-game").click();
  const response = await responsePromise;
  expect(response.status()).toBe(201);
}

export async function startGame(
  page: Page,
  options: { humanColor?: "white" | "black"; difficulty?: string } = {},
): Promise<void> {
  const { humanColor = "black", difficulty = "easy" } = options;
  const responsePromise = page.waitForResponse(
    (resp) => resp.url().includes("/api/v1/games") && resp.request().method() === "POST",
  );
  await page.getByTestId("human-color").selectOption(humanColor);
  await page.getByTestId("difficulty").selectOption(difficulty);
  await page.getByTestId("new-game").click();
  const response = await responsePromise;
  expect(response.status()).toBe(201);
  await expect(page.getByTestId("game-status")).toContainText("Status: active");
}

export function boardCell(page: Page, row: number, col: number) {
  return page.locator(`[data-testid="board-cell"][data-row="${row}"][data-col="${col}"]`);
}

export function wallHotspot(
  page: Page,
  orientation: "horizontal" | "vertical",
  row: number,
  col: number,
) {
  const testId = orientation === "horizontal" ? "wall-h" : "wall-v";
  return page.locator(`[data-testid="${testId}"][data-row="${row}"][data-col="${col}"]`);
}

export function pawnAt(page: Page, row: number, col: number) {
  return page.locator(`[data-testid^="pawn-"][data-row="${row}"][data-col="${col}"]`);
}

export async function movePawn(page: Page, fromRow: number, fromCol: number, toRow: number, toCol: number) {
  const responsePromise = page.waitForResponse(
    (resp) => resp.url().includes("/moves") && resp.request().method() === "POST",
  );
  await pawnAt(page, fromRow, fromCol).click();
  await boardCell(page, toRow, toCol).click();
  await responsePromise;
}

/** Select human pawn and move toward goal using a legal target (handles CPU walls). */
export async function movePawnLegalTowardGoal(page: Page, humanColor: "white" | "black") {
  const pawnId = humanColor === "white" ? "pawn-white" : "pawn-black";
  const row = Number(await page.getByTestId(pawnId).getAttribute("data-row"));
  const col = Number(await page.getByTestId(pawnId).getAttribute("data-col"));

  const responsePromise = page.waitForResponse(
    (resp) => resp.url().includes("/moves") && resp.request().method() === "POST",
  );
  await pawnAt(page, row, col).click();

  const ghosts = page.getByTestId("pawn-ghost");
  await expect(ghosts.first()).toBeVisible();
  const count = await ghosts.count();
  let bestIndex = 0;
  let bestRow = humanColor === "black" ? -1 : 10;
  for (let i = 0; i < count; i += 1) {
    const ghostRow = Number(await ghosts.nth(i).getAttribute("data-row"));
    if (humanColor === "black" && ghostRow > bestRow) {
      bestRow = ghostRow;
      bestIndex = i;
    } else if (humanColor === "white" && ghostRow < bestRow) {
      bestRow = ghostRow;
      bestIndex = i;
    }
  }

  const toRow = Number(await ghosts.nth(bestIndex).getAttribute("data-row"));
  const toCol = Number(await ghosts.nth(bestIndex).getAttribute("data-col"));
  await boardCell(page, toRow, toCol).click();
  await responsePromise;
}

export async function waitForHumanTurn(page: Page, options?: { timeout?: number }) {
  await expect(page.getByTestId("game-status")).toContainText("Turn: human", {
    timeout: options?.timeout ?? 60_000,
  });
}

export async function expectPawnAt(page: Page, color: "white" | "black", row: number, col: number) {
  const pawn = page.getByTestId(color === "white" ? "pawn-white" : "pawn-black");
  await expect(pawn).toHaveAttribute("data-row", String(row));
  await expect(pawn).toHaveAttribute("data-col", String(col));
}

export async function placeWall(
  page: Page,
  orientation: "horizontal" | "vertical",
  row: number,
  col: number,
) {
  const responsePromise = page.waitForResponse(
    (resp) => resp.url().includes("/moves") && resp.request().method() === "POST",
  );
  await wallHotspot(page, orientation, row, col).first().evaluate((el) => {
    el.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
  });
  await responsePromise;
}

export async function expectCpuTurn(page: Page) {
  await expect(page.getByTestId("game-status")).toContainText("Turn: cpu");
}

export async function expectNoWallHotspot(
  page: Page,
  orientation: "horizontal" | "vertical",
  row: number,
  col: number,
) {
  await expect(wallHotspot(page, orientation, row, col)).toHaveCount(0);
}

export async function expectFinished(page: Page, winner: "white" | "black") {
  await expect(page.getByTestId("game-status")).toContainText("Status: finished");
  await expect(page.getByTestId("game-status")).toContainText(`Winner: ${winner}`);
}

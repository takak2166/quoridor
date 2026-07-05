import { defineConfig, devices } from "@playwright/test";

const backendPort = process.env.BACKEND_PORT ?? "8000";
const frontendPort = process.env.FRONTEND_PORT ?? "5173";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: process.env.CI ? "github" : "list",
  timeout: 60_000,
  use: {
    baseURL: `http://127.0.0.1:${frontendPort}`,
    trace: "on-first-retry",
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        ...(process.env.CI ? {} : { channel: "chrome" as const }),
      },
    },
    {
      name: "mobile-chrome",
      use: {
        ...devices["Pixel 7"],
        ...(process.env.CI ? {} : { channel: "chrome" as const }),
      },
    },
  ],
  webServer: [
    {
      command: `cd ../backend && QUORIDOR_RATE_LIMIT_GAMES_PER_MIN=1000 QUORIDOR_RATE_LIMIT_MOVES_PER_MIN=1000 uv run uvicorn app.main:app --port ${backendPort}`,
      url: `http://127.0.0.1:${backendPort}/health/live`,
      reuseExistingServer: !(process.env.CI === "true" || process.env.E2E_FRESH_SERVER === "1"),
      timeout: 120_000,
    },
    {
      command: `pnpm dev --port ${frontendPort} --host 127.0.0.1`,
      url: `http://127.0.0.1:${frontendPort}`,
      reuseExistingServer: !(process.env.CI === "true" || process.env.E2E_FRESH_SERVER === "1"),
      timeout: 120_000,
    },
  ],
});

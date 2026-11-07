import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["src/**/*.test.ts", "envPorts.test.ts"],
    exclude: ["e2e/**"],
  },
});

import { afterEach, describe, expect, it } from "vitest";

import { readPortEnv } from "./envPorts";

describe("readPortEnv", () => {
  const key = "TEST_PORT_ENV";

  afterEach(() => {
    delete process.env[key];
  });

  it("uses default when unset", () => {
    delete process.env[key];
    expect(readPortEnv(key, "8000")).toBe("8000");
  });

  it("uses default for empty string", () => {
    process.env[key] = "";
    expect(readPortEnv(key, "8000")).toBe("8000");
  });

  it("uses default for whitespace-only", () => {
    process.env[key] = "   ";
    expect(readPortEnv(key, "8000")).toBe("8000");
  });

  it("returns trimmed non-default port", () => {
    process.env[key] = " 8001 ";
    expect(readPortEnv(key, "8000")).toBe("8001");
  });
});

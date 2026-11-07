/** Port from env: unset, empty, or whitespace-only → default (e.g. `BACKEND_PORT=` from Make). */
export function readPortEnv(name: string, defaultPort: string): string {
  const raw = process.env[name];
  if (raw == null) return defaultPort;
  const trimmed = raw.trim();
  return trimmed === "" ? defaultPort : trimmed;
}

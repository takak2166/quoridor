#!/usr/bin/env bash
# Regression: empty/whitespace Make CLI overrides must not pass blank --port values.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

assert_contains() {
  local haystack="$1"
  local needle="$2"
  local label="$3"
  if [[ "$haystack" != *"$needle"* ]]; then
    echo "$haystack" >&2
    fail "$label (expected substring: $needle)"
  fi
}

assert_not_contains() {
  local haystack="$1"
  local needle="$2"
  local label="$3"
  if [[ "$haystack" == *"$needle"* ]]; then
    echo "$haystack" >&2
    fail "$label (forbidden substring: $needle)"
  fi
}

dry_backend() {
  env -u BACKEND_PORT -u FRONTEND_PORT make -n dev-backend "$@" 2>&1 | grep uvicorn || true
}

dry_frontend() {
  env -u BACKEND_PORT -u FRONTEND_PORT make -n dev-frontend "$@" 2>&1 | grep 'pnpm dev' || true
}

dry_e2e() {
  env -u BACKEND_PORT -u FRONTEND_PORT make -n test-e2e "$@" 2>&1 | grep 'test:e2e' || true
}

# Defaults
assert_contains "$(dry_backend)" '--port 8000' 'dev-backend default'
assert_contains "$(dry_frontend)" 'BACKEND_PORT=8000' 'dev-frontend default BACKEND_PORT'
assert_contains "$(dry_frontend)" '--port 5173' 'dev-frontend default FRONTEND_PORT'

# Env empty / whitespace → normalized (unchanged behavior)
assert_contains "$(env -u BACKEND_PORT BACKEND_PORT= make -n dev-backend 2>&1 | grep uvicorn)" '--port 8000' 'env BACKEND_PORT='
assert_contains "$(dry_backend BACKEND_PORT=)" '--port 8000' 'cmdline BACKEND_PORT='
assert_contains "$(dry_backend BACKEND_PORT='   ')" '--port 8000' 'cmdline BACKEND_PORT whitespace'

assert_contains "$(dry_frontend FRONTEND_PORT=)" '--port 5173' 'cmdline FRONTEND_PORT='
assert_contains "$(dry_frontend FRONTEND_PORT='  ')" '--port 5173' 'cmdline FRONTEND_PORT whitespace'

# Non-default
assert_contains "$(dry_backend BACKEND_PORT=9001)" '--port 9001' 'cmdline BACKEND_PORT=9001'
assert_contains "$(dry_frontend BACKEND_PORT=9001 FRONTEND_PORT=3000)" 'BACKEND_PORT=9001' 'cmdline ports'
assert_contains "$(dry_frontend BACKEND_PORT=9001 FRONTEND_PORT=3000)" '--port 3000' 'cmdline FRONTEND_PORT=3000'

assert_contains "$(dry_e2e BACKEND_PORT=)" 'BACKEND_PORT=8000' 'test-e2e cmdline BACKEND_PORT='
assert_not_contains "$(dry_e2e BACKEND_PORT=)" 'BACKEND_PORT= ' 'test-e2e must not pass blank BACKEND_PORT'

echo "OK: make dev port overrides"

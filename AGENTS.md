# Agent Guidelines

<!-- Do not restructure or delete sections. Update individual values in-place when they change. -->

## Project Overview

**Project type:** Human vs CPU Quoridor web app (FastAPI API + Vite/TypeScript UI + Minimax / PPO / MCTS)
**Primary language:** Python 3.12 (backend) + TypeScript (frontend, Node 20)
**Key dependencies:** uv + FastAPI/uvicorn/Pydantic/NumPy (+ optional RL: Gymnasium, SB3, torch); pnpm + Vite/Vitest/Playwright — see `backend/pyproject.toml`, `frontend/package.json`

---

## Commands

```bash
# Development
make setup && make dev   # :8000 backend, :5173 frontend

# Testing / lint
make lint && make test-fast   # ruff + pytest (not slow); make ci ≈ lint + test-fast + build
make test-e2e                 # Playwright (needs make install-e2e)

# Docs
README.md                     # make help for full targets
```

---

## Code Conventions

- Follow the existing patterns in the codebase
- Prefer explicit over clever
- Delete dead code immediately
- Prefer `make` targets over ad-hoc uv/pnpm; game moves API needs `X-Quoridor-Session`
- Keep domain rules in `backend/quoridor/`; wire AI/RL via `app/ports` + `app/infrastructure`

---

## Architecture

```
backend/quoridor/           Pure game domain (rules, pathfinding, moves)
backend/app/api/            FastAPI routes (games, health)
backend/app/services/       Application services
backend/app/ports/          Repository / AI policy interfaces
backend/app/infrastructure/ Persistence, AI (minimax/MCTS/PPO), RL training
backend/tests/              Unit / integration / slow pytest
frontend/src/               Vite UI (board, rules mirror, API client)
frontend/e2e/               Playwright E2E
models/                     Trained PPO checkpoints
```

---

## Core Principles

- **Line budget:** Non-blank, non-HTML-comment lines = instruction body (target **30–50**). Whole file ≤**75** lines. Offload depth to `docs/`.
- Prefer deleting dead instructions over accumulating caveats.

## Maintenance Notes

1. Remove leftover `[bracket]` / TBD placeholders once filled
2. Update Commands when workflows change; rewrite Architecture on major layout changes
3. Delete anything the agent can infer from code

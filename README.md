# Quoridor

Human vs CPU Quoridor web app (FastAPI + Vite/TypeScript + Minimax / PPO / MCTS).

## Prerequisites

- Python 3.12 + [uv](https://docs.astral.sh/uv/)
- Node.js **20 LTS** + pnpm
- On first run, `make setup` copies `.env.example` to `.env`

Run `make help` for the full target list.

## Setup

```bash
make setup      # install dependencies + create .env
```

## Run

```bash
make dev        # start backend (:8000) and frontend (:5173) in parallel
```

Open http://localhost:5173 in your browser.

Individual servers:

```bash
make dev-backend
make dev-frontend
```

## API Notes

- `GET /api/v1/games/{id}`, `POST /api/v1/games/{id}/moves`, `DELETE /api/v1/games/{id}` require the `X-Quoridor-Session` header.
- `GET /health/live` is intended for public liveness checks.
- `GET /health/ready` and `GET /metrics` are intended for internal network access only.

## Tests & CI

```bash
make lint           # ruff check
make test-fast      # pytest (excluding slow)
make test-slow      # performance benchmarks, etc.
make test-all       # all pytest targets
make build          # frontend production build
make ci             # lint + test-fast + build
make install-e2e    # Playwright dependencies
make test-e2e       # E2E (includes E2E_FRESH_SERVER=1)
make ci-full        # ci + test-slow + test-e2e
make release-gate   # ci-full + self-play gates x2
```

## Training & evaluation

```bash
make install-rl     # RL dependencies (first time only)

# Hard model -> models/quoridor_ppo_v1.zip (trains vs Normal minimax by default)
make train-ppo TIMESTEPS=200000

# Expert model -> models/quoridor_ppo_best.zip
make train-ppo-expert TIMESTEPS=300000

# Hard vs Normal win-rate gate (target >70%)
make eval-selfplay GAMES=100 DIFF_A=hard DIFF_B=normal MIN_WIN_RATE=0.70 MAX_P99_MS=500
```

## Other

```bash
make format         # ruff format + fix
make typecheck      # mypy (informational)
make clean          # remove caches
make clean-all      # also remove .venv / node_modules
```

.DEFAULT_GOAL := help

BACKEND_DIR   := backend
FRONTEND_DIR  := frontend
UV            := uv
PNPM          := pnpm
BACKEND_PORT  ?= 8000
FRONTEND_PORT ?= 5173

.PHONY: help
help: ## List available targets
	@grep -E '^[a-zA-Z0-9_.-]+:.*##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# --- Setup ---

.PHONY: install
install: install-backend install-frontend ## Install backend + frontend dependencies

.PHONY: install-backend
install-backend: ## uv sync backend dependencies (includes dev)
	cd $(BACKEND_DIR) && $(UV) sync --extra dev

FRONTEND_STAMP := $(FRONTEND_DIR)/node_modules/.installed

.PHONY: install-frontend
install-frontend: ## pnpm install frontend deps (skip if already present)
	@if [ -f "$(FRONTEND_STAMP)" ] || [ -d "$(FRONTEND_DIR)/node_modules/vite" ]; then \
		mkdir -p "$(FRONTEND_DIR)/node_modules" && touch "$(FRONTEND_STAMP)"; \
	else \
		cd $(FRONTEND_DIR) && CI=true $(PNPM) install --frozen-lockfile; \
		mkdir -p "$(FRONTEND_DIR)/node_modules" && touch "$(FRONTEND_STAMP)"; \
	fi

.PHONY: install-rl
install-rl: ## Install RL extras (sb3-contrib, CPU torch, etc.)
	cd $(BACKEND_DIR) && $(UV) sync --extra dev --extra rl

.PHONY: setup
setup: install ## Copy .env.example to .env when missing
	@test -f .env || cp .env.example .env

# --- Dev servers ---

.PHONY: dev
dev: ## Run backend + frontend in parallel
	$(MAKE) -j2 dev-backend dev-frontend

.PHONY: dev-backend
dev-backend: install-rl ## FastAPI dev server (:$(BACKEND_PORT))
	cd $(BACKEND_DIR) && $(UV) run uvicorn app.main:app --reload --port $(BACKEND_PORT)

.PHONY: dev-frontend
dev-frontend: install-frontend ## Vite dev server (:$(FRONTEND_PORT))
	cd $(FRONTEND_DIR) && $(PNPM) dev --port $(FRONTEND_PORT) --host

# --- Tests & quality ---

.PHONY: test
test: test-fast ## Alias: fast tests (PR CI equivalent)

.PHONY: test-fast
test-fast: install-backend ## pytest (excluding slow)
	cd $(BACKEND_DIR) && $(UV) run pytest -m "not slow" -q

.PHONY: test-slow
test-slow: install-backend ## pytest slow only (benchmarks, self-play)
	cd $(BACKEND_DIR) && $(UV) run pytest -m slow -q

.PHONY: test-all
test-all: install-backend ## pytest (all tests)
	cd $(BACKEND_DIR) && $(UV) run pytest -q

.PHONY: lint
lint: install-backend ## ruff check
	cd $(BACKEND_DIR) && $(UV) run ruff check .

.PHONY: typecheck
typecheck: install-backend ## mypy (informational)
	cd $(BACKEND_DIR) && $(UV) run mypy app quoridor

.PHONY: format
format: install-backend ## ruff format (auto-format)
	cd $(BACKEND_DIR) && $(UV) run ruff format .
	cd $(BACKEND_DIR) && $(UV) run ruff check --fix .

.PHONY: build
build: install-frontend ## frontend production build
	cd $(FRONTEND_DIR) && $(PNPM) build

.PHONY: ci
ci: lint test-fast build ## CI equivalent (ruff + fast test + frontend build)

.PHONY: ci-full
ci-full: ci test-slow test-e2e ## nightly / pre-release: ci + slow + E2E (incl. mobile)

.PHONY: release-gate
release-gate: ci-full ensure-models ## v0.2 release gate: ci-full + self-play gates x2
	$(MAKE) eval-selfplay GAMES=100 DIFF_A=normal DIFF_B=easy MIN_WIN_RATE=0.55 MAX_P99_MS=500
	$(MAKE) eval-selfplay GAMES=100 DIFF_A=normal DIFF_B=hard MAX_P99_MS=3000

.PHONY: ensure-models
ensure-models: ## Ensure PPO model files exist (expert falls back to hard copy)
	@test -f models/quoridor_ppo_v1.zip || (echo "Missing models/quoridor_ppo_v1.zip — run: make train-ppo" && exit 1)
	@test -f models/quoridor_ppo_best.zip || cp models/quoridor_ppo_v1.zip models/quoridor_ppo_best.zip

.PHONY: install-e2e
install-e2e: install-rl ensure-models install-frontend ## E2E deps including Playwright browser + RL models
	cd $(FRONTEND_DIR) && CI=true $(PNPM) install --frozen-lockfile
	-cd $(FRONTEND_DIR) && $(PNPM) exec playwright install chromium

.PHONY: test-e2e
test-e2e: install-e2e ## Playwright E2E tests
	cd $(FRONTEND_DIR) && E2E_FRESH_SERVER=1 BACKEND_PORT=$(BACKEND_PORT) FRONTEND_PORT=$(FRONTEND_PORT) $(PNPM) test:e2e

# --- RL / evaluation ---

.PHONY: train-ppo
train-ppo: install-rl ## Train MaskablePPO Hard model (override TIMESTEPS, OPPONENT=normal)
	cd $(BACKEND_DIR) && $(UV) run python -m app.infrastructure.rl.train_ppo \
		--timesteps $(or $(TIMESTEPS),200000) \
		--opponent $(or $(OPPONENT),normal) \
		--output $(or $(OUTPUT),../models/quoridor_ppo_v1.zip)

.PHONY: train-ppo-expert
train-ppo-expert: install-rl ## Train Expert model (override TIMESTEPS, OPPONENT=normal)
	cd $(BACKEND_DIR) && $(UV) run python -m app.infrastructure.rl.train_ppo \
		--timesteps $(or $(TIMESTEPS),300000) \
		--opponent $(or $(OPPONENT),normal) \
		--output $(or $(OUTPUT),../models/quoridor_ppo_best.zip)

.PHONY: eval-selfplay
eval-selfplay: install-rl ## Self-play eval (override GAMES, MIN_WIN_RATE=0.55)
	cd $(BACKEND_DIR) && $(UV) run python -m app.infrastructure.rl.eval_selfplay \
		--games $(or $(GAMES),100) \
		--difficulty-a $(or $(DIFF_A),easy) \
		--difficulty-b $(or $(DIFF_B),normal) \
		$(if $(MIN_WIN_RATE),--min-win-rate $(MIN_WIN_RATE),) \
		$(if $(MAX_P99_MS),--max-p99-ms $(MAX_P99_MS),)

# --- Cleanup ---

.PHONY: clean
clean: ## Remove caches and build artifacts
	rm -rf $(BACKEND_DIR)/.pytest_cache $(BACKEND_DIR)/.ruff_cache $(BACKEND_DIR)/.mypy_cache
	find $(BACKEND_DIR) -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf $(FRONTEND_DIR)/dist $(FRONTEND_DIR)/node_modules/.installed

.PHONY: clean-all
clean-all: clean ## clean + remove .venv / node_modules
	rm -rf $(BACKEND_DIR)/.venv $(FRONTEND_DIR)/node_modules

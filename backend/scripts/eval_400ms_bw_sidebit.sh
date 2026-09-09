#!/usr/bin/env bash
# Evaluate a Hard zip vs factory Normal. Default is the second-player-bit joint BC.
set -euo pipefail
cd /home/ubuntu/quoridor/backend
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export QUORIDOR_MODEL_HARD="${QUORIDOR_MODEL_HARD:-../models/finetune_bw_sidebit/model.zip}"
GAMES="${1:-16}"
SEED="${2:-97}"
LOG_DIR="$(dirname "$QUORIDOR_MODEL_HARD")"
mkdir -p "$LOG_DIR"
exec python -u -m app.infrastructure.rl.eval_selfplay \
  --games "$GAMES" \
  --difficulty-a hard \
  --difficulty-b normal \
  --max-moves 400 \
  --progress \
  --seed "$SEED" \
  2>&1 | tee "${LOG_DIR}/eval_vs_normal_${GAMES}.log"

#!/usr/bin/env bash
# Evaluate a Hard zip vs factory Normal.
# Match the zip: sidebit was trained with QUORIDOR_SECOND_PLAYER_OBS_BIT=true.
# For models/finetune_bw_nobit/model.zip, set the var to false (Settings default).
set -euo pipefail
cd /home/ubuntu/quoridor/backend
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export QUORIDOR_MODEL_HARD="${QUORIDOR_MODEL_HARD:-../models/finetune_bw_sidebit/model.zip}"
export QUORIDOR_SECOND_PLAYER_OBS_BIT="${QUORIDOR_SECOND_PLAYER_OBS_BIT:-true}"
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

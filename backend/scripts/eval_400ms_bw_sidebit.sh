#!/usr/bin/env bash
# Evaluate the second-player-bit joint BC zip as Hard vs factory Normal.
set -euo pipefail
cd /home/ubuntu/quoridor/backend
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export QUORIDOR_MODEL_HARD="${QUORIDOR_MODEL_HARD:-../models/finetune_bw_sidebit/model.zip}"
GAMES="${1:-16}"
SEED="${2:-97}"
mkdir -p ../models/finetune_bw_sidebit
exec python -u -m app.infrastructure.rl.eval_selfplay \
  --games "$GAMES" \
  --difficulty-a hard \
  --difficulty-b normal \
  --max-moves 400 \
  --progress \
  --seed "$SEED" \
  2>&1 | tee ../models/finetune_bw_sidebit/eval_vs_normal_${GAMES}.log

#!/usr/bin/env bash
# Evaluate the 400ms pawn-first BC/PPO zip as Hard vs factory Normal.
set -euo pipefail
cd /home/ubuntu/quoridor/backend
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export QUORIDOR_MODEL_HARD="${QUORIDOR_MODEL_HARD:-../models/finetune_black_400ms_pawn/model.zip}"
GAMES="${1:-20}"
SEED="${2:-97}"
mkdir -p ../models/finetune_black_400ms_pawn
exec python -u -m app.infrastructure.rl.eval_selfplay \
  --games "$GAMES" \
  --difficulty-a hard \
  --difficulty-b normal \
  --max-moves 400 \
  --progress \
  --seed "$SEED" \
  2>&1 | tee ../models/finetune_black_400ms_pawn/eval_vs_normal_${GAMES}.log

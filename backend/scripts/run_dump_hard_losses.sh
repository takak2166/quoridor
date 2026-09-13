#!/usr/bin/env bash
# Dump Hard (sidebit) losses vs factory Normal for DAgger prefixes.
set -euo pipefail
cd /home/ubuntu/quoridor/backend
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export QUORIDOR_SECOND_PLAYER_OBS_BIT=true
export QUORIDOR_MODEL_HARD="${QUORIDOR_MODEL_HARD:-../models/finetune_bw_sidebit/model.zip}"
GAMES="${1:-24}"
SEED="${2:-97}"
OUT="${3:-artifacts/hard_losses_sidebit}"
mkdir -p "$OUT"
exec python -u -m app.infrastructure.rl.eval_selfplay \
  --games "$GAMES" \
  --difficulty-a hard \
  --difficulty-b normal \
  --max-moves 400 \
  --progress \
  --seed "$SEED" \
  --scoresheet-dir "$OUT" \
  --scoresheet-losses-only \
  2>&1 | tee "$OUT/dump.log"

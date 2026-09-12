#!/usr/bin/env bash
# Hunt node-limited Black wins from first uncovered race errors (64-move loop).
set -euo pipefail
cd /home/ubuntu/quoridor/backend
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export QUORIDOR_SECOND_PLAYER_OBS_BIT=true
LOSS_DIR="${1:-artifacts/hard_losses_sidebit}"
mkdir -p artifacts/black_wins_vs_400ms/black64
exec python -u scripts/dagger_from_losses.py \
  --loss-dir "$LOSS_DIR" \
  --skip-white \
  --black-race-errors \
  --black-out artifacts/black_wins_vs_400ms/black64 \
  --max-black-prefixes 4 \
  --repeats 3 \
  --max-black-wins 8 \
  --max-moves 200 \
  2>&1 | tee "$LOSS_DIR/black64_hunt.log"

#!/usr/bin/env bash
# Hunt node-limited wins from Hard's first off-book / uncovered prefixes.
set -euo pipefail
cd /home/ubuntu/quoridor/backend
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export QUORIDOR_SECOND_PLAYER_OBS_BIT=true
LOSS_DIR="${1:-artifacts/hard_losses_sidebit}"
exec python -u scripts/dagger_from_losses.py \
  --loss-dir "$LOSS_DIR" \
  --white-out artifacts/white_wins_vs_400ms/dagger \
  --black-out artifacts/black_wins_vs_400ms/dagger \
  2>&1 | tee "$LOSS_DIR/hunt.log"

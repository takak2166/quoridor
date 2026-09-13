#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/quoridor/backend
source .venv/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p ../models
exec python -u -m app.infrastructure.rl.eval_selfplay \
  --games 100 \
  --difficulty-a hard \
  --difficulty-b normal \
  --min-win-rate 0.70 \
  --max-p99-ms 3000 \
  --progress \
  --seed 97 \
  2>&1 | tee ../models/eval_hard_vs_normal_100.log

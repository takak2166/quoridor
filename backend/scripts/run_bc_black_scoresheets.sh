#!/usr/bin/env bash
# Behavior-clone saved Black-win scoresheets (vs Normal second) onto current Hard PPO.
set -euo pipefail
cd /home/ubuntu/quoridor/backend
source .venv/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p ../models/finetune_black_scoresheet_bc/checkpoints
exec python -u -m app.infrastructure.rl.train_ppo \
  --resume ../models/finetune_black_imitation/model.zip \
  --white-demo-wins 0 \
  --black-demo-wins 0 \
  --black-demo-scoresheets artifacts/black_wins_vs_normal \
  --black-demo-upsample-m14 8 \
  --black-demo-epochs 8 \
  --bc-only \
  --curriculum "" \
  --opponent normal \
  --timesteps 1 \
  --n-envs 1 \
  --vec-env dummy \
  --smoke-games 0 \
  --potential-scale 8 \
  --opening-wall-free-plies 2 \
  --output ../models/finetune_black_scoresheet_bc/model.zip \
  --checkpoint-dir ../models/finetune_black_scoresheet_bc/checkpoints \
  --tb-log runs/quoridor_finetune_black_scoresheet_bc \
  2>&1 | tee ../models/finetune_black_scoresheet_bc/bc.log

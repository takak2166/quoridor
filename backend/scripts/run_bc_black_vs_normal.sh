#!/usr/bin/env bash
# Behavior-clone Normal vs Normal first-player wins onto the current Hard PPO.
set -euo pipefail
cd /home/ubuntu/quoridor/backend
source .venv/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p ../models/finetune_black_vs_normal/checkpoints
exec python -u -m app.infrastructure.rl.train_ppo \
  --resume ../models/finetune_black_imitation/model.zip \
  --white-demo-wins 0 \
  --black-demo-wins 48 \
  --black-demo-epochs 4 \
  --black-demo-max-games 800 \
  --black-demo-workers 16 \
  --bc-only \
  --curriculum normal \
  --opponent normal \
  --timesteps 1 \
  --n-envs 1 \
  --vec-env dummy \
  --smoke-games 0 \
  --potential-scale 8 \
  --opening-wall-free-plies 2 \
  --output ../models/finetune_black_vs_normal/model.zip \
  --checkpoint-dir ../models/finetune_black_vs_normal/checkpoints \
  --tb-log runs/quoridor_finetune_black_vs_normal \
  2>&1 | tee ../models/finetune_black_vs_normal/bc.log

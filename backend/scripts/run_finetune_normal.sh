#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/quoridor/backend
source .venv/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p ../models/finetune_normal/checkpoints
exec python -u -m app.infrastructure.rl.train_ppo \
  --resume ../models/quoridor_ppo_v1.zip \
  --timesteps 400000 \
  --n-envs 8 \
  --vec-env subproc \
  --curriculum="" \
  --opponent normal \
  --no-opponent-mix \
  --potential-scale 8 \
  --opening-wall-free-plies 2 \
  --revisit-alpha 0.15 \
  --revisit-decay 0.5 \
  --revisit-max-age 4 \
  --max-wall-candidates 10 \
  --smoke-games 8 \
  --smoke-timeout-sec 300 \
  --smoke-workers 4 \
  --smoke-hard-gate-min-win-rate 0.25 \
  --smoke-hard-gate-extend-steps 100000 \
  --smoke-hard-gate-max-extends 3 \
  --output ../models/finetune_normal/model.zip \
  --checkpoint-dir ../models/finetune_normal/checkpoints \
  --checkpoint-freq 10240 \
  --tb-log runs/quoridor_finetune_normal \
  2>&1 | tee ../models/finetune_normal/train.log

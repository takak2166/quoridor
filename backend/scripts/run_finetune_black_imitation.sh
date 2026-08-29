#!/usr/bin/env bash
# Resume finetune_normal with imitation_bonus on both colors (skip white BC).
set -euo pipefail
cd /home/ubuntu/quoridor/backend
source .venv/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p ../models/finetune_black_imitation/checkpoints
exec python -u -m app.infrastructure.rl.train_ppo \
  --resume ../models/finetune_normal/model.zip \
  --white-demo-wins 0 \
  --white-win-ramp \
  --start-stage 4 \
  --timesteps 400000 \
  --imitation-bonus 0.2 \
  --n-envs 8 \
  --vec-env subproc \
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
  --output ../models/finetune_black_imitation/model.zip \
  --checkpoint-dir ../models/finetune_black_imitation/checkpoints \
  --checkpoint-freq 10240 \
  --tb-log runs/quoridor_finetune_black_imitation \
  2>&1 | tee ../models/finetune_black_imitation/train.log

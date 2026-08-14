#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/quoridor/backend
source .venv/bin/activate
export PYTHONUNBUFFERED=1
exec python -u -m app.infrastructure.rl.train_ppo \
  --timesteps 1000000 \
  --n-envs 8 \
  --vec-env subproc \
  --curriculum very_easy,easy,normal \
  --potential-scale 8 \
  --opening-wall-free-plies 2 \
  --revisit-alpha 0.15 \
  --revisit-decay 0.5 \
  --revisit-max-age 4 \
  --max-wall-candidates 10 \
  --smoke-games 4 \
  --smoke-timeout-sec 300 \
  --smoke-workers 4 \
  --smoke-hard-gate-min-win-rate 0.10 \
  --smoke-hard-gate-extend-steps 100000 \
  --smoke-hard-gate-max-extends 3 \
  --output ../models/delta_twowall_scale8/model.zip \
  --checkpoint-dir ../models/delta_twowall_scale8/checkpoints \
  --checkpoint-freq 10240 \
  --tb-log runs/quoridor_delta_twowall_lite \
  2>&1 | tee ../models/delta_twowall_scale8/train.log

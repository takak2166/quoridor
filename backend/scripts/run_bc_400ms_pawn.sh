#!/usr/bin/env bash
# Behavior-clone the 10 pawn-first Black wins vs factory 400ms Normal,
# then fine-tune PPO against the same Normal (no Easy/VeryEasy mix).
# Resume imitation Hard, not the 200-epoch single-sheet BC.
set -euo pipefail
cd /home/ubuntu/quoridor/backend
source .venv/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p ../models/finetune_black_400ms_pawn/checkpoints
exec python -u -m app.infrastructure.rl.train_ppo \
  --resume ../models/finetune_black_imitation/model.zip \
  --white-demo-wins 0 \
  --black-demo-wins 0 \
  --black-demo-scoresheets artifacts/black_wins_vs_400ms/pawn_first \
  --black-demo-upsample-m14 2 \
  --black-demo-epochs 32 \
  --curriculum "" \
  --opponent normal \
  --no-opponent-mix \
  --timesteps 102400 \
  --n-envs 2 \
  --vec-env dummy \
  --agent-white-prob 0.5 \
  --potential-scale 8 \
  --opening-wall-free-plies 2 \
  --smoke-games 8 \
  --smoke-timeout-sec 300 \
  --smoke-workers 1 \
  --smoke-hard-gate-min-win-rate 0.10 \
  --output ../models/finetune_black_400ms_pawn/model.zip \
  --checkpoint-dir ../models/finetune_black_400ms_pawn/checkpoints \
  --checkpoint-freq 10240 \
  --tb-log runs/quoridor_finetune_black_400ms_pawn \
  2>&1 | tee ../models/finetune_black_400ms_pawn/train.log

#!/usr/bin/env bash
# Thicker BC of the 400ms pawn-first book: keep all 10 sheets at 1x, but
# overweight the M(1,4)->M(1,5)->M(2,5) win (the line with H(7,4) at ply 13).
# Skip PPO here; the previous 102k fine-tune erased the opening.
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
  --black-demo-upsample-m14 1 \
  --black-demo-upsample-stem M14_M15_M25 \
  --black-demo-upsample-heavy 12 \
  --black-demo-epochs 80 \
  --bc-only \
  --curriculum "" \
  --opponent normal \
  --timesteps 1 \
  --n-envs 1 \
  --vec-env dummy \
  --smoke-games 0 \
  --potential-scale 8 \
  --opening-wall-free-plies 2 \
  --output ../models/finetune_black_400ms_pawn/model.zip \
  --checkpoint-dir ../models/finetune_black_400ms_pawn/checkpoints \
  --tb-log runs/quoridor_finetune_black_400ms_pawn_heavy \
  2>&1 | tee ../models/finetune_black_400ms_pawn/bc_heavy.log

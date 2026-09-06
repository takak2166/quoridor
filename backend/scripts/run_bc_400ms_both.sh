#!/usr/bin/env bash
# Joint BC: thick Black 400ms book + White wins vs factory 400ms.
# Loop filters live in the policy/env; this script only clones.
set -euo pipefail
cd /home/ubuntu/quoridor/backend
source .venv/bin/activate
export PYTHONUNBUFFERED=1
OUT_DIR="${OUT_DIR:-../models/finetune_bw_400ms_loop}"
mkdir -p "$OUT_DIR/checkpoints"
exec python -u -m app.infrastructure.rl.train_ppo \
  --resume ../models/finetune_black_400ms_pawn/model.zip \
  --white-demo-wins 0 \
  --white-demo-scoresheets artifacts/white_wins_vs_400ms/pawn_first \
  --white-demo-upsample 2 \
  --white-demo-upsample-stem M_7_4_M_2_4_M_6_4 \
  --white-demo-upsample-heavy 6 \
  --white-demo-epochs 80 \
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
  --revisit-alpha 0.25 \
  --revisit-max-age 8 \
  --repeat-pawn-max-visits 1 \
  --output "$OUT_DIR/model.zip" \
  --checkpoint-dir "$OUT_DIR/checkpoints" \
  --tb-log runs/quoridor_finetune_bw_400ms_loop \
  2>&1 | tee "$OUT_DIR/bc_joint.log"

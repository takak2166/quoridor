#!/usr/bin/env bash
# Resume sidebit and teach the Black 64-move race correction (M(2,0) at ply 43).
set -euo pipefail
cd /home/ubuntu/quoridor/backend
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export QUORIDOR_SECOND_PLAYER_OBS_BIT=true
OUT_DIR="${OUT_DIR:-../models/finetune_bw_black64}"
WHITE_SHEETS="artifacts/white_wins_vs_400ms/pawn_first"
# Hunt sheets taught V(4,3) at ply 41 and skipped the M(2,0) state. Labels only.
BLACK_SHEETS="artifacts/black_wins_vs_400ms/pawn_first"
mkdir -p "$OUT_DIR/checkpoints"
mkdir -p artifacts/black_wins_vs_400ms/black64
exec python -u -m app.infrastructure.rl.train_ppo \
  --resume ../models/finetune_bw_sidebit/model.zip \
  --white-demo-wins 0 \
  --white-demo-scoresheets "$WHITE_SHEETS" \
  --white-demo-upsample 2 \
  --white-demo-upsample-stem M_7_4_M_2_4_M_6_4 \
  --white-demo-upsample-heavy 6 \
  --white-demo-epochs 80 \
  --black-demo-wins 0 \
  --black-demo-scoresheets "$BLACK_SHEETS" \
  --black-demo-upsample-m14 1 \
  --black-demo-upsample-stem M14_M15_M25 \
  --black-demo-upsample-heavy 24 \
  --dagger-loss-dir artifacts/hard_losses_sidebit \
  --dagger-focus-repeat 256 \
  --dagger-focus-colors white \
  --dagger-uncovered-colors black \
  --dagger-uncovered-hold-repeat 256 \
  --dagger-uncovered-repeat 512 \
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
  --tb-log runs/finetune_bw_black64 \
  2>&1 | tee "$OUT_DIR/bc_joint.log"

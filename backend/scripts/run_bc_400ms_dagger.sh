#!/usr/bin/env bash
# Resume the sidebit joint BC and add DAgger corrections.
# Mix: pawn_first books + unique dagger wins + off-book teacher focus.
set -euo pipefail
cd /home/ubuntu/quoridor/backend
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export QUORIDOR_SECOND_PLAYER_OBS_BIT=true
OUT_DIR="${OUT_DIR:-../models/finetune_bw_dagger}"
WHITE_SHEETS="artifacts/white_wins_vs_400ms/pawn_first,artifacts/white_wins_vs_400ms/dagger"
BLACK_SHEETS="artifacts/black_wins_vs_400ms/pawn_first,artifacts/black_wins_vs_400ms/dagger"
mkdir -p "$OUT_DIR/checkpoints"
# dagger dir may be empty before the first hunt; create a placeholder skip.
mkdir -p artifacts/white_wins_vs_400ms/dagger artifacts/black_wins_vs_400ms/dagger
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
  --tb-log runs/finetune_bw_dagger \
  2>&1 | tee "$OUT_DIR/bc_joint.log"

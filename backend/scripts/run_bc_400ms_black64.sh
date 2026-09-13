#!/usr/bin/env bash
# Resume the ply-49 zip and clone the p43 hunt suffix (M(4,2) then blocking walls).
set -euo pipefail
cd /home/ubuntu/quoridor/backend
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export QUORIDOR_SECOND_PLAYER_OBS_BIT=true
OUT_DIR="${OUT_DIR:-../models/finetune_bw_black64}"
WHITE_SHEETS="artifacts/white_wins_vs_400ms/pawn_first"
# Full hunt games taught V(4,3) at ply 41. Use p43 suffix labels only.
BLACK_SHEETS="artifacts/black_wins_vs_400ms/pawn_first"
LOSS_DIRS="artifacts/hard_losses_sidebit,artifacts/black64_eval_n16"
FOLLOW_SHEETS="artifacts/black_wins_vs_400ms/black64"
RESUME="${RESUME:-../models/finetune_bw_black64/model_ply49.zip}"
if [[ ! -f "$RESUME" && -f ../models/finetune_bw_black64/model.zip ]]; then
  cp -f ../models/finetune_bw_black64/model.zip "$RESUME"
fi
mkdir -p "$OUT_DIR/checkpoints"
mkdir -p artifacts/black_wins_vs_400ms/black64
exec python -u -m app.infrastructure.rl.train_ppo \
  --resume "$RESUME" \
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
  --dagger-loss-dir "$LOSS_DIRS" \
  --dagger-focus-repeat 256 \
  --dagger-focus-colors white \
  --dagger-uncovered-colors black \
  --dagger-uncovered-hold-repeat 256 \
  --dagger-uncovered-repeat 512 \
  --dagger-follow-sheets "$FOLLOW_SHEETS" \
  --dagger-follow-stem p43-race_error \
  --dagger-follow-repeat 256 \
  --dagger-follow-max 8 \
  --dagger-follow-colors black \
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

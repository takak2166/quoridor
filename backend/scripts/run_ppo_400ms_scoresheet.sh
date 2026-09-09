#!/usr/bin/env bash
# Short PPO after the side-bit joint BC. Do not re-clone (that overwrote the
# opening before). Reward the scoresheet teacher, not greedy race.
#
# 20k steps peaked the book (H(7,4) stayed ~97%, openings rose to ~96%) but
# seed-97 16-game Hard vs factory Normal went 12.5% (BC) -> 0%. Prefer the
# BC zip in models/finetune_bw_sidebit/ until a longer or mix-free run wins.
set -euo pipefail
cd /home/ubuntu/quoridor/backend
source .venv/bin/activate
export PYTHONUNBUFFERED=1
RESUME="${RESUME:-../models/finetune_bw_sidebit/model.zip}"
OUT_DIR="${OUT_DIR:-../models/finetune_bw_sidebit_ppo}"
STEPS="${STEPS:-20480}"
mkdir -p "$OUT_DIR/checkpoints"
exec python -u -m app.infrastructure.rl.train_ppo \
  --resume "$RESUME" \
  --no-bc \
  --white-demo-wins 0 \
  --black-demo-wins 0 \
  --white-demo-scoresheets artifacts/white_wins_vs_400ms/pawn_first \
  --white-demo-upsample-stem M_7_4_M_2_4_M_6_4 \
  --black-demo-scoresheets artifacts/black_wins_vs_400ms/pawn_first \
  --black-demo-upsample-stem M14_M15_M25 \
  --imitation-source scoresheet \
  --imitation-bonus 0.2 \
  --curriculum "" \
  --opponent normal \
  --timesteps "$STEPS" \
  --n-envs 8 \
  --vec-env subproc \
  --smoke-games 0 \
  --potential-scale 8 \
  --opening-wall-free-plies 2 \
  --revisit-alpha 0.25 \
  --revisit-max-age 8 \
  --repeat-pawn-max-visits 1 \
  --output "$OUT_DIR/model.zip" \
  --checkpoint-dir "$OUT_DIR/checkpoints" \
  --checkpoint-freq 10240 \
  --tb-log runs/quoridor_finetune_bw_sidebit_ppo \
  2>&1 | tee "$OUT_DIR/train.log"

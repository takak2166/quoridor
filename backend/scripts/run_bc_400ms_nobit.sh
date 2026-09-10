#!/usr/bin/env bash
# No-bit arm of the joint BC A/B: same mix / resume / epochs as
# run_bc_400ms_both.sh, but obs[134] stays 0 for both colors.
set -euo pipefail
export QUORIDOR_SECOND_PLAYER_OBS_BIT=false
export OUT_DIR="${OUT_DIR:-../models/finetune_bw_nobit}"
exec bash /home/ubuntu/quoridor/backend/scripts/run_bc_400ms_both.sh

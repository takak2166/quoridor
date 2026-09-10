#!/usr/bin/env bash
# No-bit arm of the joint BC A/B: same mix / resume / epochs as
# run_bc_400ms_both.sh, but obs[134] stays 0 for both colors.
#
# seed 97 / 16 games vs factory Normal (timeout 0):
#   sidebit 12.50% (Black 1/8 of 61-ply, White 1/8 of 52-ply)
#   nobit    6.25% (Black 1/8 of 61-ply, White 0/8 of 49-ply)
# Opening softmax is essentially tied (H(7,4) ~97% both; White M(7,4) 66% vs 63%).
set -euo pipefail
export QUORIDOR_SECOND_PLAYER_OBS_BIT=false
export OUT_DIR="${OUT_DIR:-../models/finetune_bw_nobit}"
exec bash /home/ubuntu/quoridor/backend/scripts/run_bc_400ms_both.sh

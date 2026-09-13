#!/usr/bin/env bash
# No-bit arm of the joint BC A/B: same mix / resume / epochs as
# run_bc_400ms_both.sh, but obs[134] stays 0 for both colors.
#
# seed 97 vs factory Normal (timeout 0; 400ms search is not seed-stable):
#   n=16: sidebit 12.50% (B 1/8, W 1/8) vs nobit 6.25% (B 1/8, W 0/8)
#   n=64: sidebit 17.19% (B 5/32, W 6/32) vs nobit 7.81% (B 1/32, W 4/32)
# Opening softmax is essentially tied (H(7,4) ~97% both; White M(7,4) 66% vs 63%).
set -euo pipefail
export QUORIDOR_SECOND_PLAYER_OBS_BIT=false
export OUT_DIR="${OUT_DIR:-../models/finetune_bw_nobit}"
exec bash /home/ubuntu/quoridor/backend/scripts/run_bc_400ms_both.sh

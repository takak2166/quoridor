#!/usr/bin/env bash
# Evaluate the no-bit joint BC zip as Hard vs factory Normal.
set -euo pipefail
export QUORIDOR_SECOND_PLAYER_OBS_BIT=false
export QUORIDOR_MODEL_HARD="${QUORIDOR_MODEL_HARD:-../models/finetune_bw_nobit/model.zip}"
exec bash /home/ubuntu/quoridor/backend/scripts/eval_400ms_bw_sidebit.sh "$@"

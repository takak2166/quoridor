#!/usr/bin/env bash
# Evaluate the DAgger joint BC zip (second-player bit on).
set -euo pipefail
export QUORIDOR_SECOND_PLAYER_OBS_BIT=true
export QUORIDOR_MODEL_HARD="${QUORIDOR_MODEL_HARD:-../models/finetune_bw_dagger/model.zip}"
exec bash /home/ubuntu/quoridor/backend/scripts/eval_400ms_bw_sidebit.sh "$@"

#!/usr/bin/env bash
# Evaluate the Black-64 race-correction BC zip.
set -euo pipefail
export QUORIDOR_SECOND_PLAYER_OBS_BIT=true
export QUORIDOR_PPO_MAX_WALL_CANDIDATES="${QUORIDOR_PPO_MAX_WALL_CANDIDATES:-0}"
export QUORIDOR_MODEL_HARD="${QUORIDOR_MODEL_HARD:-../models/finetune_bw_black64/model.zip}"
exec bash /home/ubuntu/quoridor/backend/scripts/eval_400ms_bw_sidebit.sh "$@"

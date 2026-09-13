#!/usr/bin/env bash
# Evaluate the DAgger joint BC zip (second-player bit on).
# Wall-candidate cap 0 = unfiltered legal walls. The ply-14 teacher H(4,0)
# is dropped by the path-affecting top-10 (V(5,0) is rank 1).
set -euo pipefail
export QUORIDOR_SECOND_PLAYER_OBS_BIT=true
export QUORIDOR_PPO_MAX_WALL_CANDIDATES="${QUORIDOR_PPO_MAX_WALL_CANDIDATES:-0}"
export QUORIDOR_MODEL_HARD="${QUORIDOR_MODEL_HARD:-../models/finetune_bw_dagger/model.zip}"
exec bash /home/ubuntu/quoridor/backend/scripts/eval_400ms_bw_sidebit.sh "$@"

#!/usr/bin/env bash
# Sequential Expert (first) vs Normal (second) probe — MCTS budget must not be starved.
set -euo pipefail
cd /home/ubuntu/quoridor/backend
source .venv/bin/activate
export PYTHONUNBUFFERED=1
exec python -u scripts/collect_expert_black_vs_normal.py "$@"

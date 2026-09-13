#!/usr/bin/env bash
# Measure Normal (black / first) vs Normal (white / second) for BC collection.
set -euo pipefail
cd /home/ubuntu/quoridor/backend
source .venv/bin/activate
export PYTHONUNBUFFERED=1
exec python -u scripts/collect_black_wins_vs_normal.py "$@"

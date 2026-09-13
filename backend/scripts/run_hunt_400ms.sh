#!/usr/bin/env bash
# Sequential hunt for Black (first) wins vs factory Normal (400ms).
# Never use workers>1: time-budget starvation fabricates Black wins.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/backend"
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
export PYTHONUNBUFFERED=1
export PYTHONPATH=.
OUT="${1:-artifacts/black_wins_vs_400ms}"
mkdir -p "$OUT" logs
LOG="logs/hunt_400ms.log"

run() {
  local label="$1"
  shift
  echo "=== ${label} $* ===" | tee -a "$LOG"
  python -u scripts/hunt_black_wins_vs_normal.py \
    --out-dir "$OUT" \
    --white-kind factory \
    --workers 1 \
    --max-moves 200 \
    "$@" | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  echo "=== ${label} exit=${rc} ===" | tee -a "$LOG"
  return 0
}

# Fastest / most relevant first. Keep going after a win so we collect variants.
run first-pawns --mode first --pawns-only --repeats 8
run face-white --mode face-white --repeats 4
run asymmetric-expert --mode asymmetric --black-kind expert --repeats 8
run face-wall-pawns --mode face-wall --pawns-only --repeats 2
run pawn-second --mode pawn-second --repeats 1
run first-all --mode first --repeats 1
run face-wall-all --mode face-wall --repeats 1
run first-pawns-deep --mode first --pawns-only --black-kind deep --repeats 2
run asymmetric-deep --mode asymmetric --black-kind deep --repeats 2

if compgen -G "${OUT}/black_win_*.txt" > /dev/null; then
  echo "FOUND Black win(s) vs 400ms Normal:" | tee -a "$LOG"
  ls -l "$OUT"/black_win_*.txt | tee -a "$LOG"
  exit 0
fi
echo "NO Black win vs 400ms Normal in this hunt" | tee -a "$LOG"
exit 1

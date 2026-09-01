#!/usr/bin/env bash
# Sequential hunt for Black (first) wins vs node-limited Normal (second).
# Stops when a scoresheet is saved under artifacts/black_wins_vs_normal.
set -u
cd /home/ubuntu/quoridor/backend
source .venv/bin/activate
export PYTHONUNBUFFERED=1
OUT="artifacts/black_wins_vs_normal"
mkdir -p "$OUT" logs
LOG="logs/hunt_black_wins.log"

have_win() {
  compgen -G "${OUT}/black_win_*.txt" > /dev/null
}

run() {
  local label="$1"
  shift
  if have_win; then
    echo "=== skip ${label}: already have a Black win ===" | tee -a "$LOG"
    return 0
  fi
  echo "=== ${label} $* ===" | tee -a "$LOG"
  python -u scripts/hunt_black_wins_vs_normal.py \
    --out-dir "$OUT" \
    --workers 20 \
    --max-moves 200 \
    --stop-on-win \
    "$@" | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  echo "=== ${label} exit=${rc} ===" | tee -a "$LOG"
  return 0
}

run face-white-greedy --mode face-white --black-kind greedy
run face-white-normal --mode face-white --black-kind normal
run first-pawns-greedy --mode first --pawns-only --black-kind greedy
run first-pawns-normal --mode first --pawns-only --black-kind normal
run first-all-greedy --mode first --black-kind greedy
run pawn-second-greedy --mode pawn-second --black-kind greedy
run pawn-second-normal --mode pawn-second --black-kind normal
run first-all-normal --mode first --black-kind normal
run face-wall-pawns-greedy --mode face-wall --pawns-only --black-kind greedy
run face-wall-pawns-normal --mode face-wall --pawns-only --black-kind normal
run face-wall-all-greedy --mode face-wall --black-kind greedy
run asymmetric-deep --mode asymmetric --black-kind deep --workers 1
run face-white-deep --mode face-white --black-kind deep --workers 2
run first-pawns-deep --mode first --pawns-only --black-kind deep --workers 3

if have_win; then
  echo "FOUND Black win(s):" | tee -a "$LOG"
  ls -l "$OUT"/black_win_*.txt | tee -a "$LOG"
  exit 0
fi
echo "NO Black win found in prefix/asymmetric hunts" | tee -a "$LOG"
exit 1

---
name: hard-eval-gates
description: >-
  Evaluates and updates the official Hard PPO zip against factory Easy and
  Normal self-play gates. Use when changing models/quoridor_ppo_v1.zip,
  running eval_selfplay, behavior cloning, DAgger, hunt scoresheets, or when
  the user mentions Hard vs Easy, Hard vs Normal, 公式 zip, or 勝率ゲート.
---

# Hard eval gates

Official Hard is `models/quoridor_ppo_v1.zip` (MaskablePPO, second-player obs bit on). It is a short-line BC specialist, not a general PPO. Extra BC on shared openings forgets existing win lines.

## Do not

- Overwrite the official zip until **both** 100-game gates pass
- Commit experiment zips, `backend/artifacts/`, or `models/finetune_*`
- Diagnose with `--min-win-rate` (raises `SystemExit` before the summary)
- Treat 16-game smoke as the adoption number (Normal 400ms is stochastic)

## Official gate

Cwd `backend`. Fresh process per zip (`_POLICY_CACHE` is process-wide).

```bash
export PYTHONUNBUFFERED=1
export QUORIDOR_MODEL_HARD=../models/quoridor_ppo_v1.zip
export QUORIDOR_SECOND_PLAYER_OBS_BIT=true
export QUORIDOR_PPO_MAX_WALL_CANDIDATES=0
.venv/bin/python -u -m app.infrastructure.rl.eval_selfplay \
  --games 100 --difficulty-a hard --difficulty-b easy \
  --max-moves 400 --seed 97 --progress --max-p99-ms 3000
.venv/bin/python -u -m app.infrastructure.rl.eval_selfplay \
  --games 100 --difficulty-a hard --difficulty-b normal \
  --max-moves 400 --seed 97 --progress --max-p99-ms 3000
```

Pass: each win rate **≥ 70%**, timeouts 0, P99 ≤ 3000ms.

Game `i` (0-based): even `i` → Hard is Black (1-indexed games 1,3,5…); odd `i` → Hard is White.

16-game smoke is only for "did Easy/Black collapse?". Adopt only from 100-game.

After replacing the official zip, keep README's Hard vs Easy and Hard vs Normal eval examples in sync.

## If a gate fails

1. Keep the official zip unchanged.
2. Prefer **weight soup** of two complementary zips (one holds Easy/Black, one holds White) over more BC.
3. If BC is required, clone only **uncovered** positions. Put extra teacher lines in `--dagger-book-extra` so they are not cloned as full demos. Do not heavy-upsample Easy or mix conflicting labels at the same ply.

Forbidden teachers and soup notes: [reference.md](reference.md)

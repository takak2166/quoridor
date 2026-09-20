# Hard eval gates — reference

## Forbidden / toxic teachers

These overwrite a working line. Do not add them to BC demos or DAgger follow unless a new eval proves otherwise.

| Sheet / pattern | Why it breaks |
| --- | --- |
| `backend/tests/unit/fixtures/black_win_vs_normal_m14.txt` (`H(2,6)` then **`M(2,5)`**) | Official Black wins use **`M(2,7)`** after `H(2,6)` |
| `black_win_vs_normal_h73.txt` (first move `H(7,3)`) | Wipes opening `M(1,4)` |
| Black hunt full games (`H(1,1)` factory 59 / node 75) | Conflicting labels at ply 19; collapses Normal Black |
| White hunt of the **75-ply** loss (greedy) | Spawns 21-ply White losses |
| Easy scoresheets with **heavy** upsample | Drowns Normal branches that share the `M(1,4)` prefix |
| Mixing two different teachers on the same position | Policy averages both and plays neither |

Easy vs Normal Black share `M(1,4)…M(2,6)` then split at ply 8 (`H(3,6)` vs `H(2,6)`). Teach the suffix after the split; do not re-BC the shared prefix without an Easy **and** orig-Normal anchor in the same batch.

## Weight soup

When zip A holds Easy (and/or Black) and zip B holds White, averaging policy weights can beat further BC:

```python
from sb3_contrib import MaskablePPO
a = MaskablePPO.load("path_a.zip")
b = MaskablePPO.load("path_b.zip")
sa, sb = a.policy.state_dict(), b.policy.state_dict()
mixed = {k: 0.5 * sa[k] + 0.5 * sb[k] for k in sa}
out = MaskablePPO.load("path_a.zip")
out.policy.load_state_dict(mixed)
out.save("soup.zip")
```

Smoke 16, then official 100×2. Try nearby mixes (0.3/0.7) only if 0.5 fails a gate.

## DAgger book extra

`--dagger-book-extra` fills Black teacher-book holes **without** cloning that whole game into the BC batch (no overwrite of main-demo positions). Use it so `uncovered` detection starts at the real branch ply (e.g. after `H(1,1)`), then `--dagger-follow-max 1` for a single correction.

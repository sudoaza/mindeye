# Sprint 2 Summary — ZUNA Tight-Window Recovery

Date: 2026-05-14
Detailed analysis: `docs/SPRINT2_ZUNA_TIGHT1S_RECOVERY_ANALYSIS.md`
Matrix directory: `outputs/baseline_matrix/20260514_132502_matrix`

## Verdict

**PASS — proceed to Sprint 3.**

After correcting the global-run/session mapping and scaling to all available event-backed `sub-01` ImageNet runs, `zuna_real` clearly beats both randomized controls.

## Data note

The script requested global runs `1-40`, but OpenNeuro currently exposes only 32 event-backed `sub-01` ImageNet runs (`ImageNet01`-`ImageNet04`, 8 runs each). `ImageNet05/run-01..08` is absent for `sub-01`, so the valid matrix is a 32-run result, not a true 40-run result.

## Gate metrics

Validation set: held-out global run `32`, `n=125`.

| Condition | Top-1 | Top-5 | Top-10 | MRR | Median rank | Collapse score |
|---|---:|---:|---:|---:|---:|---:|
| `zuna_real` | **0.056** | **0.160** | **0.256** | **0.128** | **29** | 4.281 |
| `zuna_shuffled` | 0.024 | 0.056 | 0.112 | 0.060 | 56 | 4.735 |
| `zuna_random` | 0.008 | 0.048 | 0.096 | 0.047 | 59 | 0.418 |
| random expected | 0.008 | 0.040 | 0.080 | n/a | n/a | n/a |

## Gate check

```text
✅ zuna_real vs zuna_shuffled: Top-10 0.256 vs 0.112, MRR 0.128 vs 0.060
✅ zuna_real vs zuna_random:   Top-10 0.256 vs 0.096, MRR 0.128 vs 0.047
✅ collapse_score = 4.281 > 0.1

GATE: PASS
```

## Immediate next action

Proceed to Sprint 3 low-channel simulation using this full-channel result as the anchor/reference.

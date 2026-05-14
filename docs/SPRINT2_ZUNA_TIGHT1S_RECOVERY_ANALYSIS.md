# Sprint 2 ZUNA Tight-Window Recovery Analysis

Date: 2026-05-14
Experiment directory: `outputs/baseline_matrix/20260514_132502_matrix`
Pipeline script: `scripts/execute_recovery_v2.sh`
Gate verdict: **PASS — proceed to Sprint 3**

## Executive summary

The corrected and scaled ZUNA tight-window recovery run passed the Sprint 2 gate. With all event-backed `sub-01` ImageNet runs available in OpenNeuro, `zuna_real` beat both leakage controls by a clear margin on Top-10 retrieval and MRR:

- `zuna_real` Top-10: **0.256** (`32 / 125`)
- `zuna_shuffled` Top-10: **0.112** (`14 / 125`)
- `zuna_random` Top-10: **0.096** (`12 / 125`)
- Random Top-10 expectation for this validation pool: **0.080** (`10 / 125`)

This is the first Sprint 2 result where the real EEG/image pairing is clearly above both shuffled-label and random-target controls after the label leakage fix and tight temporal crop. The result supports continuing to Sprint 3 low-channel simulations instead of immediately pivoting to a foundation-model/MSE-only route.

## Important dataset correction: requested 40, available 32

The recovery script was originally changed to request runs `1-40` as local runs inside `ImageNet01`. That path layout is wrong for NOD/OpenNeuro. The dataset stores ImageNet runs as 8 local runs per session:

| Global run ids | OpenNeuro session/local runs |
|---:|---|
| `1-8` | `ImageNet01/run-01..08` |
| `9-16` | `ImageNet02/run-01..08` |
| `17-24` | `ImageNet03/run-01..08` |
| `25-32` | `ImageNet04/run-01..08` |
| `33-40` | would map to `ImageNet05/run-01..08`, but these files/events are not present for `sub-01` |

The pipeline now still attempts to download global runs `1-40`, so a future `ImageNet05` would be picked up automatically, but it crops/trains/evaluates only the event-backed runs exposed by `sub-01_events.csv`. For the current OpenNeuro `sub-01` dataset, that is **32 runs**.

Patch summary:

- `scripts/download_nod.py`
  - Added global-run to `(session, local_run)` mapping.
  - Default `--runs` behavior now treats `1-40` as global runs across sessions.
  - Added `--session-local-runs` for old/local-session behavior.
- `src/mindseye/zuna/cropper.py`
  - Added the same global-run mapping for raw/ZUNA FIF lookup.
  - Keeps metadata `run` as the **global run id** so run-heldout splits still work.
  - Adds `session`, `local_run`, and `global_run` metadata columns for traceability.
- `scripts/execute_recovery_v2.sh`
  - Downloads requested runs `1-40`.
  - Detects the number of event-backed runs from `sub-01_events.csv`.
  - Crops/trains/evaluates runs `1-$AVAILABLE_RUNS`; current value is `32`.
  - Uses `--val-runs 32` for the held-out run split.

## Pipeline configuration

### Data and preprocessing

- Subject: `sub-01`
- Requested global runs: `1-40`
- Available event-backed global runs: **32**
- Raw continuous FIF files downloaded: **32**
- ZUNA denoised FIF outputs: **32**
- Crop window: **tight 1.2 s**, `tmin=-0.2`, `tmax=1.0`
- Event marker channel: enabled (`--add-event-marker`)
- Cropped output directory: `data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40`
- Total cropped epochs: **3974**
- Training epochs: **3849**
- Validation epochs: **125** (`global_run=32`, `ImageNet04/run-08`)
- EEG tensor shape: **63 channels × 307 samples**
- Sample rate after ZUNA/crop: **256 Hz**

### Embeddings and targets

- Image embeddings: `data/processed/clip_embeddings/sub01_image_embeddings.pt`
- Semantic JSONL: `data/processed/clip_embeddings/image_semantics.jsonl`
- Semantic text embeddings: `data/processed/clip_embeddings/image_semantic_text_embeddings.pt`
- Template label embeddings: `data/processed/clip_embeddings/imagenet_text_embeddings.pt`
- Common target embeddings: `data/processed/clip_embeddings/common_embeddings.pt`
- Common target blend: `w_img=0.25`, `w_sem=0.75`
- Target bank size: **3974**
- Target-bank off-diagonal cosine mean: **0.9546**
- Target-bank off-diagonal cosine std: **0.0080**

The target bank remains very clustered in CLIP/common-embedding space. That makes absolute retrieval hard and makes relative/control comparisons more important than the raw Top-1 number.

### Model/training

Shared config for the full matrix:

- Model: `temporal_attn_small`
- Resolved model size: hidden dim `128`, layers `2`, heads `4`, dropout `0.35`
- Stem dropout1d: `0.15`
- Loss: CLIP-style contrastive loss
- Temperature: `0.07`
- Epochs: `50`
- Batch size: `64`
- Learning rate: `3e-4`
- Weight decay: `0.01`
- Split mode: run-heldout (`--split-mode run`)
- Validation run: global run `32`
- Seed: `13`
- Device: CUDA, NVIDIA GeForce RTX 4090
- PyTorch: `2.6.0+cu124`

Train-time EEG augmentation was enabled:

- Channel dropout: `0.1`
- Noise std: `0.03`
- Amplitude scale jitter: `0.1`
- Time mask: `24`
- Time jitter: `8`

## Matrix results

Validation pool size is `125`; therefore random expected retrieval is Top-1 `0.008`, Top-5 `0.040`, Top-10 `0.080`.

| Condition | Target mode | Top-1 | Top-5 | Top-10 | MRR | Median rank | Best epoch | Collapse score | Mean diag cosine | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `zuna_real` | true image/common target | **0.056** | **0.160** | **0.256** | **0.1277** | **29** | 50 | 4.2805 | 0.0420 | ok |
| `zuna_shuffled` | shuffled labels/control targets | 0.024 | 0.056 | 0.112 | 0.0604 | 56 | 37 | 4.7349 | 0.0055 | ok |
| `zuna_random` | Gaussian random targets | 0.008 | 0.048 | 0.096 | 0.0474 | 59 | 10 | 0.4178 | 0.0004 | ok |

Equivalent validation hit counts:

| Condition | Top-1 hits | Top-5 hits | Top-10 hits |
|---|---:|---:|---:|
| `zuna_real` | **7 / 125** | **20 / 125** | **32 / 125** |
| `zuna_shuffled` | 3 / 125 | 7 / 125 | 14 / 125 |
| `zuna_random` | 1 / 125 | 6 / 125 | 12 / 125 |
| Random expectation | 1 / 125 | 5 / 125 | 10 / 125 |

### Margins over controls

| Comparison | Top-10 absolute margin | Top-10 ratio | MRR absolute margin | MRR ratio | Median-rank improvement |
|---|---:|---:|---:|---:|---:|
| `zuna_real` vs `zuna_shuffled` | **+0.144** | **2.29×** | **+0.0673** | **2.11×** | **56 → 29** |
| `zuna_real` vs `zuna_random` | **+0.160** | **2.67×** | **+0.0803** | **2.69×** | **59 → 29** |
| `zuna_real` vs random expectation | **+0.176** | **3.20×** | n/a | n/a | n/a |

## Gate check

Sprint 2 gate criteria:

1. `zuna_real` must beat `zuna_shuffled` by a healthy margin on Top-10 and MRR.
2. `zuna_real` must beat `zuna_random` by a healthy margin on Top-10 and MRR.
3. Prediction collapse guard must pass (`collapse_score > 0.1`).

Observed gate output:

```text
✅  zuna_real vs zuna_shuffled: top10 0.256 vs 0.112, MRR 0.128 vs 0.060
✅  zuna_real vs zuna_random:   top10 0.256 vs 0.096, MRR 0.128 vs 0.047
✅  collapse_score = 4.281 (need > 0.1)

GATE: PASS — proceed to Sprint 3 ✅
```

## Interpretation

### What the pass means

The previous 8-run tight-window run failed to separate real targets from randomized controls. After fixing the global-run mapping and using all available event-backed runs, the real pairing now separates from controls. That suggests the earlier failure was at least partly a data-scale problem rather than definitive evidence that `temporal_attn_small` cannot learn the mapping.

The shuffled-label control is the most important comparison because it uses the same semantic target distribution but breaks the EEG↔image correspondence. `zuna_real` more than doubled both Top-10 and MRR relative to `zuna_shuffled`. This is the strongest evidence in this run that the model is learning something correspondence-specific rather than merely exploiting the clustered target manifold.

The random-target control is also safely lower. It sits close to random expectation on Top-1 and only slightly above random on Top-10, while `zuna_real` reaches 32 Top-10 hits against 10 expected by chance.

### Why this is not yet a final scientific win

The result is promising but still a Sprint gate, not a final claim:

- Validation is one held-out run (`125` samples). The margins are large enough to proceed, but Sprint 3 should test stability across more held-out runs/folds.
- The common target bank is highly clustered (`off_diag_mean ≈ 0.955`), so raw Top-k scores are constrained by target similarity structure.
- `zuna_real` best epoch is epoch `50`, i.e. still improving at the training horizon. A longer run may improve or overfit; this needs a controlled follow-up.
- Absolute Top-1 remains low (`7/125`), so downstream usefulness should be judged using Top-k/MRR and control separation, not Top-1 alone.
- Only `sub-01` is tested here.

### What changed relative to the failed Sprint 2 matrix

Earlier Sprint 2 matrices were effectively at/near chance:

- The real condition did not consistently beat shuffled/random controls.
- Median ranks were near the middle of the validation pool.
- Controls could match or exceed the real condition.

In this corrected run:

- `zuna_real` median rank improves to `29`, versus `56/59` for controls.
- `zuna_real` Top-10 reaches `25.6%`, versus `11.2%` and `9.6%` for controls.
- The shuffled-label leakage issue is still controlled: shuffled is lower, not falsely competitive.

## Notable log warnings

The ZUNA stage emitted repeated PyTorch symbolic-shape messages:

```text
[rank0]:W... torch/fx/experimental/symbolic_shapes.py:6307] failed during evaluate_expr(Ne(u0, 0), ...)
[rank0]:E... torch/fx/experimental/recording.py:299] failed while running evaluate_expr(*(Ne(u0, 0), None), ...)
```

These came from `torch.compile`/Dynamo internals inside ZUNA. They did **not** terminate the pipeline: all 32 ZUNA FIF outputs were written, the cropper completed, embeddings completed, and the matrix passed. There was no Python traceback or runtime exception associated with those messages.

The downloader also warned for `ImageNet05/run-01..08`. Those warnings are expected because the current `sub-01` detailed-events file exposes only `ImageNet01` through `ImageNet04`.

## Artifact map

Primary result files committed with this analysis:

- Detailed analysis: `docs/SPRINT2_ZUNA_TIGHT1S_RECOVERY_ANALYSIS.md`
- Short Sprint 2 summary: `outputs/baseline_matrix/SPRINT2_SUMMARY.md`
- Matrix summary CSV: `outputs/baseline_matrix/20260514_132502_matrix/matrix_summary.csv`
- Per-condition result directories:
  - `outputs/baseline_matrix/20260514_132502_matrix/20260514_132519_zuna_real_zuna_real_tight1s_recovery/`
  - `outputs/baseline_matrix/20260514_132502_matrix/20260514_133028_zuna_shuffled_zuna_shuffled_tight1s_recovery/`
  - `outputs/baseline_matrix/20260514_132502_matrix/20260514_133537_zuna_random_zuna_random_tight1s_recovery/`

Each per-condition directory contains:

- `config.json`
- `environment.txt`
- `git_commit.txt`
- `history.json`
- `metrics.csv`
- `metrics.json`
- `train_log.csv`

Large checkpoints (`*.pt`, `*.pth`) remain ignored by `.gitignore` and are intentionally not required for reviewing the result.

## Recommended next step: Sprint 3

Proceed to the full low-channel simulation matrix, but keep the current full-channel result as the reference anchor.

Suggested Sprint 3 design:

1. Use the same corrected global-run mapping and `global_run=32` validation split for direct comparability.
2. Keep the exact successful full-channel config as the control/reference:
   - ZUNA real
   - tight window `-0.2..1.0`
   - common targets `w_img=0.25`, `w_sem=0.75`
   - `temporal_attn_small`
   - 50 epochs
   - augmentations enabled
3. Run low-channel masks against the same controls:
   - full 63-channel reference
   - EPOC-14 simulation
   - smaller frontal/temporal/occipital subsets if relevant
4. Report each low-channel condition against both `zuna_shuffled` and `zuna_random`, not just absolute retrieval.
5. Consider adding multi-fold held-out global runs once the Sprint 3 matrix confirms a viable low-channel channel set.

Bottom line: **scaling to all available event-backed runs fixed the Sprint 2 gate. Do not pivot away from the current ZUNA + temporal_attn_small path yet; proceed to Sprint 3.**

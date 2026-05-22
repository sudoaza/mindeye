# MindEye Development Plan — z_common Canonical Latent Pipeline

## 1. Strategic direction

The project must remain **ZUNA-first**. The primary training source is **NOD-EEG** (continuous time series, event timing). Alljoined/ENIGMA will only be used later for robustness/domain-adaptation. ZUNA requires 256Hz, 5-second epochs.

## 2. Non-negotiable technical principles

1. **Do not add diffusion until EEG→CLIP beats controls** (real > shuffled, real > random, nonzero std, meaningful grid).
2. **ZUNA output is the training domain** (ZUNA output → crop → CLIP target).
3. **Timing integrity is critical** (stimulus onset → ZUNA axis → crop).
4. **Cheap-headset path must be simulated now** (full vs simulated 14-channel vs random vs 32-channel).

## 3. Target repository structure

- `configs/` for datasets, zuna, and train yamls.
- `src/mindseye/` restructured into datasets, zuna, embeddings, models, eval, diffusion, utils.
- `scripts/` containing clear modular run targets (download, zuna batch, crop, audit, embeddings, train, baselines, simulate, etc).
- `outputs/` for all run tracking.

## 4. Phase 1 — Make the current pipeline reproducible (Sprint 1)

**Objective**: Make one exact command sequence reproduce the pipeline via `Makefile` and `configs/`.
- [x] Add `configs/datasets/nod_sub01_runs01_05.yaml`.
- [x] Add `configs/zuna/zuna_real_50steps.yaml`.
- [x] Add `configs/train/eeg_clip_contrastive.yaml`.
- [x] Add `Makefile`.

## 5. Phase 2 — Timing and data-integrity audit (Sprint 1)

**Objective**: Ensure event timing is not corrupted.
- [x] Add `scripts/audit_zuna_timing.py` to compare raw FIF vs ZUNA FIF vs metadata.
- [x] ZUNA batch inference running on GPU (RunPod).
- [x] Outputs must be checked before training (Passed).
- [x] 80-epoch contrastive EEG->CLIP training (Passed).
- [x] Generate retrieval grid and verify non-collapsed pred_std.

## 6. Phase 3 — Baseline matrix (Sprint 2)

**Objective**: Run controlled matrix comparing raw, resample-only, ZUNA, shuffled labels, and random targets.
- [x] Finish `scripts/run_baseline_matrix.py` — wired to real training via subprocess.
- [x] Required conditions: `raw_runheldout`, `resample_only_runheldout`, `zuna_runheldout`, `zuna_shuffled_labels`, `zuna_random_targets`, `zuna_sameclass_distractors`.
- [x] Required metrics: top1, top5, top10, MRR, median rank, mean diagonal cosine, off-diagonal cosine mean, prediction std, target std, collapse score = pred_std / target_std, random expected top-k.
- [x] Add `--input-domain` (zuna/raw/resample) and `--target-mode` (real/shuffled/random/sameclass) flags to `train_eeg_clip.py`.
- [x] Every run outputs structured `outputs/runs/YYYYMMDD_HHMMSS_slug/` with `metrics.json`, `metrics.csv`, `train_log.csv`, `best.pt`, `history.json`, `config.json`, `environment.txt`, `git_commit.txt`.
- [x] Run `make matrix` — `zuna_runheldout` must beat all controls.
- [x] **Gate**: `zuna_runheldout` must beat shuffled/random controls. `pred_std` must be nonzero and not collapsed. Retrieval grid must not repeat the same images.
- [x] Fix execute_recovery_v2.sh to skip regenerating the embeddings if they already exist.

## 7. Phase 4 — Low-channel / cheap-headset simulation (Sprint 3)

**Objective**: Test if ZUNA can rescue EPOC-like 14-channel EEG. This remains mandatory and should not be deferred behind better encoders.
- [x] `src/mindseye/zuna/channel_simulation.py` and `scripts/simulate_low_channel_zuna.py` exist.
- [x] Add `configs/zuna/zuna_epoc14_sim.yaml`.
- [x] Simulate paths: EPOC-like 14ch → ZUNA vs raw EPOC-like 14ch without ZUNA.
- [x] Use approximate EPOC X channel list.
- [x] Report: `zuna_gain = metric(epoc14_zuna) / metric(raw_epoc14_nozuna)` and `retention = metric(epoc14_zuna) / metric(full_zuna)`.
  - **Results**:
    - `zuna_real` (EPOC-14 + ZUNA): Top-10 = 0.192, MRR = 0.094
    - `raw_real` (EPOC-14 raw): Top-10 = 0.200, MRR = 0.096
    - `zuna_shuffled` (Control): Top-10 = 0.120, MRR = 0.052
    - `zuna_gain` (MRR): 0.094 / 0.096 = **0.98x** (No significant reconstruction gain over raw masked EPOC-14)
    - `retention` (MRR vs Full ZUNA 0.1277): 0.094 / 0.1277 = **73.9%** (retains ~74% of full 64-ch performance)
- [x] `run_epoc_simulation.sh` runs FIF masking, ZUNA offline upscaling, cropping, and trains/evaluates the baseline matrix.

## 8. Phase 5 — Improve the EEG encoder (Sprint 4)

**Objective**: Upgrade from Conv1D to a Spatial-Temporal Coordinate-Aware Encoder.
- [x] Add `src/mindseye/models/spatial_temporal_encoder.py`.
- [x] Input API: `eeg: [B, C, T]`, `channel_xyz: [B, C, 3]`, `subject_id` (optional), `run_id` (optional).
- [x] Architecture target: temporal convolution / learned filterbank + coordinate embedding MLP + channel attention or transformer + temporal pooling + subject/run adapter + projection to CLIP dimension.
- [x] Keep the Conv1D baseline for comparison.
- [x] Increase dataset, include other subjects' data from the set (sub-01 + sub-02).

## 9. Phase 6 — Multi-domain semantic front (Sprint 5)

**Objective**: Move from CLIP image embeddings to structured semantic state.
- [x] Add structured targets: CLIP image embedding, CLIP text/class embedding, object caption embedding, spatial/composition embedding, color/material embedding, mood/theme embedding, abstract concept embedding, direction/action embedding.
- [x] This is core Neural-MCRL / Semantic-Prompts inspiration. Do not jump to diffusion before this has measurable signal.

## 10. Phase 7 — BReAD-style retrieval branch

**Objective**: Use retrieved image/embedding as grounding for img2img.
- [ ] Add `src/mindseye/embeddings/faiss_index.py`, `scripts/build_retrieval_index.py`, `scripts/retrieve_visual_priors.py`.
- [ ] Implement after the semantic front beats controls.

## 11. Phase 8 — Frozen diffusion img2img prototype (Sprint 6+)

**Objective**: Hook predicted semantic state to SDXL-Turbo/SD3.
- [ ] Do not implement now unless retrieval/semantic gates pass.
- [ ] Required before diffusion: real EEG->CLIP > shuffled, real EEG->semantic heads > shuffled, low-channel simulation has nonzero usable signal, retrieval grids are meaningful.
- [ ] Pipeline: semantic state + retrieved prior + current image -> SDXL-Turbo or similar img2img.
- [ ] Do not start with fine-tuning diffusion. Defer EEG2Vision-style VLM boost.

## 12. Phase 9 — Alljoined / ENIGMA comparison

**Objective**: Use Alljoined for domain adaptation / consumer-grade robustness, not as core.
- [ ] Run comparable baselines against ENIGMA after baseline matrix / low-channel simulation.

## 13. Output and experiment tracking

**Objective**: `outputs/runs/YYYYMMDD_HHMMSS_slug/` structured output tracking with metrics and audits.
- [x] Every run creates: `config.json`, `git_commit.txt`, `environment.txt`, `metrics.json`, `metrics.csv`, `train_log.csv`, `best.pt`, `history.json`.
- [ ] Add `retrieval_grid.png` auto-generation at end of each training run.
- [ ] Add `audit.json` and `notes.md` stubs.

---

## Current Status: Phase 9 — z_common Canonical Latent + Probe Ablation ✅

### Architecture (canonical, do not deviate)
```
image/VLM/attributes → z_common          (frozen CLIP + VLM fused embedding)
EEG  → z_pred_common                     (trained encoder output)
frozen_probe(normalize(z_pred_common))   (semantic regularizer, 10 tasks)
```

> [!NOTE]
> Frozen z_common probes are now canonical after single-subject probe sweep and cross-fold replication validation.
> Default `probe_weight = 0.05` (updated from 0.03 after cross-fold validation).
> Training-time probe accuracy logging has been resolved via normalization fix.

### Single-Subject Probe Sweep Results (sub-01, runs 01_40, val_runs=8)
| probe weight |    Top-10 |      MRR |  collapse |
| -----------: | --------: | -------: | --------: |
|         0.00 |     13.7% |     7.9% |     0.677 |
|         0.01 |     18.5% | **9.6%** | **0.978** |
|         0.03 |     20.2% | **9.6%** |     0.862 |
|   **0.05**   | **21.0%** |     9.4% |     0.911 |
|         0.10 |     17.7% |     9.4% |     0.645 |

### Cross-Fold Replication Sweep Results (Mean ± Std over Folds 8, 16, 24, 32)
| probe weight | Mean Top-10 | Mean MRR | Mean Collapse | Status |
| -----------: | ----------: | -------: | ------------: | :----- |
|         0.00 | 18.03% ± 3.6% | 8.95% ± 0.8% | 0.86 ± 0.13 | Baseline |
|         0.03 | 19.64% ± 2.6% | 8.83% ± 1.1% | 0.67 ± 0.07 | Promising |
|   **0.05**   | **20.64%** ± 2.3% | **9.14%** ± 1.0% | 0.60 ± 0.16 | **Canonical Default** |

### Known Issues / Status
- **Training-time probe accuracy logging**: Probe evaluation on validation sets returns correct non-zero values (e.g. `probe_is_animate_acc` ~70-80% during epoch logging) matching the normalization fix.
- **Cross-Fold Replication Sweep**: Completed successfully. `probe_weight = 0.05` consistently out-performs `0.03` and `0.00` on average Top-10 accuracy and MRR.
- **Promotion to default**: Set the default `--probe-weight` in `train_eeg_clip.py` to `0.05`.

### Immediate Next Steps
1. Transition to Phase 10 (Image Reconstruction / Diffusion) using grounded visual priors retrieval.

---

- ✅ **Sprint 1 complete** — ZUNA inference, timing audit, retrieval grid.
- ❌ **Sprint 2 failed** — 1.25s crops did not beat controls.
- ✅ **Phase 3.5 complete** — Back-aligned 1.2s ZUNA windows with event marker and combined VLM/Text CLIP supervision successfully avoided collapse and robustly beat shuffled/random controls across Top-10 and MRR! Matrix evaluation gate passed.
- ✅ **Phase 4 complete** — Sprint 3: Simulated EPOC-14 channel subset masking, processed through ZUNA, and evaluated baseline matrix. ZUNA-upscaled signal retains ~74% of full 64-channel density performance, though it doesn't show a direct reconstruction benefit over raw EPOC-14 (0.98x gain).
- ✅ **Phase 5 complete** — Spatial-Temporal Coordinate-Aware encoder implemented and multi-subject scaling evaluated.
- ✅ **Phase 6 complete** — Multi-domain semantic front implemented with VLM attributes, linear warmup, and validated on the combined dataset.
- ✅ **Phase 7 complete** — Built FAISS visual retrieval index for Grounded Image Generation, beating shuffled/random controls on visual priors retrieval.
- ✅ **Phase 8 complete** — Replaced BatchNorm with GroupNorm to stabilize temporal feature extraction; implemented subject-specific FiLM adapters.

---

## 🗺️ Roadmap

### Phase 1 & 2: Baseline & Signal Validation (Complete)
- ✅ Process continuous NOD data through ZUNA foundation.
- ✅ Audit event timing and signal integrity.

### Phase 3: Baseline Matrix (Complete)
- ❌ Sprint 2: 1.25s Crop Matrix (No signal found).
- ✅ **Phase 3.5: Full-window ZUNA Recovery** (Passed)
  - [x] Generate ZUNA windows for sub-01.
  - [x] Implement `temporal_attn` encoder.
  - [x] supervision: Image CLIP + Text CLIP + VLM Semantics.
  - [x] Result: Beat shuffled/random controls robustly.

### Phase 4: Low-Channel Simulation (Complete)
- ✅ Sprint 3: EPOC-14 channel subset masking and baseline matrix comparison.
- ✅ Validate retrieval robustness to channel loss (ZUNA retains ~74% of full-density signal; ZUNA gain vs raw EPOC-14 is ~0.98x).

### Phase 5: Improve EEG Encoder (Complete)
- [x] Sprint 4: Implement Spatial-Temporal Coordinate-Aware Encoder.
  - [x] Grouped temporal convolution stem with strided time downsampling (preserving temporal sequence structure).
  - [x] MNE-based standard 1005 electrode physical 3D coordinate lookup with robust EOG fallbacks.
  - [x] Coordinate projection MLP to map 3D positions to spatial positional embeddings.
  - [x] Spatial transformer over channel tokens.
- [x] Evaluate coordinate-aware architecture vs baseline temporal_attn on single subject.
  - [x] Identified raw channel temporal overfitting. Prepended Early Spatial Mixing (1x1 Conv1d) to act as a learned CAR / spatial filter.
  - [x] Results: Spatial-Temporal Coordinate-Aware (with spatial mix) achieved Top-10 of **0.232** (MRR = **0.1084**), closing the gap to baseline `temporal_attn` (**0.256**).
- [x] Train on multiple subjects to scale performance (sub-01 + sub-02 datasets).

### Phase 6: Multi-Domain Semantic Front (Complete)
- [x] Extract VLM semantic attributes for sub-01 and sub-02 visual stimuli.
- [x] Implement linear auxiliary weight warmup scaling (epochs 1-20).
- [x] Train combined multi-subject baseline and multitask architectures.
- [x] Transition multi-domain semantic front to rely exclusively on a single canonical embedding representation (z_common) with frozen multi-task probe heads prediction.
- [x] Results: Successfully deleted direct label embedding target space, separate attribute heads, and w_label loss in favor of frozen multi-task probe loss predictions, maintaining single latent integrity. Multitask regularization with warmup successfully stabilized combined multi-subject training and improved Top-10 score from 12.05% to 13.25%.

### Phase 7: BReAD-style Retrieval Branch (Complete)
- [x] Build FAISS retrieval index over target image library.
- [x] Query index with predicted embeddings to retrieve visual grounding priors.

### Phase 8: BatchNorm Cleanup and Subject FiLM Adapters (Complete)
- [x] Replace remaining BatchNorm1d layers with GroupNorm.
- [x] Implement subject-specific FiLM scale/shift adapters.
- [x] Validate on 3-condition baseline matrix.

### Phase 9: z_common Canonical Latent + Frozen Probe Regularization (Current)
- [x] Enforce single canonical representation: `z_common` for images, `z_pred_common` for EEG.
- [x] All semantic/label tasks are frozen probe heads from `z_common` — no separate latent spaces.
- [x] Pretrain `CommonProbeModel` on 40-run VLM attributes (10 tasks active, class_label 21.6%).
- [x] Scale to 4 subjects × 40 runs dataset.
- [x] Run 3-condition gate: `zuna_real` top10=20.2% vs shuffled 9.7% vs random 11.3%. **GATE PASS**.
- [x] Fix probe normalization mismatch (`pred` must be normalized before probe, matching pretraining).
- [x] Add subject audit to setup JSON (subjects_loaded, samples_per_subject, subjects_skipped).
- [x] Add `scripts/eval_probe_sanity.py` for probe diagnostic evaluation.
- [x] **Ablation**: run `make ablation` — prove probe improves or is neutral vs no-probe baseline.
- [x] **Probe-weight sweep**: `make probe_sweep` — find optimal weight (0 / 0.01 / 0.03 / 0.05 / 0.10).
- [x] **Subject audit**: verify all 4 subjects actually load via subject audit fields in setup JSON.
- [x] **Cross-fold validation**: validate gate holds with different val-run splits.

### Phase 10: Image Reconstruction / Diffusion (Current)
- [x] **Prerequisite**: probe ablation must confirm probes help or are neutral.
- [x] **Prerequisite**: multi-subject generalization verified.
- [ ] Hook `z_pred_common` to SDXL-Turbo / SD3 img2img via retrieval grounding.

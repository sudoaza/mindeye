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

## Current Status: Phase 13 — decode_unit Canonical Pipeline ✅

### Architecture (canonical, do not deviate)
```
image → CLIP image encoder → decode_unit embedding (1024-dim, unnormalized)
EEG   → temporal_attn_small → z_pred_decode_unit
frozen_probe(normalize(z_pred)) → 10 semantic tasks (class_label 40.3%)

Generation:
  z_pred_decode_unit → soft-kNN retrieve → target_raw embedding → Stable unCLIP
```

### Phase 13 Canonical Model (12B_probe_001)
```
Architecture  : temporal_attn_small
Target space  : decode_unit
Loss          : contrastive (InfoNCE, T=0.07)
Probe         : frozen decode-probe v2, probe_weight=0.01, probe_start_epoch=5
Grad clipping : clip_grad_norm_(model, 1.0)
Best epoch    : 13 / 30

Metrics:
  within-val Top-10  : 0.04195  (2.5× random)
  full_bank Top-10   : 0.00671  (2.68× random, primary gate metric)
  full_bank MRR      : 0.00318
  collapse score     : 0.847
```

### Phase 13 Gate Result
| Condition | Result |
|---|---|
| `full_bank_top10 > A_real_repro` (4.0×) | ✅ PASS |
| `full_bank_mrr ≥ A_real_repro` | ✅ PASS |
| `real > shuffled` (4.0×) | ✅ PASS |
| `real > random mean` (+2.45σ) | ✅ PASS |
| `collapse > 0.1` | ✅ PASS |

### Key Finding: full_bank is the honest retrieval metric
> A_real_repro (no probe) achieved within-val Top-10 = 0.042 but full_bank Top-10 = 0.00168 (BELOW random 0.0025). The probe forces global embedding alignment. Within-val is diagnostic only.

### Immediate Next Steps
1. Multi-subject: add sub-02, sub-03, sub-04 with 12B_probe_001 config.
2. Generation grid: oracle / 12B_probe_001 kNN / shuffled kNN / random kNN (k=5, T=0.05).

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

### Phase 10: Image Reconstruction / Diffusion ✅
- [x] **Prerequisite**: probe ablation must confirm probes help or are neutral.
- [x] **Prerequisite**: multi-subject generalization verified.
- [x] Hook `z_pred_common` to Stable Diffusion img2img via retrieval grounding (Phase 10A).
- [x] Implement Attribute-Constrained Semantic Montage Reranking to diversify and semantically align retrieved priors (Phase 10B).

### Phase 11: Visual Calibration Battery & Frozen Diffusion Demo ✅
- [x] **Visual calibration stimulus generation**: generated 636 calibration trials covering shapes (matched area), colors, textures, and animacy/faces.
- [x] **Embed calibration stimuli**: mapped calibration targets into canonical `z_common` manifold and updated `common_embeddings.pt`.
- [x] **Dataloader mixing**: implemented `MixedBalancedDataset` to interleave natural and calibration trials (50/50).
- [x] **Calibrated loss scaling**: updated `train_eeg_clip.py` with custom sample weight routing and separate validation metrics.
- [x] **Frozen diffusion demo**: implemented and executed `demo_diffusion.py` with Mode A (text-only semantic steering) and Mode B (prior-guided img2img) outputs.

### Phase 13: decode_unit Canonical Pipeline ✅
- [x] Architect-mandated code fixes: target_space logging, full-bank retrieval, decode wrapper --mode, --probe-start-epoch, task-normalized probe loss, grad clipping.
- [x] Decode probe v2 retrained cleanly (lr=1e-4, task-normalized loss, grad clip, 50 epochs, 0 NaN).
- [x] Experiment matrix: A_real_repro, 12B_probe_0005, **12B_probe_001** (winner), 12B_probe_002.
- [x] Gate: all 4 conditions pass for 12B_probe_001 (full_bank_top10 = 2.68× random, +2.45σ vs random controls).
- [x] **Key finding**: within-val Top-10 inflates signal 2–4×. Full-bank is the honest metric. A_real_repro (no probe) is BELOW random on full-bank despite good within-val numbers.
- [x] **Canonical model promoted**: 12B_probe_001 (probe_weight=0.01, probe_start_epoch=5).

### Phase 14: Multi-subject CLIP Generation Grid ✅
- [x] Scale to sub-01 → sub-04 with 12B_probe_001 config (decode_unit + probe).
- [x] Generation grid: oracle / EEG-kNN / shuffled / random rows, k=5, T=0.05.
- [x] Full-bank metric primary gate for all future runs.
- [x] Result: Grid shows meaningful retrieval above shuffled/random controls.

### Phase 15: Multi-Subject Scaling + Subject Adapters ✅
- [x] 4-subject training (sub-01→04 × 40 runs) with per-subject FiLM adapters.
- [x] Validated subject audit (all 4 subjects load, per-subject sample counts logged).
- [x] Ablation: probe-on vs probe-off; probe is beneficial or neutral on full-bank.

### Phase 16: Subject Adapter Ablation ✅
- [x] Ran subject adapter ablation matrix across multiple weight configurations.
- [x] Confirmed `temporal_attn_small` with subject FiLM adapters is the canonical encoder.

### Phase 17: RAE Exploration — Replace CLIP Generation Backbone ✅

Motivation: CLIP's 1024-dim global vector is a weak target for image reconstruction. RAE
(`AutoencoderRAE` w/ DINOv2-base encoder) outputs high-dimensional spatial latents `[768, 16, 16]`
that the decoder can use to reconstruct images with rich detail.

**Split A: Image RAE tokens → code → reconstruct (frozen)**
- [x] Phase 17.1: RAE oracle quality — direct `[768, 16, 16]` → decoder → image. Confirmed reconstruction fidelity.
- [x] Phase 17.2: RAE centered unit (global 768-dim) as EEG target, kNN retrieval → RAE decode.
- [x] Phase 17.3: RAE centered kNN grid vs CLIP kNN grid — RAE visually cleaner.
- [x] Key finding: RAE decoder is powerful; the bottleneck is EEG→latent fidelity, not the decoder.

### Phase 18A: RAE Token Bottleneck Training ✅

Goal: learn a compact `code = compress(tokens)` / `tokens ≈ expand(code)` autoencoder so EEG
only needs to predict the lower-dim code.

- [x] Implemented `_SpatialPoolBottleneck` (AdaptiveAvgPool2d + learned 1×1 refine conv) and conv variants.
- [x] Trained all spatial grid sizes: 2×2, 3×2, 3×3, 4×3, 4×4 and conv variants.
- [x] Key results (oracle token cosine / collapse rate):
  - `spatial_768x4x4`: **0.833**, 0% collapse — **best healthy bottleneck** ✅
  - `spatial_768x3x3`: 0.802, 0% collapse
  - `spatial_768x3x2`: 0.766, 0% collapse (below gate)
  - All conv variants: collapse to >90% (eliminated)
- [x] Gate: use spatial only; 4×4 (0.833) and 3×3 (0.802) pass.

### Phase 18B: EEG → spatial_768x3x3 Code ✅

**Split B: EEG → compressed RAE code**

- [x] Extracted `rae_bottleneck_codes_3x3.pt` (6912-dim codes) for all 15,865 images.
- [x] Trained `TemporalAttnEncoder` with `spatial_cosine` loss (no raw MSE, no centering).
  - Best epoch 87; spatial_cosine = **0.608** in val; 0% channel collapse.
  - Scale ratio: pred_std = **4.8× target_std** (scale inflation).
  - Expanded token cosine (EEG vs oracle): **0.293** vs shuffled **0.290** (gap +0.003).
- [x] **Scale sensitivity test** (inference-time rescale A/B/C + oracle scale factors 0.25→5.0):
  - All rescale variants give identical expanded_token_cosine (0.293).
  - Oracle scale factors 0.25–5.0 all give expanded_token_cosine ≈ **0.695** (flat).
  - **Conclusion: expander is scale-invariant** (1×1 refine conv = channel mixing, not magnitude-sensitive).
  - Root cause of 0.003 gap: EEG code has correct bulk direction (cosine 0.608) but insufficient per-channel/per-site fidelity for the 3×3 bottleneck → expand path.
  - Increasing bottleneck oracle ceiling is the lever (4×4 > 3×3).

### Phase 18C & 18C-v2: EEG → spatial_768x4x4 Code + Probe ✅

Rationale: 4×4 oracle cosine = 0.833 vs 3×3 = 0.802. More spatial information per code gives EEG encoder a higher ceiling. Probe gives semantic anchor to prevent directional drift.

- [x] Extracted `rae_bottleneck_codes_4x4.pt` (12,288-dim codes) for 15,891 images.
- [x] Trained RAE-code probe on `mean_pool([768,4,4]) → [768] → normalize` (11 active tasks beat baseline; class_label 60.1%, animal_visible 98.2%, human_visible 94.2%, dominant_color 53.9%).
- [x] **Phase 18C (Cold Start)**: Trained EEG encoder from scratch.
  - Spatial cosine: 0.5835; scale inflation: 4.83×.
  - Expanded token cosine: 0.2920 vs shuffled 0.2897 (gap +0.0023, overlapping CIs).
- [x] **Phase 18C-v2 (Warm Start)**: Initialized from Phase 16c best.pt. Deployed probe logging fix.
  - Spatial cosine: 0.5850; scale inflation: 3.31×.
  - Expanded token cosine: **0.2955** vs shuffled **0.2884** (gap **+0.0071**, non-overlapping 95% CIs).
  - Probe metrics: Captured `animal_visible` (74.3% vs 59.1% baseline) and `dominant_color` (31.4% vs 27.4%), while `class_label` remains at chance (0.13%).
- [x] Visual grids generated for both runs.

### Phase 18D: Loss Cleanup & Paired Bootstrap (Complete) ✅

Rationale: Suppress scale inflation using target-anchored site norm loss (`--loss spatial_cosine_norm`) and implement a 10,000-iteration paired bootstrap significance test.

- [x] Fix `normalize_output` initialization bug in training and evaluation pipelines.
- [x] Implement relative spatial site norm matching loss anchored to target scale (`scale = target_site_norm.mean().detach() + 1e-6`).
- [x] Log raw/relative norm statistics every epoch.
- [x] Implement paired bootstrap resampling deltas (`delta_i = eeg_cos_i - shuffled_cos_i`) over 10,000 iterations.
- [x] Results:
  - **Scale inflation resolved**: `pred_code_std` = 1.03 vs 0.93 target (ratio **1.10x** vs 3.31x in 18C-v2).
  - **No expansion gap**: Expanded token cosine EEG vs Shuffled shows **0.2922** vs **0.2922** (gap = **0.00000**, 95% CI `[-0.00004, 0.00004]`).
  - **Conclusion**: Distribution alignment works, but the predicted EEG codes lack the high-fidelity features required to survive the frozen non-linear expansion mapping. A stronger mapping or capacity increase is required.

---

## 🗺️ Future Work

### Phase 18E: RAE Capacity Scaling & Direct Regress
- [ ] **Scale Spatial Grid**: Test `spatial_768x5x5` codes to increase the oracle ceiling beyond 0.85.
- [ ] **Direct Token Regression**: Attempt direct regression of `[768, 16, 16]` space using convolutional priors on the projection head.
- [ ] **Backfill VLM Attributes**: Re-annotate remaining 11 attributes to expand probe to 29 tasks and increase coverage.

### Phase 7 (Deferred): BReAD-style Retrieval Branch
- [ ] Add `src/mindseye/embeddings/faiss_index.py`, `scripts/build_retrieval_index.py`.
- [ ] Implement after EEG→RAE code achieves expanded_token_cosine gap ≥ 0.02.

### Phase 8 (Deferred): Frozen Diffusion img2img
- [ ] Pipeline: EEG code → expand → RAE decode → SDXL-Turbo img2img refinement.
- [ ] Implement after EEG→RAE reconstructions show clear shape/layout separability from shuffled in visual grids.

### Phase 9 (Deferred): Alljoined / ENIGMA Comparison
- [ ] Domain adaptation / consumer-grade robustness after core pipeline matures.


# MindEye Development Plan — ZUNA-first EEG→Semantic→Image Pipeline

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
- [ ] Run `make matrix` — `zuna_runheldout` must beat all controls.
- [ ] **Gate**: `zuna_runheldout` must beat shuffled/random controls. `pred_std` must be nonzero and not collapsed. Retrieval grid must not repeat the same images.

## 7. Phase 4 — Low-channel / cheap-headset simulation (Sprint 3)

**Objective**: Test if ZUNA can rescue EPOC-like 14-channel EEG. This remains mandatory and should not be deferred behind better encoders.
- [x] `src/mindseye/zuna/channel_simulation.py` and `scripts/simulate_low_channel_zuna.py` exist.
- [ ] Add `configs/zuna/zuna_epoc14_sim.yaml`.
- [ ] Simulate paths: full NOD → ZUNA, canonical 32ch → ZUNA, EPOC-like 14ch → ZUNA, random 14ch → ZUNA, raw EPOC-like 14ch without ZUNA.
- [ ] Use approximate EPOC X channel list.
- [ ] Report: `zuna_gain = metric(epoc14_zuna) / metric(raw_epoc14_nozuna)` and `retention = metric(epoc14_zuna) / metric(full_zuna)`.
- [ ] `make simulate` runs FIF masking for all 5 runs; then train baseline matrix on simulated crops.

## 8. Phase 5 — Improve the EEG encoder (Sprint 4)

**Objective**: Upgrade from Conv1D to a Spatial-Temporal Coordinate-Aware Encoder.
- [ ] Add `src/mindseye/models/spatial_temporal_encoder.py`.
- [ ] Input API: `eeg: [B, C, T]`, `channel_xyz: [B, C, 3]`, `subject_id` (optional), `run_id` (optional).
- [ ] Architecture target: temporal convolution / learned filterbank + coordinate embedding MLP + channel attention or transformer + temporal pooling + subject/run adapter + projection to CLIP dimension.
- [ ] Keep the Conv1D baseline for comparison.

## 9. Phase 6 — Multi-domain semantic front (Sprint 5)

**Objective**: Move from CLIP image embeddings to structured semantic state.
- [ ] Add structured targets: CLIP image embedding, CLIP text/class embedding, object caption embedding, spatial/composition embedding, color/material embedding, mood/theme embedding, abstract concept embedding, direction/action embedding.
- [ ] This is core Neural-MCRL / Semantic-Prompts inspiration. Do not jump to diffusion before this has measurable signal.

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

## Current Status: Phase 3.5 (Recovery) 🚧

- ✅ **Sprint 1 complete** — ZUNA inference, timing audit, retrieval grid.
- ❌ **Sprint 2 failed** — 1.25s crops did not beat controls.
- 🚀 **Phase 3.5 active** — Full-window ZUNA semantic baseline (5s windows).

---

## 🗺️ Roadmap

### Phase 1 & 2: Baseline & Signal Validation (Complete)
- ✅ Process continuous NOD data through ZUNA foundation.
- ✅ Audit event timing and signal integrity.

### Phase 3: Baseline Matrix (Failed/Retrying)
- ❌ Sprint 2: 1.25s Crop Matrix (No signal found).
- 🚀 **Phase 3.5: Full-window ZUNA Recovery** (Active)
  - [ ] Generate 5s ZUNA windows for sub-01 runs 01–10.
  - [ ] Implement `temporal_attn` encoder for [C, 1280] input.
  - [ ] supervision: Image CLIP + Text CLIP (from class labels).
  - [ ] Result: Beat shuffled/random controls with 5s windows.

### Phase 4: Low-Channel Simulation (Next)
- [ ] Sprint 3: EPOC-14 channel subset masking.
- [ ] Validate retrieval robustness to channel loss.

*Do not add diffusion until semantic retrieval beats shuffled baselines.*

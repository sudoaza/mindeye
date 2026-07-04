# QFormer Pipeline — Code & Architecture Review

Scope: the **active** decode path driven by `scripts/prepare_multisubject_data.sh`
(download → ZUNA denoise → crop → RAE bank → cache latents → QFormer grid + gate).
This is a **review-and-document** pass. No functional code was changed. Each finding
lists severity, `file:line` citations, impact, and a one-line recommended fix.

Data flow reviewed:

```
raw NOD FIF + detailed_events CSV
  └─ cropper.py            onset→epoch, attach image_id        [N,62,1281] npz + metadata
      └─ cache_zuna_latents.py  trim 1281→1280, ZUNA encode    [2480,32] latents.pt + metadata.pt
          └─ ZunaLatentTargetDataset  crop tc[20:36)→[992,32]
ImageNet stimuli
  └─ build_rae_latent_bank.py  AutoencoderRAE.encode → rae_unit [768]
      └─ (target bank)
  → ZunaToVisionQFormer → [B,768] → retrieval_topk + paired-bootstrap gate
```

---

## HIGH — can silently invalidate results

### H1. Gate ranks within the val set, not the full image bank (docs↔code mismatch)
- **Where:** `scripts/train_zuna_to_vision.py` `evaluate_model`/`save_eval_metadata` L288–356, L386; `src/mindseye/models/eeg_encoder.py` `retrieval_topk` L262–303.
- **What:** `retrieval_topk` computes `logits = pred_n @ tgt_n.T` where `tgt_n` is the **concatenation of the N val-set targets only**. The diagonal is truth; ranking is over the ~N val samples, not the ~35k RAE image bank.
- **Why it matters:** The docs repeatedly state the opposite and gate on it — README/HANDOVER §5, §7.1: *"predictions ranked against the full RAE/DINO image bank"* and *"Full-set retrieval is the only honest metric; within-val ranking is diagnostic only (inflated)."* The grid gate in `run_qformer_grid.py` keys on `val_mrr_norm`/`val_top10_norm`, i.e. the **inflated within-val** metric the docs say is diagnostic-only. Absolute numbers are optimistic and vary with val-set size.
- **Extra wrinkle:** targets are stored **per sample**, not per unique image. When two val rows share an `image_id`, their target vectors are identical, so a query ties with its duplicate; argsort tie-breaking then perturbs the true rank up or down non-deterministically.
- **Fix:** Rank each prediction against the **full unique-image RAE bank** (dedup by `image_id`), with the row's own image as the single positive; keep within-val as a clearly-labelled diagnostic. Update the gate to consume the full-bank metric, or fix the docs to admit the gate is within-val.

### H2. Onset↔events-CSV pairing is positional, with no time-join or count assertion
- **Where:** `src/mindseye/zuna/cropper.py` L178–233 (`crop_run_to_epochs`); `events_for_run` L87–93; `stim_onsets_from_raw` L76–84.
- **What:** FIF annotation onsets (`onset_seconds`, L178) are paired with the detailed-events CSV rows (`metadata`, L182) **by row position**: `n = min(len(onsets), len(metadata))` (L183), `onset_seconds[:n]` (L184), `metadata.iloc[:n]` (L216), then `stim_onset_sec` is attached positionally (L226). `events_for_run` filters by run/session but does **not** sort by onset nor join on a time column.
- **Why it matters:** If the CSV's row order for a run differs from the annotation-stream order (or a single event is dropped at the head), **every `image_id` label shifts by one for the whole run** — a silent, systematic mislabel that no downstream step detects. The `valid` out-of-bounds mask (L229–233) is applied *after* pairing, so it preserves any pre-existing misorder.
- **Fix:** Join onsets↔CSV on the CSV's own onset/latency column (NOD provides one) instead of `iloc` zip; at minimum assert `len(onset_seconds) == len(metadata)` per run and `abs(csv_onset - fif_onset) < tol`, and fail loudly on mismatch. The existing `avg_others > 1.0` print (L213) only warns about window overlap, not misordering.

---

## MEDIUM — dimensions / config coupling / control integrity

### M1. Training crop window `tc[20:36)` is decoupled from the actual onset
- **Where:** `scripts/train_zuna_to_vision.py` `crop_zuna_latent` L40–59; onset provenance `scripts/cache_zuna_latents.py` L172.
- **What:** The latent window is **hard-coded** `tc[20:36)` and is only correct for the `-3.0/+2.0` back-aligned crop (onset at sample 768 → `onset_tc = 768/32 = 24`). `cache_zuna_latents` computes and stores `onset_tc` per row (L172) but training never reads it — the slice is fixed regardless of `tmin`. The fast-path only asserts the *input* shape `(2480,32)`, not the onset position.
- **Fix:** Derive the slice from stored `onset_tc` (e.g. `[onset_tc-4 : onset_tc+12)`), or assert `onset_tc == 24` before using the fixed window so a different crop config fails loudly instead of silently windowing the wrong latents.

### M2. FiLM subject index assumes a contiguous `sub-01..sub-0N` cohort
- **Where:** `scripts/train_zuna_to_vision.py` L244 (`record["subject_id"] - 1`); `src/mindseye/adapters/qformer.py` L118–119 (`nn.Embedding(num_subjects, …)`); `scripts/prepare_multisubject_data.sh` L40, L96 (`NUM_SUBJECTS=$(wc -w)`).
- **What:** The FiLM embedding is indexed by `literal_subject_number - 1`, while its size is `num_subjects = count(subjects)`. For `sub-01..sub-09` this happens to line up (indices 0–8, size 9). For any cohort **not** starting at 1 or with gaps — e.g. `SUBJECTS="sub-03 sub-04 sub-05"`, `NUM_SUBJECTS=3` — indices become 2,3,4 while the embedding has size 3 → out-of-bounds / CUDA device-side assert at runtime.
- **Fix:** Build a cohort-relative map `{subject_number → 0..N-1}` at dataset construction and index FiLM by that, decoupling embedding size from raw subject numbering.

### M3. Global run splits are not image-disjoint across subjects
- **Where:** `scripts/train_zuna_to_vision.py` run-split logic L494–524; noted in HANDOVER §8.
- **What:** Train/val/test are split purely by `run_id`. The same `image_id` can be a **train** target for subject A and a **val** query for subject B, so the model has already fit that exact target vector.
- **Why it matters:** Optimistic retrieval — the val target isn't novel. HANDOVER §8 flags this as "acceptable for now," but it compounds H1.
- **Fix:** Add an optional image-disjoint split (partition by `image_id`, then assign runs), or at least log the train∩val `image_id` overlap so the inflation is quantified.

### M4. Channel positions re-derived from a generic montage with silent `[0,0,0]` fallback
- **Where:** `scripts/cache_zuna_latents.py` L113–130 (fallback at L127).
- **What:** Caching rebuilds 3D positions from `make_standard_montage("standard_1005")` by name match; any unmatched NOD channel silently gets `[0,0,0]`. This corrupts ZUNA's 4D-RoPE spatial encoding for that channel. It also **discards the montage already present in the ZUNA FIF**, using a generic template instead.
- **Fix:** Hard-fail (or explicitly report) on any unmatched channel; prefer reading positions from the FIF montage that the cropper preserved, falling back to the template only when the FIF has none. Confirm all 62 NOD channel names resolve in `standard_1005`.

### M5. Grid gate: fragile CI-string re-parsing and N/A→0.0 coercion
- **Where:** `scripts/run_qformer_grid.py` L241–312 (coercion L277–278, CI re-parse L286–288).
- **What:** The gate re-parses a **formatted** CI string (`float(ci.strip("[]").split(",")[0])`) instead of using the numeric CI it already computed, and coerces a missing control's `"N/A"` delta to `0.0`. A missing control therefore fails the gate (safe) but for an opaque reason ("below +0.005"), and the string parsing is brittle. The paired **top10** deltas from `align_and_compute_deltas` are computed and reported but the gate criteria only use MRR deltas.
- **Fix:** Pass numeric CI tuples through to the gate (don't round-trip through strings); make a missing control an explicit hard error; either gate on the top10 paired delta or drop it from the pipeline to avoid implying it's used.

---

## LOW — robustness / clarity (no correctness impact confirmed)

### L1. `subprocess` PIPE-buffer deadlock risk in the grid
- **Where:** `scripts/run_qformer_grid.py` L120–172.
- **What:** Three training subprocesses are launched with `stdout=PIPE` and only drained via `.communicate()` after all are started; three concurrent trainings also share one `--device cuda`. A child that fills its ~64KB stdout pipe blocks until read → potential deadlock; concurrent runs can also contend/OOM on one GPU.
- **Fix:** Stream/drain each child's output (threads or `select`), or write child logs to files; consider serializing or explicitly sharding the GPU.

### L2. RAE preprocessing is handled inside the model — no code bug (finding downgraded)
- **Where:** `src/mindseye/generation/rae_backend.py` L104–118 vs `diffusers` `AutoencoderRAE._resize_and_normalize`.
- **What:** `extract_rae_latent` does only `TF.to_tensor` (→[0,1]), no resize/normalize. Initially flagged HIGH, but `AutoencoderRAE._encode` calls `_resize_and_normalize`, which **bicubic-resizes to `encoder_input_size` (224)** and applies **ImageNet mean/std** internally. So the bank is correct.
- **Residual note (not a bug):** the internal resize force-squares any H×W to 224×224 (no shorter-side + center-crop), so non-square stimuli are aspect-distorted. This is inherent to the RAE API and applied uniformly, so it does not bias the target bank; document it and, if desired, center-crop upstream for closer DINOv2-eval parity.

### L3. Misleading defaults / stale artifact names
- `src/mindseye/adapters/qformer.py` L80 default `d_in=1024` vs real ZUNA dim 32.
- `scripts/build_rae_latent_bank.py` L15 default output still `..._sub01_04_runs01_40.pt`; `scripts/run_qformer_grid.py` default paths point at stale single-subject artifacts.
- **Fix:** Update defaults to the multi-subject artifacts or make them required args.

### L4. Duplicate `AttentionPooler`
- Defined in both `src/mindseye/adapters/qformer.py` and `src/mindseye/models/eeg_encoder.py`.
- **Fix:** Keep one; import it in the other (DRY).

### L5. `num_bins` / positions provenance note
- `verify_params.py` L27 probes `discretize_chan_pos(..., 100)` and questions 50, while `latent_extractor.py` hardcodes `num_bins = 50`. Confirm 50 is the value ZUNA was trained with (RoPE binning must match training) — a mismatch would quietly degrade spatial encoding.

---

## Verify-on-pod checklist (runtime confirmation)

Cheap asserts/prints to run once on the pod to convert the above from "suspected" to "confirmed":

1. **H2 label alignment** — for one run, print the events CSV head with its onset column next to `stim_onsets_from_raw(...)`; assert `len(onsets) == len(metadata)` and per-row `abs(csv_onset - fif_onset) < 0.05s`. Decode 3 random epochs → save the paired stimulus image and eyeball plausibility.
2. **H1 eval scope** — log `tgt_n.shape` inside `retrieval_topk`; confirm N ≈ val-sample count, **not** ~35k. Recompute MRR against the deduped full bank and compare — expect a large drop.
3. **M1 crop window** — assert `metadata["onset_tc"].unique() == [24]`; if not, the fixed `tc[20:36)` is wrong for that cohort.
4. **M2 FiLM bounds** — assert `max(subject_id-1) < num_subjects` at dataset build; smoke-test a non-`sub-01` cohort (e.g. `sub-03..05`).
5. **M3 leakage** — print `len(set(train_image_ids) & set(val_image_ids))`; quantify overlap.
6. **M4 channels** — assert zero channels fell back to `[0,0,0]`; print any unmatched names.
7. **Shapes end-to-end** — `[N,62,1281]` npz → trim `[N,62,1280]` → ZUNA `[2480,32]` → crop `[992,32]` → QFormer out `[B,768]` == RAE `rae_unit` dim. Assert each hop.

## Summary

| ID | Severity | One-line |
|----|----------|----------|
| H1 | HIGH | Gate uses within-val ranking, not full-bank; contradicts docs |
| H2 | HIGH | Onset↔CSV pairing is positional; no time-join/count assert → possible whole-run label shift |
| M1 | MED | Fixed `tc[20:36)` latent window ignores stored `onset_tc` |
| M2 | MED | FiLM index assumes contiguous `sub-01..0N`; breaks other cohorts |
| M3 | MED | Run splits not image-disjoint across subjects (optimistic) |
| M4 | MED | Channel positions silently fall back to `[0,0,0]` |
| M5 | MED | Gate re-parses CI strings; N/A→0.0; top10 paired delta unused |
| L1 | LOW | Grid subprocess PIPE-buffer deadlock / shared-GPU contention |
| L2 | LOW | RAE preprocess handled in-model (downgraded); only aspect-distortion note |
| L3 | LOW | Misleading defaults / stale artifact names |
| L4 | LOW | Duplicate `AttentionPooler` |
| L5 | LOW | Confirm ZUNA RoPE `num_bins=50` matches training |

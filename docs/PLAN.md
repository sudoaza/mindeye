# MindEye Development Plan — ZUNA → QFormer → RAE

> **Way of work**: all GPU work runs on a RunPod pod via the runpod MCP; data + big weights
> persist on a network volume. See [`INFRA.md`](INFRA.md). Current state lives in
> [`HANDOVER.md`](HANDOVER.md); this file is the roadmap. Doc index: [`README.md`](README.md).

## 1. Strategic direction

Decode mental imagery from EEG using three best-in-class components, each frozen except the bridge:

| Component | Role | Why it was chosen |
|---|---|---|
| **ZUNA** | EEG embedding (frozen) | EEG foundation model; best signal-feature recovery. Requires 256 Hz, 5 s epochs. We cache its `post_mmd` latents. |
| **QFormer** | Learned bridge | Simple linear/MLP adapters did **not** work. Query-token cross-attention selects relevant ZUNA tokens. |
| **RAE / DINOv2** | Reconstruction target (frozen) | Best visual reconstruction fidelity. CLIP/ViT was semantically ok but visually imprecise. |

Primary source is **NOD-EEG** (continuous, event-timed). Alljoined/ENIGMA are for later robustness only.

## 2. Non-negotiable technical principles

1. **Full-set retrieval + paired bootstrap** vs `real / shuffled / random` is the only honest gate. Within-val ranking inflates signal 2–4× and is diagnostic only. The 3-way control split has repeatedly caught data/pipeline bugs and stays mandatory.
2. **ZUNA and RAE are frozen; only the QFormer bridge trains.**
3. **Onset back-alignment is assumed.** Epochs for ZUNA latent caching are **5s back-aligned** (`run_cropper.py --full5s-backaligned`, `-3.0/+2.0`, 1280 samples, onset at sample 768), so the fixed ZUNA latent window `tc[20:36)` is correct. The tight `-0.2/+1.0` window is the deprecated semantic-classifier path and crashes latent caching (asserts 1280 timepoints).
4. **No RAE decoder / diffusion** until the QFormer retrieval gate (Δ real−shuffled > +0.005, 95% CI excludes 0) is consistently met.
5. **Timing integrity is critical** (stimulus onset → ZUNA axis → crop).
6. `PYTHONPATH=src` before any `python scripts/` call on the pod.

## 3. Current architecture (live)

```
EEG (256 Hz, 5s, 64 ch)
  └─► ZUNA denoiser (frozen)          [scripts/run_zuna_batch.py]
        └─► cache post_mmd latents    [scripts/cache_zuna_latents.py]   [2480,32] = [62ch × 40tc, 32d]
              └─► onset crop tc[20:36)                                   → [992,32]
                    └─► ZunaToVisionQFormer   [src/mindseye/adapters/qformer.py]
                          input_proj 32→256 · 32 query tokens (+CLS) · subject FiLM
                          4× (self-attn → cross-attn(queries→ZUNA) → FFN) · CLS readout
                          → proj_head → d_out → LayerNorm → L2-normalize
                          └─► vision embedding (RAE/DINO-768 target; CLIP dropped)
```

- **Loss**: InfoNCE + cosine + variance-floor (anti-collapse, 0.05).
- **Eval**: full-set retrieval against the real RAE/DINO bank (`rae_unit`); always ranked vs the true image target even under shuffled/random training.
- **Grid**: `scripts/run_qformer_grid.py` trains real/shuffled/random for each target space (DINO-Unit-768, DINO-PCA-256/128 — CLIP dropped) then runs a 10,000-iter paired bootstrap.

## 4. Live roadmap

### Phase Q1 — QFormer retrieval gate (🚧 In Progress)
- [x] Cache ZUNA `post_mmd` latents; onset-crop plumbing (`crop_zuna_latent`, `--latent-tc-start/-end`).
- [x] `ZunaToVisionQFormer` bridge; InfoNCE + cosine + variance-floor loss.
- [x] `run_qformer_grid.py` with real/shuffled/random + paired bootstrap over the DINO target spaces.
- [x] Multi-subject cohort support: `subject` column in cropper, multi-`--epochs-dir` merged caching, `--num-subjects` FiLM plumbing (`scripts/prepare_multisubject_data.sh`).
- [ ] **Full 9-subject cohort grid running on the A100 pod** (`sub-01..09 × 32 runs`, `--num-subjects 9`) — in progress as of 2026-07-03; see HANDOVER §0. Positive results have historically needed this scale.
- [ ] **Gate**: paired Δ (real − shuffled) > +0.005, 95% CI excludes 0, `collapse_pct` < 20%, on full-set retrieval against the RAE bank.
- [ ] Pick the winning target space (expectation: `DINO-Unit-768`; PCA variants test whether a lower-rank target is easier to hit).
- [ ] Decide whether combined-cohort FiLM helps vs per-subject; consider an image-disjoint split if leakage is suspected (current split is global run-based, shared across subjects).

### Phase Q2 — Reconstruction bridge (after Q1 gate) 🔜
> **Open architectural gap.** The current QFormer pools to a single vector — a *retrieval* bridge.
> RAE reconstruction needs the `[768,16,16]` token grid, not a pooled vector.
- [ ] Extend the QFormer to predict the RAE token grid (or a faithfully-expandable code).
- [ ] Validate token-grid fidelity vs the pooled-retrieval baseline.

### Phase Q3 — RAE decode (after Q2)
- [ ] EEG → QFormer → RAE token grid → frozen RAE decoder → image.
- [ ] 3-way visual grids (target / RAE oracle / EEG reconstruction) must separate from shuffled.

### Phase Q4 — Retrieval priors + diffusion (deferred)
- [ ] FAISS kNN visual priors once the full-set gap ≥ 0.02.
- [ ] Frozen diffusion img2img refinement on RAE decode.

### Deferred / robustness
- [ ] **VLM attribute backfill** — 11 calibration attrs missing from the original Qwen prompt. Canonical doc: [`VLM_ATTRIBUTES.md`](VLM_ATTRIBUTES.md). Does not block the QFormer gate.
- [ ] **Alljoined / ENIGMA** — domain adaptation / consumer-grade robustness after the core pipeline matures.

## 5. Output & experiment tracking

Every run writes a timestamped dir with `config.json`, `metrics.json`, `history.csv`, checkpoints,
and `val/test_eval_preds.pt` (for the paired bootstrap). Grid outputs under
`outputs/qformer_cohort9_grid/grid_<timestamp>/` (or the `--out-dir` passed to the grid).

---

## 6. Completed history (context, do not re-run)

The path below reached the current architecture. Superseded phases are kept for context only.

| Phase | Outcome |
|---|---|
| 1–2 | ZUNA inference, timing/data-integrity audit, retrieval grid. |
| 3.5 | Back-aligned ~1.2s ZUNA windows + VLM/Text CLIP supervision beat shuffled/random (Sprint-2 1.25s crops had failed). |
| 4 | EPOC-14 low-channel sim: ZUNA retains ~74% of full 64-ch signal (no direct gain over raw EPOC-14, ~0.98×). |
| 5 | Spatial-Temporal Coordinate-Aware encoder; multi-subject scaling. |
| 6 | Multi-domain semantic front → single canonical `z_common` + frozen multi-task probe. |
| 7 | FAISS visual retrieval index (beat controls on priors). |
| 8 | BatchNorm→GroupNorm; subject FiLM adapters. |
| 9 | `z_common` canonical latent + frozen-probe regularisation; 4-subject gate pass. |
| 10–11 | Retrieval-grounded img2img; calibration battery; frozen diffusion demo. |
| 13 | `decode_unit` canonical pipeline (12B_probe_001). Established **full-bank is the honest metric** (within-val below random on full-bank). |
| 14–16 | Multi-subject CLIP generation grid; subject-adapter ablation; `temporal_attn_small`+FiLM canonical. |
| 17 | **DINOv2-RAE decoder swap.** Found the decoder is powerful; bottleneck is EEG→latent fidelity. Established RAE as the reconstruction target. |

### ⛔ Abandoned: RAE code-bottleneck (Phase 18B–18E)

Tried compressing EEG into a tiny `768×4×4` RAE code and expanding it back to tokens.

- **18A**: `spatial_768x4x4` bottleneck (oracle cosine 0.833) was the best healthy compressor; conv variants collapsed (>90%).
- **18B/C**: EEG→code warm-start gave a fragile expanded-token gap (+0.0071, 18C-v2).
- **18D**: fixing scale inflation collapsed the gap to ~0 — the expander is scale-invariant, so matching scale didn't help.
- **Conclusion**: squeezing EEG through a lossy code discards the per-channel/per-site fidelity the RAE decoder needs. **Superseded by the QFormer bridge** (learn the ZUNA→vision mapping directly; keep RAE for reconstruction).

The `scripts/run_phase18*.sh`, `train_rae_token_bottleneck.py`, `build_rae_bottleneck_codes.py`,
`build_rae_code_stats.py` scripts remain on disk for reference but are **not part of the live plan**.

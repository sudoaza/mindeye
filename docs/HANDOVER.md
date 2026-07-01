# MindEye Handover Document

> **Current Phase**: QFormer Bridge — ZUNA → QFormer → RAE retrieval grid
> **Updated**: 2026-06-21

This document summarises the current project state, remote environment, and step-by-step instructions to resume work. Keep this file updated when major milestones are reached.

---

## 1. Executive Summary & Objective

**MindEye** decodes mental imagery from EEG signals recorded during visual stimulation.

### Canonical thesis (current architecture)

Three frozen/learned components, each chosen because it is the best available tool for its job:

| Component | Role | Why |
|---|---|---|
| **ZUNA** | EEG embedding | EEG foundation model; recovers signal features best. Used frozen; we cache its `post_mmd` latents. |
| **QFormer** | Bridge | Learned ZUNA→vision adapter. Simple linear/MLP adapters did **not** work; QFormer's query-token cross-attention is the bridge. |
| **RAE** | Reconstruction target | Best visual reconstruction fidelity. CLIP/ViT was acceptable semantically but visually imprecise, so it has been dropped as a target. |

The pipeline:

1. Process continuous EEG through **ZUNA** (frozen EEG foundation diffusion denoiser) → cache `post_mmd` latents.
2. Onset-crop the cached latents to a tight window around stimulus onset.
3. Train a **QFormer bridge** (`ZunaToVisionQFormer`) to map ZUNA latents → visual embedding space.
4. Target the **RAE (DINOv2-based) embedding bank** for reconstruction-grade visual fidelity. **CLIP has been dropped** — it was only ever a semantic baseline and is no longer a training target.

**Primary gate metric**: full-set retrieval rank against the entire RAE/DINO image bank (`val_top10_norm` / `val_mrr_norm`), with a **paired bootstrap** vs `shuffled`/`random` controls. `within_val` ranking inflates signal and is diagnostic only.

### Phase History at a Glance

| Phase | Status | Key Outcome |
|---|---|---|
| Phase 1–2 | ✅ Complete | ZUNA inference, timing audit, retrieval grid |
| Phase 3.5 | ✅ Complete | Back-aligned ZUNA windows beat shuffled/random |
| Phase 4 | ✅ Complete | EPOC-14 sim: ZUNA retains ~74% of full 64-ch signal |
| Phase 5–9 | ✅ Complete | Coordinate-aware encoder, VLM semantic front, FAISS index, FiLM adapters, frozen probe |
| Phase 10–16 | ✅ Complete | Retrieval branch, calibration stimuli, multi-subject scaling, full-bank eval |
| Phase 17 | ✅ Complete | DINOv2-RAE decoder swap; established RAE as the reconstruction target |
| Phase 18B–18E | ⛔ Deprecated | EEG→RAE 4×4 *code bottleneck* + expander-aligned loss. Abandoned: codes lacked per-site fidelity to survive non-linear expansion; warm-start Δ collapsed to ~0. **Superseded by the QFormer bridge.** |
| **QFormer Bridge** | 🚧 In Progress | ZUNA `post_mmd` → QFormer → RAE (DINO-Unit-768) retrieval grid with paired bootstrap controls |

> **Why we left the 18B–18E code-bottleneck path**: forcing EEG into a tiny `768×4×4` RAE code and then expanding it discarded the per-channel/per-site fidelity RAE's decoder needs. Rather than fight the bottleneck, we now learn the ZUNA→vision mapping directly with a QFormer and rank against the full RAE bank.

---

## 2. Remote Environment & SSH Access

All GPU training runs on **RunPod Secure Cloud**. The local dev machine has no GPU. `data/` and `outputs/` are **not** tracked in git — they live on the pod's persistent volume.

### Pod Management (via MCP)
```bash
mcp_runpod_list-pods
mcp_runpod_start-pod  {"podId": "<POD_ID>"}
mcp_runpod_stop-pod   {"podId": "<POD_ID>"}
```

### Provisioning a Fresh Pod
```json
{
  "cloudType": "SECURE",
  "gpuCount": 1,
  "gpuTypeIds": ["NVIDIA A40"],
  "volumeInGb": 200,
  "containerDiskInGb": 100,
  "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
  "name": "mindeye-qformer",
  "ports": ["22/tcp"],
  "volumeMountPath": "/workspace",
  "env": {"PUBLIC_KEY": "<contents of ~/.ssh/runpod.pub>"}
}
```
> The MCP `create-pod` cannot attach an existing network volume → MCP pods get a fresh empty volume
> (bootstrap via `cold_start.sh`). Reuse a persistent volume via the RunPod web UI. See `docs/INFRA.md`.

### SSH & Setup
Two keys: **`~/.ssh/runpod`** = laptop→pod; **`~/.ssh/id_ed25519`** = GitHub (SSH git URL always).
The pod has no GitHub key → forward the agent (`ssh -A` after `ssh-add ~/.ssh/id_ed25519`).
```bash
ssh-add ~/.ssh/id_ed25519
ssh -A root@<IP> -p <PORT> -i ~/.ssh/runpod -o StrictHostKeyChecking=no

# On pod: clone / pull latest code over the SSH git URL (uses forwarded key)
cd /workspace && git clone git@github.com:sudoaza/mindeye.git mindeye
# or: cd /workspace/mindeye && git pull origin master

# Install deps. Image ships torch 2.8.0+cu128 — do NOT reinstall torch.
# Install only non-torch deps (--break-system-packages for Ubuntu 24.04 PEP 668):
cd /workspace/mindeye
grep -vE '^(--extra-index-url|torch==|torchvision==|torchaudio==|nvidia-cudnn|$)' requirements.txt > /tmp/reqs_notorch.txt
pip install --break-system-packages -r /tmp/reqs_notorch.txt

export PYTHONPATH=src   # always prefix this
```

> **CAUTION**: Never `rsync` local → remote without pulling `outputs/` first. Remote checkpoints will be silently overwritten.

---

## 3. Canonical Architecture (QFormer Bridge)

```
EEG (256 Hz, 5s, 64 ch)
  └─► ZUNA diffusion denoiser (frozen)  [scripts/run_zuna_batch.py]
        └─► cache post_mmd latents  [scripts/cache_zuna_latents.py]
              shape per epoch: [2480, 32]  = [62 ch × 40 tc, 32 d]
              └─► onset crop tc[20:36)  →  [992, 32]
                    (62 ch × 16 frames; [-0.5s, +1.5s] @ 125 ms/frame, onset_tc≈24)
                    └─► ZunaToVisionQFormer  [src/mindseye/adapters/qformer.py]
                          • input_proj: 32 → 256 (Linear+LN+GELU+Dropout)
                          • 32 learnable query tokens (+1 CLS); optional subject FiLM
                          • 4× QFormer blocks:
                              self-attn(queries) → cross-attn(queries → ZUNA kv) → FFN
                          • CLS readout → proj_head → d_out → LayerNorm → L2-normalize
                          └─► vision embedding  (RAE / DINO-768 target)
```

**Loss**: InfoNCE (contrastive) + cosine + variance-floor (anti-collapse, weight 0.05).

**Evaluation**: predictions ranked against the full RAE/DINO image bank (`rae_unit`).
Every target is always evaluated against the **true** image embedding (`eval_target`), even in shuffled/random training modes.

### ⚠️ Open architectural gap — reconstruction vs retrieval

The current QFormer pools to a **single vector** `[B, d_out]` (CLS readout). This is a **retrieval** bridge.
RAE *image reconstruction* needs the `[768, 16, 16]` token grid, not a pooled vector. Closing the
retrieval→reconstruction gap (predicting the RAE token grid, or a code the RAE decoder can expand) is
the next architectural decision once the retrieval gate is consistently beaten. Do not wire up the RAE
decoder for image generation until then.

---

## 4. Key Data Paths (on pod at `/workspace/mindeye/`)

| Artifact | Path |
|---|---|
| Raw EEG BIDS | `data/raw/nod/sub-0{1..4}/` |
| ZUNA FIF outputs | `data/processed/zuna_real/4_fif_output/` |
| Cropped epochs (tight1s) | `data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/` |
| **Cached ZUNA latents** | `data/processed/zuna_latents/sub01_runs01_32/` (`latents_post_mmd.pt`, `metadata.pt`) |
| NOD stimuli (ImageNet) | `data/raw/nod/stimuli/ImageNet/` |
| **RAE/DINOv2 embedding bank** (target) | `data/processed/rae_embeddings/rae_dinov2_base_sub01_04_runs01_40.pt` |
| VLM attributes | `data/processed/clip_embeddings/vlm_attributes.json` |
| QFormer grid outputs | `outputs/qformer_aligned_grid/grid_<timestamp>/` |

---

## 5. Resuming — Cold Start

`cold_start.sh` automates the full lifecycle end-to-end (env → download → ZUNA → crop → targets →
cache latents → QFormer grid). If the persistent volume already has data/checkpoints, skip to the
relevant step. Defaults to sub-01 runs 1-8 for dev; widen runs/subjects for a full run.

```bash
bash cold_start.sh
```

Key individual steps (see `cold_start.sh` and `scripts/README.md` for the full canonical order):

### Cache ZUNA latents (QFormer input)
```bash
python scripts/cache_zuna_latents.py \
    --epochs-dir data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_08 \
    --output-dir data/processed/zuna_latents/sub01_runs01_08 \
    --layers post_mmd
```

### Build target banks
```bash
# RAE / DINOv2 bank (reconstruction target)
python scripts/build_rae_latent_bank.py \
    --image-dir data/raw/nod/stimuli/ImageNet \
    --output data/processed/rae_embeddings/rae_dinov2_base_sub01_04_runs01_40.pt
```

### Run the QFormer bridge grid
```bash
python scripts/run_qformer_grid.py \
    --latents-pt data/processed/zuna_latents/sub01_runs01_32 \
    --rae-pt     data/processed/rae_embeddings/rae_dinov2_base_sub01_04_runs01_40.pt \
    --train-runs 1-24 --val-runs 25-28 --test-runs 29-32 \
    --epochs 40 --patience 8 --batch-size 64 --lr 3e-4 \
    --device cuda \
    --out-dir outputs/qformer_aligned_grid
```

The grid trains, for each target space, all three control modes (`real` / `shuffled` / `random`) and
then runs the paired bootstrap. Target spaces (all RAE/DINO — **CLIP has been dropped**):
- `DINO-Unit-768` — primary RAE target
- `DINO-PCA-256-Unit`, `DINO-PCA-128-Unit` — lower-rank DINO targets (is a reduced target easier to hit?)

Smoke test (RAE DINO-Unit-768 only, runs 1-6 train / 7-8 val, fast):
```bash
python scripts/run_qformer_grid.py --smoke-test \
    --latents-pt <...> --rae-pt <...> --device cuda
```

### Gate (both must hold on full-set retrieval)
- Paired bootstrap Δ (real − shuffled) > +0.005 with 95% CI excluding 0.
- `collapse_pct` < 20% (variance-floor working; no dimensional collapse).

---

## 6. Beyond the Retrieval Gate (Deferred)

| Step | Condition | Description |
|---|---|---|
| **RAE token-grid bridge** | After retrieval gate passes | Extend QFormer to predict the RAE `[768,16,16]` token grid (or a faithfully-expandable code) — the actual reconstruction bridge. |
| **RAE decode** | After token-grid bridge | EEG → QFormer → RAE token grid → frozen RAE decoder → image. |
| **FAISS kNN priors** | After gap ≥ 0.02 | BReAD-style retrieval priors for visual grounding. |
| **Frozen diffusion** | After visual grids clear | img2img refinement on RAE decode. |

---

## 7. Non-Negotiable Rules

1. **Full-set retrieval is the only honest metric.** `within_val` ranking is diagnostic only (inflated).
2. **Run controls every time.** Every experiment includes `real / shuffled / random` and reports all three in the paired-bootstrap table. This split has repeatedly caught data/pipeline bugs.
3. **Do not add the RAE decoder / diffusion** until the QFormer retrieval gate (Δ > +0.005, CI excludes 0) is consistently met.
4. **ZUNA and RAE are frozen.** Only the QFormer bridge trains.
5. **Onset back-alignment is assumed.** Epochs are cropped to stimulus onset, so the fixed latent window `tc[20:36)` is correct. If alignment changes, revisit `crop_zuna_latent`.
6. **`PYTHONPATH=src` must be set** before any `python scripts/` call on the pod.
7. **Never mix target spaces across checkpoints.** Confirm a checkpoint's target space before any warm-start.

---

## 8. Key Findings & Lessons Learned

### Signal is real but fragile
- ZUNA latents carry decodable visual signal, but the margin over controls is small — controls are mandatory to distinguish real signal from leakage.
- `within_val` ranking inflates apparent performance; a model below random on full-set can look strong within-val.

### Why the code-bottleneck path (18B–18E) was abandoned
- Compressing EEG into a `768×4×4` RAE code and expanding it discarded per-channel/per-site fidelity the RAE decoder needs.
- Even after fixing scale inflation (18D), the expanded-token cosine gap collapsed to ~0 — the expander is scale-invariant, so matching scale didn't help.
- Conclusion: learn the ZUNA→vision mapping directly (QFormer) and keep RAE for what it's best at (reconstruction), rather than squeezing EEG through a lossy code.

### Why QFormer, not a simple adapter
- Linear / MLP adapters from ZUNA latents to vision space underperformed.
- QFormer's learnable query tokens + cross-attention over the full ZUNA token sequence gives the bridge capacity to select relevant ZUNA tokens, which simple pooling/projection cannot.

### Architecture choices recap
- **ZUNA** for embedding (EEG foundation model, best feature recovery).
- **RAE** for reconstruction (best visual fidelity; CLIP/ViT semantically ok but visually imprecise).
- **QFormer** as the bridge between them.

### Bootstrap protocol
- 10,000-iteration paired bootstrap over per-sample deltas aligned by `sample_id`.
- Gate: `mean_δ > +0.005` AND CI lower bound > 0.

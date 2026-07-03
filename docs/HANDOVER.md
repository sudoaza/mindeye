# MindEye Handover Document

> **Current Phase**: QFormer Bridge — ZUNA → QFormer → RAE retrieval grid
> **Updated**: 2026-07-03
> **Active run**: 9-subject full-cohort grid on RunPod pod `0w6hgf17v0xs46` (A100 80GB) — see §0.

This document summarises the current project state, remote environment, and step-by-step instructions to resume work. Keep this file updated when major milestones are reached.

---

## 0. Right Now — What's Running (read this first)

**Goal of this run**: past experience is that positive results only emerge at scale, so we are training a **single combined-cohort QFormer** over **9 subjects × 32 runs** (not per-subject models). Cohort is `sub-01..sub-09` — these are the only NOD ds005811 subjects with full 32-run (4 session × 8) coverage; `sub-10+` have just 16 runs, so there are **no 40-run subjects** (the old `_runs01_40` naming was always a misnomer).

**Where**: RunPod Secure Cloud pod **`0w6hgf17v0xs46`**, A100 80GB PCIe, EU-RO-1, 200GB volume at `/workspace`, torch 2.8.0+cu128. Repo at `/workspace/mindeye`, HEAD should be the latest `master`.

**How it was launched** (backgrounded, survives disconnects):
```bash
cd /workspace/mindeye
export HF_HOME=/workspace/hf_cache
nohup env SKIP_ENV=1 HF_HOME=$HF_HOME bash scripts/prepare_multisubject_data.sh > /workspace/cohort9.log 2>&1 &
```

**Progress at last update** (2026-07-03, checked live): step **[3/7] cropping** — sub-01..04 done, now cropping **sub-05** of 9 (epochs correctly `[N, 62, 1281]`). Steps already complete and reusable if the run dies: raw download (~6GB), 288 ZUNA-denoised runs (`data/processed/zuna_real/4_fif_output/`), the 14GB RAE bank (`rae_dinov2_base_all.pt`), and 35,466 stimulus images. ZUNA denoising **skips already-processed files**, so relaunching is cheap.

**Remaining steps**: [4/7] stimulus sync (skips, images present) → [5/7] rebuild RAE bank → [6/7] cache ZUNA latents into ONE merged cohort dir → [7/7] QFormer grid (real/shuffled/random × DINO targets, `--num-subjects 9`).

**Monitor / drive the pod** via the `pod-exec` MCP (see §2) — e.g. tail `/workspace/cohort9.log`, check `outputs/qformer_cohort9_grid/`.

**Next meaningful checkpoint**: `cache_zuna_latents` finishing (it is the step that previously crashed — see §8), then the grid emitting `val_top10_norm` / paired-bootstrap numbers.

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

### `pod-exec` MCP — run commands on the pod without SSH-approval prompts

There is a small custom MCP server that execs arbitrary shell commands on the pod over SSH, so the agent can drive the pod without a local execution-approval prompt per command.

- **Server code**: `~/.cursor/mcp-servers/pod-exec/server.py` (dependency-free stdio JSON-RPC; runs under system `python3`).
- **Connection config**: `~/.cursor/mcp-servers/pod-exec/pod.json` — `{host, port, user, key, forward_agent, default_cwd}`. Update `host`/`port` here whenever the pod changes; no code edit or Cursor restart needed for connection changes.
- **Registered in** `~/.cursor/mcp.json` as server `pod-exec`; Cursor namespaces the tools as **`user-pod-exec`**.
- **Tools**: `pod_exec {command, cwd?, timeout?}` and `pod_config {host?, port?, ...}` (read/update connection).
- Long-running commands must be backgrounded on the pod (`nohup ... &`) since `pod_exec` blocks until the command returns or hits `timeout` (default 600s).
- Current pod.json points at `0w6hgf17v0xs46` (see §0). If the tool isn't listed, reload MCP servers in Cursor Settings.

### Pod Management (via RunPod MCP, server `user-runpod`)
```
user-runpod-list-pods / get-pod / create-pod / start-pod / stop-pod / delete-pod
user-runpod-list-network-volumes / create-network-volume   (volume tools now exist)
```

### Provisioning a Fresh Pod (A100 recommended for full-cohort runs)
```json
{
  "cloudType": "SECURE",
  "computeType": "GPU",
  "gpuCount": 1,
  "gpuTypeIds": ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"],
  "volumeInGb": 200,
  "containerDiskInGb": 60,
  "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
  "name": "mindeye-cohort9",
  "ports": ["22/tcp"],
  "volumeMountPath": "/workspace",
  "env": {"PUBLIC_KEY": "<contents of ~/.ssh/runpod.pub>"}
}
```
> **Sizing reality check**: NOD raw EEG is small — ~605 MB/subject, so 9 subjects raw ≈ 6 GB. Processed ZUNA output + the 14 GB RAE bank + cached latents fit comfortably in **200 GB**. The old "~20 GB/subject" estimate was wildly off; do not over-provision the volume (large volumes also fail with *"no instances available with enough disk space"*).
> **create-pod caveats**: (1) It cannot attach an existing network volume → MCP pods get a fresh empty volume; reuse a persistent volume via the RunPod web UI. (2) v2 accepts **one** GPU type per pod (extra `gpuTypeIds` are ignored). (3) Very large `volumeInGb`/`containerDiskInGb` requests fail on host availability — keep them modest. See `docs/INFRA.md`.

### SSH & Setup
Two keys: **`~/.ssh/runpod`** = laptop→pod; **`~/.ssh/id_ed25519`** = GitHub (SSH git URL always).
The pod has no GitHub key → forward the agent (`ssh -A`, after `ssh-add ~/.ssh/id_ed25519` into the *active* agent).
```bash
ssh-add ~/.ssh/id_ed25519
ssh -A root@<IP> -p <PORT> -i ~/.ssh/runpod -o StrictHostKeyChecking=no

# On pod: git over SSH needs relaxed host-key checking on a fresh pod, or the
# clone fails with "Host key verification failed" even though `ssh -T` worked:
export GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null'
cd /workspace && git clone git@github.com:sudoaza/mindeye.git mindeye
# or: cd /workspace/mindeye && git fetch origin && git reset --hard origin/master

# Install deps. Image ships torch 2.8.0+cu128 — do NOT let anything change it.
cd /workspace/mindeye
grep -vE '^(--extra-index-url|torch==|torchvision==|torchaudio==|nvidia-cudnn|$)' requirements.txt > /tmp/reqs_notorch.txt
pip install --break-system-packages -r /tmp/reqs_notorch.txt

# ⚠️ A transitive dep (zuna/lm-eval chain) will pull torch 2.12.1 and break the
# cu128 stack (torchvision/torchaudio pin torch==2.8.0). Re-pin torch afterwards:
pip install --break-system-packages torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128
python3 -c "import torch,torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"

export PYTHONPATH=src   # always prefix this
```

> **CAUTION**: Never `rsync` local → remote without pulling `outputs/` first. Remote checkpoints will be silently overwritten.

---

## 3. Canonical Architecture (QFormer Bridge)

```
EEG (256 Hz, 5s, 64 ch)
  └─► ZUNA diffusion denoiser (frozen)  [scripts/run_zuna_batch.py]
        └─► crop 5s BACK-ALIGNED epochs  [scripts/run_cropper.py --full5s-backaligned]
              window [-3.0s, +2.0s] → 1280 samples @ 256 Hz, onset at sample 768
              (⚠️ NOT the tight -0.2/+1.0 window — see §8 crop-window bug)
              └─► cache post_mmd latents  [scripts/cache_zuna_latents.py]
                    shape per epoch: [2480, 32]  = [62 ch × 40 tc, 32 d]
                    └─► onset crop tc[20:36)  →  [992, 32]
                          (62 ch × 16 frames; onset_tc≈24)
                          └─► ZunaToVisionQFormer  [src/mindseye/adapters/qformer.py]
                                • input_proj: 32 → 256 (Linear+LN+GELU+Dropout)
                                • 32 learnable query tokens (+1 CLS); subject FiLM (num_subjects>1)
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

Cohort = `sub-01..sub-09`, runs 1-32. `RUN_TAG=runs01_32`.

| Artifact | Path |
|---|---|
| Raw EEG BIDS | `data/raw/nod/derivatives/preprocessed/raw/sub-0{1..9}_ses-ImageNet0{1..4}_...fif` |
| ZUNA FIF outputs | `data/processed/zuna_real/4_fif_output/` (288 files = 9 × 32) |
| **Cropped epochs (5s back-aligned)** | `data/processed/semantic_epochs/zuna_full5s_backaligned_sub0N_runs01_32/` (one per subject; `[N, 62, 1281]`) |
| **Cached ZUNA latents (merged cohort)** | `data/processed/zuna_latents/cohort9_runs01_32/` (`latents_post_mmd.pt`, `metadata.pt`) |
| NOD stimuli (ImageNet) | `data/raw/nod/stimuli/ImageNet/` (~35,466 images) |
| **RAE/DINOv2 embedding bank** (target, cohort-wide, keyed by `image_id`) | `data/processed/rae_embeddings/rae_dinov2_base_all.pt` (~14 GB) |
| QFormer grid outputs | `outputs/qformer_cohort9_grid/grid_<timestamp>/` |
| Run log | `/workspace/cohort9.log` |

> The RAE bank is **not** subject-specific — it maps every stimulus `image_id` to its DINOv2 embedding, shared across all subjects. The old `rae_dinov2_base_sub01_04_runs01_40.pt` name was misleading; use `_all`.

---

## 5. Resuming — Full-Cohort Run & Cold Start

### Full-cohort pipeline (the current canonical run) — `scripts/prepare_multisubject_data.sh`

This is the end-to-end **combined 9-subject** runner: download → ZUNA denoise (single global pass) → per-subject 5s back-aligned crop → stimulus include-list + sync → single RAE bank → **merged** latent cache → combined QFormer grid with `--num-subjects 9`. It is env-configurable (see the vars at the top) and does **not** create a venv (`SKIP_ENV`-style: run on a pod with deps already installed).

```bash
cd /workspace/mindeye
export HF_HOME=/workspace/hf_cache
nohup env SKIP_ENV=1 HF_HOME=$HF_HOME bash scripts/prepare_multisubject_data.sh > /workspace/cohort9.log 2>&1 &
```

Every step is idempotent/resumable: download re-verifies existing files, ZUNA skips already-processed runs, so a relaunch after a crash is cheap. Override cohort/paths via env, e.g. `SUBJECTS="sub-01 sub-02" RUNS_SPEC=1-8 RUNS_SEQ="$(seq 1 8)" RUN_TAG=runs01_08 bash scripts/prepare_multisubject_data.sh` for a quick dev slice.

### Single-subject dev cold start — `cold_start.sh`

`cold_start.sh` is the smaller single-subject dev lifecycle (sub-01, runs 1-8). Use `SKIP_ENV=1` to reuse an installed pod env. Prefer `prepare_multisubject_data.sh` for real runs — **positive results have historically required the full multi-subject scale.**

Key individual steps (canonical order):

### Cache ZUNA latents (QFormer input) — merged cohort
`cache_zuna_latents.py` accepts **multiple** `--epochs-dir` and merges them into one output (unique `sample_id` per subject now that the cropper writes a `subject` column — see §8).
```bash
python scripts/cache_zuna_latents.py \
    --epochs-dir data/processed/semantic_epochs/zuna_full5s_backaligned_sub0{1..9}_runs01_32 \
    --output-dir data/processed/zuna_latents/cohort9_runs01_32 \
    --layers post_mmd --device cuda
```

### Build target bank
```bash
# RAE / DINOv2 bank (reconstruction target), keyed by image_id, cohort-wide
python scripts/build_rae_latent_bank.py \
    --image-dir data/raw/nod/stimuli/ImageNet \
    --output data/processed/rae_embeddings/rae_dinov2_base_all.pt
```

### Run the QFormer bridge grid (combined cohort)
```bash
python scripts/run_qformer_grid.py \
    --latents-pt data/processed/zuna_latents/cohort9_runs01_32 \
    --rae-pt     data/processed/rae_embeddings/rae_dinov2_base_all.pt \
    --num-subjects 9 \
    --train-runs 1-24 --val-runs 25-28 --test-runs 29-32 \
    --epochs 40 --patience 8 --batch-size 64 --lr 3e-4 \
    --device cuda \
    --out-dir outputs/qformer_cohort9_grid
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
5. **Epochs for ZUNA latent caching MUST be 5s back-aligned** (`run_cropper.py --full5s-backaligned`, window `-3.0/+2.0`, 1281 samples, onset at sample 768). The `ZunaLatentExtractor` hard-asserts 1280 timepoints. The tight `-0.2/+1.0` window belongs to the deprecated semantic-classifier path and will crash caching. Because onset is back-aligned, the fixed latent window `tc[20:36)` is correct.
6. **`PYTHONPATH=src` must be set** before any `python scripts/` call on the pod.
7. **Never mix target spaces across checkpoints.** Confirm a checkpoint's target space before any warm-start.
8. **Combined cohort, not per-subject models.** Positive results have needed multi-subject scale; train one QFormer over the merged cohort with subject FiLM (`--num-subjects 9`), not 9 separate models.
9. **Keep torch at 2.8.0+cu128 on the pod.** Deps try to upgrade it and break the CUDA stack; re-pin after any `pip install` (see §2).

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

### Operational lessons from the 9-subject run (2026-07-03)

- **Crop-window bug**: the multi-subject pipeline initially cropped tight `-0.2/+1.0` (~308-sample) epochs (copied from the deprecated semantic path). `cache_zuna_latents.py` → `ZunaLatentExtractor` asserts 1280 timepoints and crashed with *"Expected 1280 timepoints, got 308"*. Fix: `--full5s-backaligned`. Lesson: the ZUNA latent path and the old semantic-classifier path use **different** crop windows; don't copy invocations between them.
- **Dataset reality (NOD ds005811)**: 30 subjects total, but only **sub-01..sub-09 have 32 runs** (4 sessions × 8); sub-10+ have 16. There are **no 40-run subjects** — historical `_runs01_40` names are misnomers. Verify coverage before assuming a cohort size.
- **Data is small**: ~605 MB raw per subject (run FIF ~11 MB, subject epoch FIF ~215 MB). 9 subjects raw ≈ 6 GB. The RAE bank (~14 GB) dominates disk. 200 GB volume is ample; don't over-request (large volumes fail on host availability).
- **Multi-subject correctness fixes** (this run):
  - `cropper.py` now writes a numeric `subject` column (+ `subject_label`). Previously every trial defaulted to `subject_id=1`, which (a) disabled subject FiLM and (b) caused `sample_id` collisions that silently overwrote latents when merging subjects.
  - `cache_zuna_latents.py` now takes multiple `--epochs-dir`, groups by `(dir, npz_file)`, and merges into one cohort dir without collisions.
  - `run_qformer_grid.py` now forwards `--num-subjects` so subject FiLM actually activates.
- **Run splits are global & shared** across subjects (train runs 1-24 / val 25-28 / test 29-32 apply to every subject). No image-disjoint guard: the same `image_id` can land in train for one subject and val for another. Acceptable for now; revisit if leakage is suspected.
- **Torch pin**: the image ships torch 2.8.0+cu128, but installing `requirements.txt` deps pulls torch 2.12.1 via the zuna/lm-eval chain and breaks torchvision/torchaudio + CUDA libs. Always re-pin torch/vision/audio to 2.8.0 (cu128) after dep install.
- **Infra ergonomics**: git clone on a fresh pod needs `GIT_SSH_COMMAND` with relaxed host-key checking even after `ssh -T` succeeds. The `pod-exec` MCP (see §2) removes per-command SSH-approval friction and is the preferred way to drive the pod. Backgrounded (`nohup`) runs survive Cursor/backend disconnects.

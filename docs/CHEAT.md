# MindEye Dev Cheat Sheet

> IMPORTANT: All GPU work runs on the **remote RunPod pod**. The dev machine has no GPU and no data.
> **Never run pipeline steps locally** — always spin up / start a pod and run over SSH.
> Doc index: [`README.md`](README.md). Pod sizing, network-volume strategy, and provisioning
> live in [`INFRA.md`](INFRA.md). runpod MCP tool reference: [`RunPod_SKILL.md`](RunPod_SKILL.md).

## Index
- [RunPod Management](#runpod-management)
- [SSH & File Transfer](#ssh--file-transfer)
- [Running the Pipeline](#running-the-pipeline)
- [Background Execution & Logging](#background-execution--logging)
- [Common Pitfalls & Troubleshooting](#common-pitfalls--troubleshooting)
  - [EEG Model Architecture](#eeg-model-architecture--training)
  - [Pipeline, Data & Scripts](#pipeline-data--script-execution)
  - [RunPod & Infrastructure](#runpod--infrastructure)


---

## RunPod Management

Use the `runpod` MCP tools. **Pod sizing, provisioning JSON, and the network-volume
detach/reattach workflow are in [`INFRA.md`](INFRA.md).** Full tool reference in
[`RunPod_SKILL.md`](RunPod_SKILL.md).

```bash
mcp_runpod_list-pods
mcp_runpod_get-pod    {"podId": "<POD_ID>", "includeMachine": true}
mcp_runpod_start-pod  {"podId": "<POD_ID>"}
mcp_runpod_stop-pod   {"podId": "<POD_ID>"}
```

---

## SSH & File Transfer

**Two keys**: `~/.ssh/runpod` = laptop→pod (`ssh -i ~/.ssh/runpod`); `~/.ssh/id_ed25519` = GitHub.
Always use the **SSH git URL** (`git@github.com:sudoaza/mindeye.git`). The pod has no GitHub key →
**forward the agent**: `ssh-add ~/.ssh/id_ed25519` then `ssh -A ... -i ~/.ssh/runpod`.

### Preferred: `pod-exec` MCP (no SSH-approval prompts)
The custom `pod-exec` MCP (tools namespaced `user-pod-exec`; `pod_exec {command, cwd?, timeout?}`)
runs any command on the pod over SSH without a local approval prompt. Set the pod host/port in
`~/.cursor/mcp-servers/pod-exec/pod.json` (or via the `pod_config` tool). Background long jobs on the
pod with `nohup ... &` — they survive Cursor/backend disconnects. See [`HANDOVER.md`](HANDOVER.md) §2.

### git pull on pod (manual SSH)
```bash
ssh-add ~/.ssh/id_ed25519
# Fresh pod needs relaxed host-key checking or clone fails ("Host key verification failed"):
ssh -A -p <PORT> root@<IP> -i ~/.ssh/runpod \
  "cd /workspace/mindeye && GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' git fetch origin && git reset --hard origin/master"
```

### Rsync (last resort — debugging only)
```bash
# Install rsync first if needed:
ssh -p <PORT> root@<IP> "apt-get update && apt-get install -y rsync"

# Push code (exclude data and outputs to avoid clobbering remote checkpoints):
rsync -avz --no-o --no-g --no-perms \
  --exclude 'venv' --exclude '.git' --exclude '__pycache__' \
  --exclude 'outputs/' --exclude 'data/' \
  -e "ssh -p <PORT>" . root@<IP>:/workspace/mindeye/

# Pull remote outputs/checkpoints first:
rsync -avz --no-o --no-g --no-perms \
  -e "ssh -p <PORT>" \
  root@<IP>:/workspace/mindeye/outputs/ outputs/
```

> [!CAUTION]
> **Never rsync local → remote without pulling outputs first.** Any `outputs/` generated on the pod (checkpoints, probe models, matrix results) will be silently overwritten by stale local copies.

### Install Dependencies
```bash
# The pod image ships torch 2.8.0+cu128 — install only non-torch deps, then RE-PIN torch
# (a transitive dep upgrades it to 2.12.1 and breaks the cu128 stack). --break-system-packages
# for Ubuntu 24.04 PEP 668:
ssh -p <PORT> root@<IP> -i ~/.ssh/runpod \
  "cd /workspace/mindeye && \
   grep -vE '^(--extra-index-url|torch==|torchvision==|torchaudio==|nvidia-cudnn|\$)' requirements.txt > /tmp/reqs_notorch.txt && \
   pip install --break-system-packages -r /tmp/reqs_notorch.txt && \
   pip install --break-system-packages torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128"
```

---

## Running the Pipeline

Current architecture: **ZUNA → QFormer → RAE** (see [`PLAN.md`](PLAN.md), [`HANDOVER.md`](HANDOVER.md)).
For step-by-step details see [`scripts/README.md`](../scripts/README.md). Always run on the pod with
`export PYTHONPATH=src` set (see [`INFRA.md`](INFRA.md) for the full env block).

### Cold start (end-to-end)
```bash
# Full 9-subject cohort (recommended — positive results need scale):
SKIP_ENV=1 bash scripts/prepare_multisubject_data.sh   # download → ZUNA → 5s crop → RAE bank → merged latents → grid
# Single-subject dev slice:
SKIP_ENV=1 bash cold_start.sh
```

### Cache ZUNA latents (QFormer input) — merged multi-subject cohort
```bash
# Epochs MUST be 5s back-aligned (1280 samples); accepts multiple --epochs-dir (merged, no collisions):
python scripts/cache_zuna_latents.py \
  --epochs-dir data/processed/semantic_epochs/zuna_full5s_backaligned_sub0{1..9}_runs01_32 \
  --output-dir data/processed/zuna_latents/cohort9_runs01_32 \
  --layers post_mmd --device cuda
```

### Run the QFormer bridge grid (real / shuffled / random + paired bootstrap)
```bash
python scripts/run_qformer_grid.py \
  --latents-pt data/processed/zuna_latents/cohort9_runs01_32 \
  --rae-pt     data/processed/rae_embeddings/rae_dinov2_base_all.pt \
  --num-subjects 9 \
  --train-runs 1-24 --val-runs 25-28 --test-runs 29-32 \
  --epochs 40 --patience 8 --batch-size 64 --lr 3e-4 \
  --device cuda --out-dir outputs/qformer_cohort9_grid
```

Target spaces (all RAE/DINO — **CLIP dropped**): `DINO-Unit-768` (primary RAE target),
`DINO-PCA-256-Unit`, `DINO-PCA-128-Unit`. Smoke test: add `--smoke-test` (DINO-Unit-768 only, runs 1-6/7-8).

**Gate**: paired Δ (real − shuffled) > +0.005 with 95% CI excluding 0, `collapse_pct` < 20%,
on full-set retrieval against the RAE bank.

> **Deprecated command paths** (kept for reference only, not the live plan): the `make matrix` /
> decode_unit unCLIP branch (Phase 12A/13) and the RAE code-bottleneck branch (`train_rae_token_bottleneck.py`,
> Phase 17/18). See [`PLAN.md`](PLAN.md) §6 for why the code-bottleneck path was abandoned.

On the pod, prefix direct script calls with `export PYTHONPATH=src &&`:
```bash
ssh -p <PORT> root@<IP> "cd /workspace/mindeye && export PYTHONPATH=src && nohup python scripts/run_qformer_grid.py ... > qformer_grid.log 2>&1 &"
```

---

## Background Execution & Logging

Long jobs **must** use `nohup` to survive SSH disconnection:

```bash
nohup make matrix > matrix.log 2>&1 &
nohup bash run_epoc_simulation.sh > epoc14.log 2>&1 &
```

### Process Management
```bash
ps aux | grep python               # Check running jobs
tail -f /workspace/mindeye/matrix.log  # Follow log live
pkill -f 'python.*run_baseline_matrix' # Kill matrix run
```

### Output Locations
| Artifact | Location |
|---|---|
| Console logs | `/workspace/mindeye/*.log` (e.g. `matrix.log`, `epoc14.log`) |
| Per-run metrics | `outputs/runs/<timestamp>_<slug>/metrics.json` |
| Per-epoch CSV | `outputs/runs/<timestamp>_<slug>/train_log.csv` |
| Matrix summary | `outputs/baseline_matrix/<timestamp>_matrix/matrix_summary.csv` |
| Probe model (decode space v2) | `outputs/decode_probe_v2/common_probe.pt` |
| Probe model (old z_common)    | `outputs/common_probe/common_probe.pt` |

**Key `metrics.json` fields**: `top1`, `top5`, `top10`, `mrr`, `median_rank`, `mean_diag_cosine`, `collapse_score`, `best_epoch`, `full_bank_top10`, `full_bank_mrr`.

> **Primary gate metric is `full_bank_top10`** (predictions ranked against all 4000 images, expected random = 0.0025). `within_val_top10` inflates signal by ~20–4× and is diagnostic only. `collapse_score` must be **> 0.1** to pass baseline gates.

---

## Common Pitfalls & Troubleshooting

### EEG Model Architecture & Training

- **BatchNorm → GroupNorm**: `BatchNorm1d` in temporal/spatial stems causes train/eval discrepancy on small-batch EEG, collapsing validation performance. Use `GroupNorm` instead — it computes statistics per-sample.
- **Augmentation on marker channel**: The event marker channel (last channel, e.g. index 63) must be **excluded** from noise/masking augmentations. Corrupting it destroys event-EEG alignment.
- **Control label leakage**: In `shuffled`/`random` conditions, auxiliary probe labels must be shuffled/randomized *identically* to the main contrastive target. Otherwise control scores are artificially inflated.
- **Collapse guard in checkpoint saving**: Never select checkpoints purely on MRR/Top-10 — the model may have collapsed to a constant vector. Enforce `collapse_score > 0.1` (ideally > 1.0) as a hard gate before saving.
- **Channel count mismatch after checkpoint load**: A model trained with N channels (e.g. 63 + event marker = 64) will crash at inference if the evaluation dataset returns N-1 channels. Ensure `--add-event-marker` is passed consistently between training and evaluation. Error: `RuntimeError: expected input to have 63 channels, but got 62`.
- **`return_features` API signature**: Models used with downstream multitask heads must support `return_features: bool = False` in `forward()`, returning `(output, features)` when `True`.

### Pipeline, Data & Script Execution

- **VLM output field name mismatch**: If the VLM generates field `structured_embedding_text` but the parser reads `embedding_text`, the fallback becomes `"empty"` for every image — collapsing the entire target embedding space to a single point. Always verify JSON key names end-to-end.
- **CLIP model output type**: `CLIPModel.get_image_features()` returns a plain tensor, but `CLIPModel()` (full forward pass) returns a `BaseModelOutputWithPooling` object. Calling `.norm()` on the latter raises `AttributeError`. Use `.image_embeds` or call `get_image_features()` directly.
- **Missing `argparse` attribute at runtime**: Adding a new CLI flag to a script but forgetting to add it to a downstream caller (e.g. `evaluate_retrieved_priors.py` missing `--num-grid`) causes `AttributeError: 'Namespace' object has no attribute 'num_grid'`. Always add defaults to `parse_args()` and keep CLI consistent across callers.
- **CSV fieldname drift in training loop**: If new probe metrics (e.g. `calib_probe_class_label_top10_acc`) are added to the row dict but not to the `fieldnames` list passed to `csv.DictWriter`, training crashes at the end of the first epoch with `ValueError: dict contains fields not in fieldnames`. Keep `log_fields` and the metrics dict in sync.
- **Crop window is path-specific (don't cross the streams)**: The **ZUNA-latent path** (`cache_zuna_latents.py` → `ZunaLatentExtractor`) requires **5s back-aligned** epochs = **1280 samples** (`run_cropper.py --full5s-backaligned`, window `-3.0/+2.0`, onset at sample 768). It hard-asserts 1280 timepoints and crashes with *"Expected 1280 timepoints, got 308"* if fed tight epochs. The tight `-0.2/+1.0` (1.2s, ~308-sample) window below belongs to the **deprecated semantic-classifier path only** — do not copy it into the QFormer pipeline.
- **Stimulus overlap in RSVP / wide windows (semantic-classifier path only)**: NOD presents images every ~1.3–1.7s. For the old semantic classifier, wide windows capture multiple stimuli; that path used tight `-0.2/+1.0`. This does **not** apply to ZUNA latent caching (which needs the 5s window; ZUNA handles the temporal axis internally).
- **Off-by-one sample rounding**: `--full5s-backaligned` yields 1281 samples; `cache_zuna_latents.py` trims to 1280. Accept both lengths and slice rather than raising.
- **Pipeline execution order**: VLM attribute generation must complete before CLIP/text embedding generation. Use cached `common_embeddings.pt` to avoid regenerating large embeddings on every run.
- **VAL_RUN / metadata alignment**: Dynamically read the actual number of downloaded runs (e.g. 32) rather than hardcoding a higher expected value (e.g. 40). Mismatched run counts produce an empty validation split crash: `ValueError: Invalid run split: train=3974 val=0`.

### RunPod & Infrastructure

- **SSH key mismatch on pod creation**: Pods not provisioned with `~/.ssh/runpod.pub` as `PUBLIC_KEY` will prompt for a password and block SSH. Always explicitly pass the ed25519 pub key. (The GitHub key `~/.ssh/id_ed25519` is separate — forward it with `ssh -A` for git operations.)
- **rsync overwrites remote checkpoints**: See the [rsync note above](#ssh--file-transfer). Pull before push.
- **Keep torch at 2.8.0+cu128 — re-pin after dep install**: The pod image ships torch 2.8.0+cu128 (has FlexAttention). Installing `requirements.txt` deps pulls **torch 2.12.1** via the zuna/lm-eval chain, breaking torchvision/torchaudio + CUDA libs. After any `pip install`, re-pin `torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128`. (The old "upgrade to torch 2.6.0+cu124" advice is obsolete — that was for the previous base image.)
- **`source venv/bin/activate` on pod**: The pod uses system Python with no venv. Replace `source venv/bin/activate` with `export PYTHONPATH=src` in any scripts synced to the pod.
- **Venv corruption on pod**: If a venv was created on the pod and then the container stopped/restarted, the venv may have zero-byte binaries (`python3` is 0 bytes, 000 permissions). Delete and recreate, or just use system Python with `PYTHONPATH=src`.
- **Multiple debug commands**: Gather all needed info in one SSH call rather than running separate commands for each check. SSH connection overhead is significant. (The `pod-exec` MCP avoids per-command approval prompts entirely.)
- **Disk-availability on create-pod**: A100 hosts reject large disk requests (`volumeInGb`/`containerDiskInGb`) with *"There are no longer any instances available with enough disk space."* 500–700 GB failed; **200 GB volume + 60 GB container** succeeded. NOD data is tiny (~0.6 GB/subject) — don't over-provision.
- **create-pod can't attach existing volumes / one GPU type**: MCP `create-pod` gives a fresh empty volume (no `networkVolumeId` field) and accepts only one GPU type per pod (extra `gpuTypeIds` ignored). Reuse a persistent volume via the RunPod web UI. See [`INFRA.md`](INFRA.md).
- **`pod_exec` JSON arg escaping**: Complex shell with special chars (`\[`, nested quotes) in the `pod_exec` command arg can fail JSON parsing. Keep commands simple or write them to a script on the pod and exec that.

- **Within-val vs full-bank retrieval gap**: `within_val_top10` compares EEG predictions against only the val batch (n≈596). This inflates numbers 2–4× vs. the honest `full_bank_top10` metric (predictions vs all 4000 image embeddings). Always use `full_bank_top10` and `full_bank_mrr` as primary gate metrics. A model can achieve within-val Top-10 = 0.042 while being BELOW random on full-bank (as A_real_repro demonstrated).
- **Target extraction compatibility**: Use `StableUnCLIPImg2ImgPipeline` instead of `StableUnCLIPPipeline` for image-to-image extraction.
- **Unnormalized targets**: When building `z_decode_common`, use `extract_teacher_embeds(normalize=False)`. Unnormalized embeds retain essential structural information for image reconstruction; L2-normalized embeddings fall back to random-level chance.
- **Relative evaluation gating**: Absolute cosine scores vary. Always evaluate relative performance against random embedding baselines: `Oracle Cosine > Random Cosine + 0.05`.

### Phase 17 (DINOv2-RAE)
- **Attribute Probe Mismatch**: Never feed RAE-space target vectors into `outputs/decode_probe_v2/common_probe.pt` (which was trained on Stable unCLIP decode_unit space). Use the RAE-native probe `outputs/rae_probe/common_probe.pt`.
- **Target Centering in Evaluation**: Generated images must be encoded using the RAE backend, subtracted by the training set mean (`rae_center_mean`), and then L2-normalized. Raw unit comparison will hide retrieval performance.
- **Option B Coordinate Alignment**: Make sure `pred_for_loss` is defined as `normalize(pred - target_center)` during training, and the auxiliary probe model gets `pred_for_loss` as input (instead of uncentered prediction).

### ⛔ Deprecated: RAE code-bottleneck (Phase 18A–18E)

The `train_rae_token_bottleneck.py` / `build_rae_bottleneck_codes.py` / `run_phase18*.sh` path
is **abandoned** — squeezing EEG through a `768×4×4` code discarded per-site fidelity and the
expanded-token gap collapsed to ~0. Superseded by the **QFormer bridge**. The scripts remain on
disk for reference only. See [`PLAN.md`](PLAN.md) §6 for the full post-mortem.


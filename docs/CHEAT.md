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

### Preferred: git pull on pod (no file transfer needed)
```bash
ssh-add ~/.ssh/id_ed25519
ssh -A -p <PORT> root@<IP> -i ~/.ssh/runpod "cd /workspace/mindeye && git pull origin master"
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
# The pod image already ships torch 2.8.0+cu128 — do NOT reinstall torch.
# Install only the non-torch deps (system Python, --break-system-packages for Ubuntu 24.04 PEP 668):
ssh -p <PORT> root@<IP> -i ~/.ssh/runpod \
  "cd /workspace/mindeye && \
   grep -vE '^(--extra-index-url|torch==|torchvision==|torchaudio==|nvidia-cudnn|\$)' requirements.txt > /tmp/reqs_notorch.txt && \
   pip install --break-system-packages -r /tmp/reqs_notorch.txt"
```

---

## Running the Pipeline

Current architecture: **ZUNA → QFormer → RAE** (see [`PLAN.md`](PLAN.md), [`HANDOVER.md`](HANDOVER.md)).
For step-by-step details see [`scripts/README.md`](../scripts/README.md). Always run on the pod with
`export PYTHONPATH=src` set (see [`INFRA.md`](INFRA.md) for the full env block).

### Cold start (end-to-end)
```bash
bash cold_start.sh   # env → download → ZUNA → crop → targets → cache latents → QFormer grid
```

### Cache ZUNA latents (QFormer input)
```bash
python scripts/cache_zuna_latents.py \
  --epochs-dir data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_08 \
  --output-dir data/processed/zuna_latents/sub01_runs01_08 \
  --layers post_mmd
```

### Run the QFormer bridge grid (real / shuffled / random + paired bootstrap)
```bash
python scripts/run_qformer_grid.py \
  --latents-pt data/processed/zuna_latents/sub01_runs01_32 \
  --rae-pt     data/processed/rae_embeddings/rae_dinov2_base_sub01_04_runs01_40.pt \
  --train-runs 1-24 --val-runs 25-28 --test-runs 29-32 \
  --epochs 40 --patience 8 --batch-size 64 --lr 3e-4 \
  --device cuda --out-dir outputs/qformer_aligned_grid
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
- **Stimulus overlap in RSVP / wide windows**: NOD presents images every ~1.3–1.7s. Wide windows (e.g. `[-3s, +2s]` or `[-1s, +4s]`) capture responses to multiple stimuli simultaneously. Use tight windows (`-0.2s` to `+1.0s`, 1.2s total) to isolate the target response.
- **Off-by-one sample rounding**: Window clipping at 256Hz can produce 1281 instead of 1280 samples due to floating-point rounding. Accept both lengths in the dataset validator and slice/truncate to the expected size rather than raising a hard error.
- **Pipeline execution order**: VLM attribute generation must complete before CLIP/text embedding generation. Use cached `common_embeddings.pt` to avoid regenerating large embeddings on every run.
- **VAL_RUN / metadata alignment**: Dynamically read the actual number of downloaded runs (e.g. 32) rather than hardcoding a higher expected value (e.g. 40). Mismatched run counts produce an empty validation split crash: `ValueError: Invalid run split: train=3974 val=0`.

### RunPod & Infrastructure

- **SSH key mismatch on pod creation**: Pods not provisioned with `~/.ssh/runpod.pub` as `PUBLIC_KEY` will prompt for a password and block SSH. Always explicitly pass the ed25519 pub key. (The GitHub key `~/.ssh/id_ed25519` is separate — forward it with `ssh -A` for git operations.)
- **rsync overwrites remote checkpoints**: See the [rsync note above](#ssh--file-transfer). Pull before push.
- **FlexAttention OOM / missing features**: PyTorch < 2.6.0 on the base RunPod image causes `flex_attention` CUDA OOMs. Always upgrade to `torch==2.6.0+cu124` after pod creation.
- **`source venv/bin/activate` on pod**: The pod uses system Python with no venv. Replace `source venv/bin/activate` with `export PYTHONPATH=src` in any scripts synced to the pod.
- **Venv corruption on pod**: If a venv was created on the pod and then the container stopped/restarted, the venv may have zero-byte binaries (`python3` is 0 bytes, 000 permissions). Delete and recreate, or just use system Python with `PYTHONPATH=src`.
- **Multiple debug commands**: Gather all needed info in one SSH call rather than running separate commands for each check. SSH connection overhead is significant.

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


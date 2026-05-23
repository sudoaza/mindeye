# MindEye Dev Cheat Sheet

> IMPORTANT: All GPU work runs on the **remote RunPod pod**. The dev machine has no GPU.

## Index
- [RunPod Management](#runpod-management)
- [SSH & File Transfer](#ssh--file-transfer)
- [Running the Pipeline](#running-the-pipeline)
- [Background Execution & Logging](#background-execution--logging)
- [Common Pitfalls & Troubleshooting](#common-pitfalls--troubleshooting)

---

## RunPod Management

Use the `runpod` MCP tools. See [`docs/RunPod_SKILL.md`](RunPod_SKILL.md) for full details.

```bash
mcp_runpod_list-pods
mcp_runpod_get-pod    {"podId": "<POD_ID>", "includeMachine": true}
mcp_runpod_start-pod  {"podId": "<POD_ID>"}
mcp_runpod_stop-pod   {"podId": "<POD_ID>"}
```

### Create a New Pod
```json
{
  "cloudType": "SECURE",
  "gpuCount": 1,
  "volumeInGb": 80,
  "containerDiskInGb": 50,
  "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
  "name": "mindeye-zuna",
  "ports": ["22/tcp"],
  "volumeMountPath": "/workspace",
  "env": {"PUBLIC_KEY": "ssh-ed25519 AAA... user@example.com"}
}
```

---

## SSH & File Transfer

Always use `-i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no`.

### Preferred: git pull on pod (no file transfer needed)
```bash
ssh -p <PORT> root@<IP> "cd /workspace/mindeye && git pull origin master"
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
# Install requirements (uses system Python — no venv needed on pod):
ssh -p <PORT> root@<IP> "cd /workspace/mindeye && pip install -r requirements.txt"

# IMPORTANT: Upgrade PyTorch to 2.6.0 for flex_attention support on CUDA 12.4:
ssh -p <PORT> root@<IP> "pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
```

---

## Running the Pipeline

For step-by-step details see [`scripts/README.md`](../scripts/README.md). The canonical entry points are via `make`:

```bash
make pipeline    # Full end-to-end: download → ZUNA → crop → embeddings → matrix
make matrix      # Matrix only (assumes prior steps done) — canonical training run
make ablation    # no-probe vs probe comparison (faster, ~2-3h)
make probe_sweep # Sweep probe weights 0 / 0.01 / 0.03 / 0.05 / 0.10
make simulate    # EPOC-14 low-channel simulation
```

On the pod, prefix with `export PYTHONPATH=src &&` when running scripts directly:
```bash
ssh -p <PORT> root@<IP> "cd /workspace/mindeye && export PYTHONPATH=src && nohup make matrix > matrix.log 2>&1 &"
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
| Probe model | `outputs/common_probe/common_probe.pt` |

**Key `metrics.json` fields**: `top1`, `top5`, `top10`, `mrr`, `median_rank`, `mean_diag_cosine`, `collapse_score`, `best_epoch`.

> `collapse_score` must be **> 0.1** to pass baseline gates (guards against dimensional collapse). Healthy runs typically score > 1.0.

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

- **SSH key mismatch on pod creation**: Pods not provisioned with `~/.ssh/id_ed25519.pub` as `PUBLIC_KEY` will prompt for a password and block SSH. Always explicitly pass the ed25519 pub key.
- **rsync overwrites remote checkpoints**: See the [rsync note above](#ssh--file-transfer). Pull before push.
- **FlexAttention OOM / missing features**: PyTorch < 2.6.0 on the base RunPod image causes `flex_attention` CUDA OOMs. Always upgrade to `torch==2.6.0+cu124` after pod creation.
- **`source venv/bin/activate` on pod**: The pod uses system Python with no venv. Replace `source venv/bin/activate` with `export PYTHONPATH=src` in any scripts synced to the pod.

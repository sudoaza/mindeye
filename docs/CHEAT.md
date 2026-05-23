# RunPod Cheat Sheet

## Index
- [List Pods](#list-pods)
- [Get Pod Info (IP & Ports)](#get-pod-info)
- [Start a Pod](#start-a-pod)
- [Create a New Pod](#create-a-new-pod)
- [Sync Code to Pod](#sync-code-to-pod)
- [Sync Safety — Pull Before Push](#sync-safety--pull-before-push)
- [Install Dependencies on Pod](#install-dependencies-on-pod)
- [Run Pipeline on Pod](#run-pipeline-on-pod)
- [Background Execution & Logging](#background-execution--logging)
- [Common Pitfalls & Troubleshooting](#common-pitfalls--troubleshooting)

---

IMPORTANT: Do all work in the remote pod, dev system has no GPU.

## RunPod Management

### List Pods
List all available pods to check their IDs and statuses.
```bash
# Using the MCP tool
mcp_runpod_list-pods
```

### Get Pod Info
Retrieve specific details about a pod, including its Public IP and mapped ports.
```bash
# Using the MCP tool
mcp_runpod_get-pod {"podId": "<POD_ID>", "includeMachine": true}
```

### Start a Pod
Start an existing pod that is in the EXITED state.
```bash
# Using the MCP tool
mcp_runpod_start-pod {"podId": "<POD_ID>"}
```

### Create a New Pod
Create a new GPU pod using a specific Docker image and environment variables (like your SSH key).
```json
// MCP mcp_runpod_create-pod payload
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

Use -o StrictHostKeyChecking=no for SSHing into pods and key named id_ed25519 under ~/.ssh/.

### Sync Code to Pod

Prefer git clone over rsync. Only rsync as last resort when testing and debugging code changes. When using rsync, make sure to ignore heavy virtual environments, __pycache__ and any data folders. Ensure you use the exact mapped port and Public IP provided by the pod info.

```bash
# First, ensure rsync is installed on the pod:
ssh -p <MAPPED_PORT> root@<PUBLIC_IP> "apt-get update && apt-get install -y rsync"

# Then sync the directory (excluding venv, .git, and cache):
rsync -avz --no-o --no-g --no-perms --exclude 'venv' --exclude '.git' --exclude '__pycache__' -e "ssh -p <MAPPED_PORT>" . root@<PUBLIC_IP>:/workspace/mindeye/
```

---

## Sync Safety — Pull Before Push

> [!CAUTION]
> **Never rsync local → remote if the remote has checkpoint files that were generated there** (e.g. `outputs/common_probe/common_probe.pt`, `outputs/baseline_matrix/`). A naive push will silently overwrite the new remote checkpoints with your stale local copies.

### Safe workflow

```bash
# 1. ALWAYS pull remote outputs first
rsync -avz --no-o --no-g --no-perms \
  -e "ssh -p <PORT>" \
  root@<IP>:/workspace/mindeye/outputs/common_probe/ \
  outputs/common_probe/

# 2. Then push code (excluding outputs/ which is gitignored anyway)
rsync -avz --no-o --no-g --no-perms \
  --exclude 'venv' --exclude '.git' --exclude '__pycache__' \
  --exclude 'outputs/' --exclude 'data/' \
  -e "ssh -p <PORT>" \
  . root@<IP>:/workspace/mindeye/
```

**Preferred alternative**: use `git pull` on the pod instead of rsync for code changes.

---

### Install Dependencies on Pod
Install requirements on the pod's system Python (avoids redundant venv downloads if the base image already has PyTorch).
```bash
ssh -p <MAPPED_PORT> root@<PUBLIC_IP> "cd /workspace/mindeye && pip install -r requirements.txt"
```
**Important:** If using the RunPod PyTorch 2.4 image with ZUNA, ensure you install `torch==2.6.0` built for CUDA 12.4 to avoid `flex_attention` OOM crashes and missing checkpointing features:
```bash
ssh -p <MAPPED_PORT> root@<PUBLIC_IP> "pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
```

### Run Pipeline on Pod
Execute Python scripts directly on the pod, setting PYTHONPATH so local modules are recognized. Below is the step-by-step pipeline execution sequence with arguments.

#### 1. ZUNA Denoising Inference
Runs the frozen foundation model on the continuous EEG data.
```bash
ssh -p <MAPPED_PORT> root@<PUBLIC_IP> "cd /workspace/mindeye && export PYTHONPATH=src && python src/mindseye/zuna/offline_pipeline.py configs/zuna/zuna_real_50steps.yaml"
```

#### 2. Event-Aligned Cropping
Extracts 5-second semantic crops locked to the image stimulus events.
```bash
ssh -p <MAPPED_PORT> root@<PUBLIC_IP> "cd /workspace/mindeye && export PYTHONPATH=src && python scripts/run_cropper.py --runs 1 2 3 4 5 --zuna-dir data/processed/zuna_output/4_fif_output"
```

#### 3. Integrity Audit
Verifies that the ZUNA duration and crop timestamps mathematically align with the raw recording.
```bash
ssh -p <MAPPED_PORT> root@<PUBLIC_IP> "cd /workspace/mindeye && export PYTHONPATH=src && python scripts/audit_zuna_timing.py"
```

#### 4. Generate Semantic CLIP Embeddings
Downloads/parses ImageNet stimuli and passes them through frozen CLIP to generate the embedding table for contrastive training.
```bash
ssh -p <MAPPED_PORT> root@<PUBLIC_IP> "cd /workspace/mindeye && export PYTHONPATH=src && python scripts/generate_clip_embeddings.py"
```

#### 5. Contrastive Training
Trains the EEG spatial-temporal encoder against the CLIP target vectors.
```bash
ssh -p <MAPPED_PORT> root@<PUBLIC_IP> "cd /workspace/mindeye && export PYTHONPATH=src && python scripts/train_eeg_clip.py --loss contrastive --center-clip --split-mode run --val-runs 5 --epochs 30"
```

---

## Background Execution & Logging

To ensure pipeline steps survive terminal disconnection, run them in the background using `nohup`.

### Background execution (nohup)

When running long-running jobs (such as baseline matrix evaluations or EPOC-14 low-channel simulations), background execution is mandatory.

```bash
# Start EPOC-14 simulation in background
nohup bash run_epoc_simulation.sh > epoc14.log 2>&1 &

# Start Baseline Matrix training in background (canonical command — matches Makefile)
nohup make matrix > matrix_v2.log 2>&1 &

# Or equivalently, run the script directly:
nohup python -u scripts/run_baseline_matrix.py \
  --subjects sub-01,sub-02,sub-03,sub-04 \
  --run-range 01_40 \
  --common-embeddings data/processed/clip_embeddings/common_embeddings.pt \
  --val-runs 8 \
  --window-mode tight1s \
  --model temporal_attn_small \
  --epochs 50 \
  --batch-size 64 \
  --device cuda \
  --slug common_space_sprint2 \
  --add-event-marker \
  --augment-eeg \
  --common-probe outputs/common_probe/common_probe.pt \
  --conditions zuna_real zuna_shuffled zuna_random > matrix_v2.log 2>&1 &
```

### Process Management

```bash
# Check running Python processes
ps aux | grep python

# Monitor background logs in real-time
tail -f /workspace/mindeye/epoc14.log
tail -f /workspace/mindeye/matrix_v2.log

# Terminate running background tasks safely
pkill -f 'python.*simulate_low_channel_zuna.py'
pkill -f 'python scripts/run_baseline_matrix.py'
```

### Log File Formats & Locations

Our pipeline outputs logs in two main formats: console redirection files (`.log`) and structured output artifacts (CSV, JSON, PT).

#### 1. Console Redirect Logs (`.log`)
- **Location:** Saved directly in `/workspace/mindeye/` (e.g., `epoc14.log`, `matrix_v2.log`, `matrix_epoc14.log`).
- **Format:** Plain text logs capturing standard output (stdout) and standard error (stderr). Useful for real-time progress tailing, CLI validation outputs, and validation gate checks.

#### 2. Structured Outputs & Metrics
For each training session or baseline matrix evaluation, structured outputs are saved in timestamped run directories under `outputs/runs/` or `outputs/baseline_matrix/`.

- **`matrix_run.log`**: 
  - **Location:** `outputs/baseline_matrix/<timestamp>_matrix/matrix_run.log`
  - **Format:** Combined plain text console log recording detailed stderr/stdout from all evaluated conditions.
- **`metrics.json`**:
  - **Location:** `outputs/runs/<timestamp>_<slug>/metrics.json`
  - **Format:** JSON object containing final epoch metrics. Key fields:
    ```json
    {
      "loss": 1.4582,
      "n": 640,
      "top1": 0.082,
      "top5": 0.285,
      "top10": 0.443,
      "mrr": 0.187,
      "median_rank": 14.0,
      "mean_diag_cosine": 0.112,
      "collapse_score": 0.418,
      "random_top10_expected": 0.0156,
      "best_epoch": 32,
      "best_score": 0.297
    }
    ```
    *Note: `collapse_score` must remain > 0.1 to pass baseline gates (guards against dimensional collapse).*
- **`train_log.csv`**:
  - **Location:** `outputs/runs/<timestamp>_<slug>/train_log.csv`
  - **Format:** CSV file tracking training progress epoch-by-epoch.
  - **Headers:** `epoch`, `train_loss`, `val_loss`, `val_score`, `top1`, `top5`, `top10`, `mrr`, `median_rank`, `mean_diag_cosine`, `collapse_score`
- **`matrix_summary.csv`**:
  - **Location:** `outputs/baseline_matrix/<timestamp>_matrix/matrix_summary.csv`
  - **Format:** CSV aggregating key validation metrics (`top1`, `top10`, `mrr`, `collapse_score`, `status`, etc.) across all compared matrix conditions (e.g. `zuna_real`, `zuna_shuffled`, `zuna_random`).

---

## Common Pitfalls & Troubleshooting

### 1. EEG Model Architecture & Training
- **BatchNorm1d vs. GroupNorm in Stems**: Using `BatchNorm1d` in spatial/temporal stems causes severe train/eval discrepancy on small-batch/noisy EEG data, resulting in validation performance flatlining. Replacing `BatchNorm1d` with `GroupNorm` (which computes statistics independently per-sample) stabilizes validation and prevents representation collapse.
- **EEG Augmentation on Marker Channels**: When applying augmentations (like adding noise, scaling, or masking) to the EEG signal, the **event marker channel** (typically the last channel, e.g. index 63) must be excluded from the augmentation. Corrupting the event marker destroys event synchronization and temporal alignment.
- **Validation Target Leakage**: If control evaluations (shuffled, random) are run, any auxiliary multi-task labels must be shuffled/randomized *identically* to the main target to prevent label leakage, which artificially inflates control scores.
- **Dimensional Collapse Guard**: During checkpoint saving, relying only on evaluation metrics (like MRR or Top-10) can lead to saving degenerate collapsed models. Checkpoint selection should enforce a `collapse_score` threshold (e.g., must be $> 0.1$, ideally $> 1.0$) to reject collapsed models.
- **Multitask `return_features` Signature**: Models evaluated with downstream multitask attributes (e.g. classification heads) must support `return_features: bool = False` in their `forward()` method, returning `(output, features)` when `True`, otherwise training will crash with a `TypeError`.

### 2. Pipeline, Data & Script Execution
- **VLM Output Field Name Mismatches**: When generating semantic texts using VLMs (e.g., Qwen2-VL), ensuring prompt schema fields align perfectly with JSON parser keys is crucial. For example, generating `structured_embedding_text` but saving/reading `embedding_text` caused the text embedding generator to fall back to the string `"empty"` for all images, collapsing the target embedding space to a single point.
- **Physiological Noise / Stimulus Overlap in RSVP**: In rapid serial visual presentation (RSVP) paradigms (like NOD where images appear every ~1.3-1.7s), wide time windows (e.g., 5-second windows `[-3s, +2s]` or `[-1s, +4s]`) capture signals from multiple overlapping future/past stimuli. Slicing tight windows (e.g., `-0.2s` to `1.0s` / 1.2s total) isolates the specific stimulus response and eliminates this overlap noise.
- **Rounding/Off-by-One Sample Alignment**: Window clipping logic (like `full5s_backaligned` at 256Hz) can produce 1281 samples instead of exactly 1280 due to floating-point rounding. Datasets must handle this variation by allowing `1280` or `1281` and slicing/truncating to exactly `1280` rather than raising a strict shape check error.
- **Order of Pipeline Operations**: Generating text/CLIP embeddings must occur *after* VLM attribute synthesis. Ensure scripts are executed in the correct order, and use cached files (`common_embeddings.pt`) instead of regenerating large embeddings on every run to save time.
- **Run / Metadata Alignment**: When setting `VAL_RUN` or validating splits, ensure you check the actual available runs in the downloaded dataset (e.g., 32 runs) rather than hardcoding a higher expected run number (like 40), which causes empty validation split crashes.

### 3. RunPod & Infrastructure Management
- **SSH Key Mismatch**: Setting a generic key name or the wrong key type when creating a pod via MCP causes authentication failures. Ensure you explicitly pass the user's `~/.ssh/id_ed25519.pub` public key in the environment dictionary (`PUBLIC_KEY` environment variable).
- **Strict Host Key Checking**: Always use `-o StrictHostKeyChecking=no` to prevent SSH connection hangs when interacting with newly spun-up pods.
- **Rsync Overwrites Remote Checkpoints**: Running `rsync local → remote` after training has completed remotely will overwrite the newly generated `outputs/common_probe/common_probe.pt` (and other checkpoints) with stale local copies. **Always pull remote outputs first** (`rsync remote → local`) before pushing code. Better yet, use `git pull` on the pod for code changes and only rsync outputs in the remote→local direction. See [Sync Safety](#sync-safety--pull-before-push).
- **Rsync Exclusions**: When using `rsync` to copy local code to remote pods, explicitly exclude `venv`, `.git`, `__pycache__`, `outputs/`, and heavy data directories (`data/raw`, `data/processed`) to avoid massive transfer overhead or overwriting remote checkpoints.
- **Flexible Attention / FlexAttention OOMs**: On PyTorch versions under `2.6.0`, running `flex_attention` can cause CUDA OOMs or is missing required checkpointing features. Updating to `torch==2.6.0` built for CUDA 12.4 is required to run ZUNA without OOMs.


# RunPod Cheat Sheet

## Index
- [List Pods](#list-pods)
- [Get Pod Info (IP & Ports)](#get-pod-info)
- [Start a Pod](#start-a-pod)
- [Create a New Pod](#create-a-new-pod)
- [Sync Code to Pod](#sync-code-to-pod)
- [Install Dependencies on Pod](#install-dependencies-on-pod)
- [Run Pipeline on Pod](#run-pipeline-on-pod)
- [Background Execution & Logging](#background-execution--logging)

---

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

# Start Baseline Matrix training in background
nohup python scripts/run_baseline_matrix.py \
  --metadata data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv \
  --epochs-dir data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40 \
  --common-embeddings data/processed/clip_embeddings/combined_common_embeddings.pt \
  --val-runs 32 \
  --window-mode tight1s \
  --target-space common \
  --model temporal_attn_small \
  --epochs 50 \
  --batch-size 64 \
  --device cuda \
  --slug tight1s_recovery \
  --add-event-marker \
  --augment-eeg \
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


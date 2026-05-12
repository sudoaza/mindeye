# RunPod Cheat Sheet

## Index
- [List Pods](#list-pods)
- [Get Pod Info (IP & Ports)](#get-pod-info)
- [Start a Pod](#start-a-pod)
- [Create a New Pod](#create-a-new-pod)
- [Sync Code to Pod](#sync-code-to-pod)
- [Install Dependencies on Pod](#install-dependencies-on-pod)
- [Run Pipeline on Pod](#run-pipeline-on-pod)

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

Use -o StrictHostKeyChecking=no for SSHing into pods.

### Sync Code to Pod
Quickly sync your local workspace to the pod, ignoring heavy virtual environments. Ensure you use the exact mapped port and Public IP provided by the pod info.
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

# MindEye — RunPod Cheat Sheet

Quick reference for spinning up a pod that can actually run everything without memory gymnastics.

---

## Recommended Pod Configuration

| Parameter | Value | Why |
|-----------|-------|-----|
| **GPU** | RTX 3090, RTX 4090, A100, or H100 | Stable unCLIP fits comfortably in ~8-12 GB VRAM; H100/A100 speeds up iteration |
| **VRAM** | ≥ 16 GB | Stable unCLIP model stack fits in ~6 GB in float16; 16 GB gives room for batching |
| **CPU RAM** | ≥ 32 GB | For dataloaders and dataset preloading |
| **Container disk** | **100 GB** | HF model cache is ~15 GB; venv ~4 GB; data/outputs need headroom |
| **Network volume** | 50-100 GB (attached) | For persistent `/workspace` across pod restarts; keep model cache here |
| **Image** | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` | Matches pinned torch/CUDA versions in requirements.txt |

> [!IMPORTANT]
> Always create the pod in the **same datacenter** as the network volume.
> RunPod network volumes are datacenter-local — a pod in `EUR-NO-2` cannot mount a volume from `EU-RO-1`. Check the volume's `dataCenterId` first.

---

## Quick Setup (fresh pod)

```bash
# 1. Directories
mkdir -p /workspace/{mindeye,hf_cache,tmp}

# 2. Venv
python3 -m venv /workspace/mindeye/venv
/workspace/mindeye/venv/bin/pip install --upgrade pip

# 3. Sync code from local (run on your laptop)
rsync -avz --no-o --no-g --no-perms --exclude 'venv' --exclude '.git' --exclude '__pycache__' --exclude 'outputs/' --exclude 'data/' \
  -e "ssh -p <PORT> -i ~/.ssh/id_ed25519" \
  . root@<IP>:/workspace/mindeye/

# 4. Install deps
/workspace/mindeye/venv/bin/pip install -r /workspace/mindeye/requirements.txt

# 5. Download Stable unCLIP models (auto-cached to HF_HOME)
export HF_HOME=/workspace/hf_cache
/workspace/mindeye/venv/bin/python3 -c "
from diffusers import StableUnCLIPImg2ImgPipeline
StableUnCLIPImg2ImgPipeline.from_pretrained('sd2-community/stable-diffusion-2-1-unclip-small')
"

# 6. Verify pipeline (Gate 1 Smoke Test)
cd /workspace/mindeye
export PYTHONPATH=src HF_HOME=/workspace/hf_cache TMPDIR=/workspace/tmp
/workspace/mindeye/venv/bin/python3 scripts/smoke_test_gate1.py
```

---

## Environment Variables (always set before running)

```bash
export PYTHONPATH=/workspace/mindeye/src
export HF_HOME=/workspace/hf_cache
export TMPDIR=/workspace/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## SSH shortcut (add to ~/.ssh/config on laptop)

```
Host mindeye-pod
  HostName <PUBLIC_IP>
  Port <SSH_PORT>
  User root
  IdentityFile ~/.ssh/id_ed25519
  StrictHostKeyChecking no
```

Then just: `ssh mindeye-pod`

---

## Model Memory Footprint (float16)

| Component | Params | VRAM |
|-----------|--------|------|
| CLIP Image Encoder (ViT-L/14) | ~300 M | ~600 MB |
| Stable Diffusion UNet | ~860 M | ~1.7 GB |
| Prior / Text Encoder | ~1 B | ~2.0 GB |
| VAE Decoder | — | ~300 MB |
| **Total at rest** | | **~4.6 GB** |
| Peak during generation (512×512, 20 steps) | | ~6.0–8.0 GB |

---

## Disk Usage Breakdown

| Path | Size | Notes |
|------|------|-------|
| `/workspace/hf_cache` | ~15 GB | Stable unCLIP weights |
| `/workspace/mindeye/venv` | ~4 GB | Python packages |
| `/workspace/mindeye/src` + `scripts` | ~5 MB | Code (synced from laptop) |
| `/workspace/tmp` | variable | Torch tmp; point `TMPDIR` here |
| **Total** | **~20 GB** | → 100 GB container leaves plenty of room for outputs/data |

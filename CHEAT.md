# MindEye — RunPod Cheat Sheet

Quick reference for spinning up a pod that can actually run everything without
memory gymnastics.

---

## Recommended Pod Configuration

| Parameter | Value | Why |
|-----------|-------|-----|
| **GPU** | H100 80GB HBM3 (or A100 80GB) | Full model stack fits in ~30 GB; 80 GB gives room for batching later |
| **VRAM** | ≥ 80 GB | Qwen-Image transformer (~15 GB) + text encoder (~14 GB) + VAE (~1 GB) = ~30 GB in float16; no quantisation needed |
| **CPU RAM** | ≥ 100 GB | Text encoder can fall back to CPU during heavy GPU loads |
| **Container disk** | **200 GB** | HF model cache alone is ~35–40 GB; venv ~4 GB; data/outputs need headroom |
| **Network volume** | 100 GB (attached) | For persistent `/workspace` across pod restarts; keep model cache here |
| **Image** | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` | Matches pinned torch/CUDA versions in requirements.txt |

> [!IMPORTANT]
> Always create the pod in the **same datacenter** as the network volume.
> RunPod network volumes are datacenter-local — a pod in `EUR-NO-2` cannot
> mount a volume from `EU-RO-1`. Check the volume's `dataCenterId` first.

> [!TIP]
> If RunPod ignores `dataCenterId` and lands on the wrong one, stop the pod,
> either (a) create a new network volume in that DC, or (b) use a large
> container disk (200 GB) and re-download models (~80 s at 3 Gbps).

---

## Quick Setup (fresh pod)

```bash
# 1. Directories
mkdir -p /workspace/{mindeye,hf_cache,tmp,offload}

# 2. Venv
python3 -m venv /workspace/mindeye/venv
/workspace/mindeye/venv/bin/pip install --upgrade pip

# 3. Sync code from local (run on your laptop)
rsync -avz --no-o --no-g -e "ssh -p <PORT> -i ~/.ssh/id_ed25519" \
  src/ scripts/ requirements.txt \
  root@<IP>:/workspace/mindeye/

# 4. Install deps
/workspace/mindeye/venv/bin/pip install -r /workspace/mindeye/requirements.txt

# 5. Download models (auto-cached to HF_HOME)
export HF_HOME=/workspace/hf_cache
/workspace/mindeye/venv/bin/python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen-Image')
snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct', ignore_patterns=['*.safetensors','*.bin'])
"

# 6. Verify pipeline
cd /workspace/mindeye
export PYTHONPATH=src HF_HOME=/workspace/hf_cache TMPDIR=/workspace/tmp
/workspace/mindeye/venv/bin/python3 scripts/verify_qwen.py
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

## Model Memory Footprint (float16, H100)

| Component | Params | VRAM |
|-----------|--------|------|
| Qwen-Image transformer | ~7 B | ~15 GB |
| Qwen2.5-VL-7B text encoder | 7 B | ~14 GB |
| Qwen-Image VAE | — | ~1 GB |
| **Total at rest** | | **~30 GB** |
| Peak during generation (512×512, 20 steps) | | ~40–50 GB |

With an 80 GB card there is **no need** for:
- bitsandbytes 8-bit quantisation
- `device_map` CPU offloading
- VAE GPU ↔ CPU juggling
- dummy latent hacks

---

## Disk Usage Breakdown

| Path | Size | Notes |
|------|------|-------|
| `/workspace/hf_cache` | ~40 GB | `Qwen/Qwen-Image` full weights |
| `/workspace/mindeye/venv` | ~4 GB | Python packages |
| `/workspace/mindeye/src` + `scripts` | ~5 MB | Code (synced from laptop) |
| `/workspace/tmp` | variable | Torch tmp; point `TMPDIR` here |
| `/workspace/offload` | 0–5 GB | Fallback layer offload (not needed on H100) |
| **Total** | **~45 GB** | → 200 GB container leaves plenty of room for outputs/data |

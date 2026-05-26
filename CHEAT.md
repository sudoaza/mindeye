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

---

## Multi-Subject Configuration & Gotchas

> [!WARNING]
> When running multi-subject experiments, you **must** specify matching comma-separated paths for both `--metadata` and `--epochs-dir`.
> If `--epochs-dir` contains only one path while `--metadata` has multiple, the dataset builder will silently duplicate the single epochs directory for all metadata files. If data for the other subjects does not exist in that single directory, the dataset loader will silently skip those subjects, causing training to fallback to a single subject without raising an error.
> 
> Always format multi-subject training commands with equal-length comma-separated lists:
> ```bash
> python scripts/train_eeg_clip.py \
>   --metadata "data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv,data/processed/semantic_epochs/zuna_tight1s_sub02_runs01_40/all_runs_metadata.csv" \
>   --epochs-dir "data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40,data/processed/semantic_epochs/zuna_tight1s_sub02_runs01_40" \
>   ...
> ```
> 
> ### Rebuild Target Embeddings Gotcha
> Each subject in the Natural Object Dataset sees a different set of ImageNet stimulus images. If you add new subjects but do not rebuild the common target embeddings file (`decode_common_embeddings.pt`), the dataset loader will silently filter out 100% of the new subjects' samples because it cannot find target embeddings for their image IDs.
> 
> You **must** rebuild the embeddings file whenever you introduce new subjects or stimulus images:
> ```bash
> python scripts/build_decode_common_embeddings.py \
>   --image-dir data/raw/nod/stimuli/ImageNet \
>   --output data/processed/clip_embeddings/decode_common_embeddings.pt
> ```


### Multi-Subject Data Prep Sequence
Before training, each subject's dataset must be downloaded, denoised via ZUNA, and cropped:
```bash
# 1. Download runs 1-32 for subjects 02, 03, 04
for sub in sub-02 sub-03 sub-04; do
  python scripts/download_nod.py --subject $sub --runs 1-32
done

# 2. Denoise (skips already processed files automatically)
python scripts/run_zuna_batch.py --diffusion-steps 15

# 3. Crop tight1s semantic epochs
for sub in sub-02 sub-03 sub-04; do
  python scripts/run_cropper.py --mode zuna --tmin -0.2 --tmax 1.0 --add-event-marker \
    --runs 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 \
    --subject $sub \
    --output-dir data/processed/semantic_epochs/zuna_tight1s_${sub}_runs01_40
done
```

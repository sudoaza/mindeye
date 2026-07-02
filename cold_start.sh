#!/usr/bin/env bash
# ==============================================================================
# MindEye EEG-to-Vision Pipeline: End-to-End Cold Start Script
# ==============================================================================
# This script automates the complete lifecycle:
# 1. Environment Setup (deps install; skip with SKIP_ENV=1 if already installed)
# 2. Raw NOD-EEG dataset downloading from OpenNeuro
# 3. ZUNA Denoising
# 4. Segmenting continuous recordings into semantic epochs
# 5. Syncing ImageNet stimulus images from S3 using include-lists
# 6. Building the RAE/DINOv2 visual reconstruction target bank
# 7. Caching ZUNA activations
# 8. Training and evaluating the QFormer bridge (ZUNA -> RAE) with real/shuffled/random controls
# ==============================================================================
set -e

# On a RunPod pod the image already ships torch and deps are installed with
# --break-system-packages (system Python, no venv). Set SKIP_ENV=1 to skip this step.
if [ "${SKIP_ENV:-0}" != "1" ]; then
  echo "=== [1/8] Setting up Python virtual environment ==="
  python3 -m venv venv
  source venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
else
  echo "=== [1/8] SKIP_ENV=1 — using existing environment ==="
fi

# Create necessary directories
mkdir -p data/raw/nod
mkdir -p data/processed/rae_embeddings
mkdir -p data/processed/zuna_latents
mkdir -p outputs/qformer_aligned_grid

echo "=== [2/8] Downloading raw NOD-EEG (ds005811) files from OpenNeuro ==="
# We default to subject sub-01, runs 1-8 for development. 
# For a full run, change runs to 1-40.
python scripts/download_nod.py --subject sub-01 --runs 1-8

echo "=== [3/8] Running ZUNA diffusion-based continuous denoising ==="
python scripts/run_zuna_batch.py --diffusion-steps 15

echo "=== [4/8] Cropping continuous EEG recordings into semantic epochs ==="
python scripts/run_cropper.py \
    --mode zuna \
    --tmin -0.2 \
    --tmax 1.0 \
    --add-event-marker \
    --runs 1 2 3 4 5 6 7 8 \
    --zuna-dir data/processed/zuna_real/4_fif_output \
    --output-dir data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_08

echo "=== [5/8] Building include-list and syncing stimulus images from S3 ==="
# Derive the list of required ImageNet stimulus files from the cropped epochs
python scripts/generate_clip_embeddings.py \
    --metadata data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_08/all_runs_metadata.csv \
    --write-openneuro-include-list data/raw/nod/stimuli_include.txt

# Download / Sync target stimulus images from OpenNeuro's S3 bucket
python scripts/sync_stimuli_s3_targeted.py

echo "=== [6/8] Building RAE/DINOv2 latent bank target embeddings ==="
python scripts/build_rae_latent_bank.py \
    --image-dir data/raw/nod/stimuli/ImageNet \
    --output data/processed/rae_embeddings/rae_dinov2_base_sub01_04_runs01_40.pt

echo "=== [7/8] Caching ZUNA activations ==="
python scripts/cache_zuna_latents.py \
    --epochs-dir data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_08 \
    --output-dir data/processed/zuna_latents/sub01_runs01_08 \
    --layers post_mmd

echo "=== [8/8] Running QFormer bridge grid & bootstrap evaluation (ZUNA -> RAE) ==="
python scripts/run_qformer_grid.py \
    --latents-pt data/processed/zuna_latents/sub01_runs01_08 \
    --rae-pt data/processed/rae_embeddings/rae_dinov2_base_sub01_04_runs01_40.pt \
    --epochs 40 \
    --patience 8 \
    --batch-size 64 \
    --lr 3e-4 \
    --device cuda \
    --out-dir outputs/qformer_aligned_grid

echo "=== Cold start pipeline completed successfully! ==="

#!/usr/bin/env bash
# ==============================================================================
# MindEye EEG-to-Vision Pipeline: End-to-End Cold Start Script
# ==============================================================================
# This script automates the complete lifecycle:
# 1. Environment Setup (venv and pip install)
# 2. Raw NOD-EEG dataset downloading from OpenNeuro
# 3. ZUNA Denoising
# 4. Segmenting continuous recordings into semantic epochs
# 5. Syncing ImageNet stimulus images from S3 using include-lists
# 6. Building visual-semantic target latent spaces (CLIP & DINO/RAE)
# 7. Caching ZUNA activations
# 8. Training and evaluating QFormer adapters comparing DINO vs CLIP
# ==============================================================================
set -e

echo "=== [1/10] Setting up Python virtual environment ==="
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/raw/nod
mkdir -p data/processed/clip_embeddings
mkdir -p data/processed/rae_embeddings
mkdir -p data/processed/zuna_latents
mkdir -p outputs/qformer_aligned_grid
mkdir -p outputs/common_probe

echo "=== [2/10] Downloading raw NOD-EEG (ds005811) files from OpenNeuro ==="
# We default to subject sub-01, runs 1-8 for development. 
# For a full run, change runs to 1-40.
python scripts/download_nod.py --subject sub-01 --runs 1-8

echo "=== [3/10] Running ZUNA diffusion-based continuous denoising ==="
python scripts/run_zuna_batch.py --diffusion-steps 15

echo "=== [4/10] Cropping continuous EEG recordings into semantic epochs ==="
python scripts/run_cropper.py \
    --mode zuna \
    --tmin -0.2 \
    --tmax 1.0 \
    --add-event-marker \
    --runs 1 2 3 4 5 6 7 8 \
    --zuna-dir data/processed/zuna_real/4_fif_output \
    --output-dir data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_08

echo "=== [5/10] Building include-list and syncing stimulus images from S3 ==="
# Generate the list of required ImageNet stimulus files based on cropped epochs
python scripts/generate_clip_embeddings.py \
    --metadata data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_08/all_runs_metadata.csv \
    --write-openneuro-include-list data/raw/nod/stimuli_include.txt

# Download / Sync target stimulus images from OpenNeuro's S3 bucket
python scripts/sync_stimuli_s3_targeted.py

echo "=== [6/10] Extracting Image & Text CLIP targets ==="
# Generate image CLIP embeddings
python scripts/generate_clip_embeddings.py \
    --metadata data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_08/all_runs_metadata.csv \
    --stimuli-root data/raw/nod/stimuli/ImageNet \
    --output data/processed/clip_embeddings/sub01_image_embeddings.pt

# Generate ImageNet template label text embeddings
python scripts/generate_text_embeddings.py \
    --metadata data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_08/all_runs_metadata.csv \
    --output data/processed/clip_embeddings/imagenet_text_embeddings.pt \
    --source templates

# Generate VLM structured semantic captions (Qwen-VL)
python scripts/generate_image_semantics.py \
    --metadata data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_08/all_runs_metadata.csv \
    --image-root data/raw/nod/stimuli/ImageNet \
    --output data/processed/clip_embeddings/image_semantic_text_embeddings.jsonl

# Generate text embeddings from VLM captions
python scripts/generate_text_embeddings.py \
    --metadata data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_08/all_runs_metadata.csv \
    --output data/processed/clip_embeddings/image_semantic_text_embeddings.pt \
    --source image_semantics \
    --semantics-jsonl data/processed/clip_embeddings/image_semantic_text_embeddings.jsonl

# Generate visual-semantic attributes for probing
python scripts/generate_vlm_attributes.py \
    --metadata data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_08/all_runs_metadata.csv \
    --image-dir data/raw/nod/stimuli/ImageNet \
    --output data/processed/clip_embeddings/vlm_attributes.json \
    --batch-size 8

echo "=== [7/10] Fusing CLIP spaces to build z_common ==="
python scripts/build_common_embeddings.py \
    --image-embeddings data/processed/clip_embeddings/sub01_image_embeddings.pt \
    --semantic-embeddings data/processed/clip_embeddings/image_semantic_text_embeddings.pt \
    --label-embeddings data/processed/clip_embeddings/imagenet_text_embeddings.pt \
    --metadata data/raw/nod/derivatives/detailed_events/sub-01_events.csv \
    --w-img 0.25 --w-sem 0.65 --w-lbl 0.10 \
    --output data/processed/clip_embeddings/common_embeddings.pt

echo "=== [8/10] Building RAE/DINOv2 latent bank target embeddings ==="
python scripts/build_rae_latent_bank.py \
    --image-dir data/raw/nod/stimuli/ImageNet \
    --output data/processed/rae_embeddings/rae_dinov2_base_sub01_04_runs01_40.pt

echo "=== [9/10] Caching ZUNA activations ==="
python scripts/cache_zuna_latents.py \
    --epochs-dir data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_08 \
    --output-dir data/processed/zuna_latents/sub01_runs01_08 \
    --layers post_mmd

echo "=== [10/10] Running minimal QFormer training grid & bootstrap evaluation (CLIP vs DINO) ==="
python scripts/run_qformer_grid.py \
    --latents-pt data/processed/zuna_latents/sub01_runs01_08 \
    --clip-pt data/processed/clip_embeddings/common_embeddings.pt \
    --rae-pt data/processed/rae_embeddings/rae_dinov2_base_sub01_04_runs01_40.pt \
    --epochs 40 \
    --patience 8 \
    --batch-size 64 \
    --lr 3e-4 \
    --device cuda \
    --out-dir outputs/qformer_aligned_grid

echo "=== Cold start pipeline completed successfully! ==="

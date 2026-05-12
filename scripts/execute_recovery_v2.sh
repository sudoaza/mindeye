#!/bin/bash
# Comprehensive Recovery Execution Script
set -e

source venv/bin/activate

echo "=== [1/7] Downloading NOD Runs 1-8 ==="
python scripts/download_nod.py --subject sub-01 --runs 1-8

echo "=== [2/7] Syncing Stimuli from S3 ==="
python scripts/sync_stimuli_s3.py

echo "=== [3/7] Running ZUNA Batch Pipeline (15 steps) ==="
# Note: This creates data/processed/zuna_real/4_fif_output
python scripts/run_zuna_batch.py --diffusion-steps 15

echo "=== [4/7] Cropping Full 5s Windows ==="
# This creates data/processed/semantic_epochs/zuna_full5s_sub01_runs01_08
python scripts/run_cropper.py --mode zuna --full5s --runs 1 2 3 4 5 6 7 8 \
    --zuna-dir data/processed/zuna_real/4_fif_output \
    --output-dir data/processed/semantic_epochs/zuna_full5s_sub01_runs01_08

echo "=== [5/7] Generating Embeddings ==="
python scripts/generate_text_embeddings.py \
    --metadata data/raw/nod/derivatives/detailed_events/sub-01_events.csv \
    --output data/processed/clip_embeddings/imagenet_text_embeddings.pt

python scripts/generate_clip_embeddings.py \
    --metadata data/raw/nod/derivatives/detailed_events/sub-01_events.csv \
    --output data/processed/clip_embeddings/sub01_image_embeddings.pt

echo "=== [6/7] Smoke Test (2 epochs) ==="
python scripts/train_eeg_clip.py \
  --metadata data/processed/semantic_epochs/zuna_full5s_sub01_runs01_08/all_runs_metadata.csv \
  --epochs-dir data/processed/semantic_epochs/zuna_full5s_sub01_runs01_08 \
  --clip-embeddings data/processed/clip_embeddings/sub01_image_embeddings.pt \
  --text-embeddings data/processed/clip_embeddings/imagenet_text_embeddings.pt \
  --input-domain zuna \
  --window-mode full5s \
  --semantic-target image_text \
  --target-mode real \
  --model temporal_attn \
  --val-runs 8 \
  --epochs 2 \
  --batch-size 16 \
  --device cuda \
  --slug smoke_full5s

echo "=== [7/7] Launching Full Recovery Matrix ==="
python scripts/run_baseline_matrix.py \
  --metadata data/processed/semantic_epochs/zuna_full5s_sub01_runs01_08/all_runs_metadata.csv \
  --epochs-dir data/processed/semantic_epochs/zuna_full5s_sub01_runs01_08 \
  --clip-embeddings data/processed/clip_embeddings/sub01_image_embeddings.pt \
  --text-embeddings data/processed/clip_embeddings/imagenet_text_embeddings.pt \
  --val-runs 8 \
  --window-mode full5s \
  --semantic-target image_text \
  --model temporal_attn \
  --epochs 50 \
  --batch-size 64 \
  --device cuda \
  --slug full5s_recovery \
  --conditions zuna_runheldout zuna_shuffled zuna_random

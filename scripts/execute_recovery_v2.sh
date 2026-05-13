#!/bin/bash
# Comprehensive Recovery Execution Script
set -e

source venv/bin/activate

echo "=== [1/7] Downloading NOD Runs 1-8 ==="
python scripts/download_nod.py --subject sub-01 --runs 1-8

echo "=== [2/7] Syncing Stimuli from S3 ==="
python scripts/generate_clip_embeddings.py \
    --metadata data/raw/nod/derivatives/detailed_events/sub-01_events.csv \
    --write-openneuro-include-list data/raw/nod/stimuli_include.txt

python scripts/sync_stimuli_s3_targeted.py

echo "=== [3/7] Running ZUNA Batch Pipeline (15 steps) ==="
python scripts/run_zuna_batch.py --diffusion-steps 15

echo "=== [4/7] Cropping Full 5s Back-aligned Windows ==="
python scripts/run_cropper.py \
  --mode zuna \
  --full5s-backaligned \
  --add-event-marker \
  --runs 1 2 3 4 5 6 7 8 \
  --zuna-dir data/processed/zuna_real/4_fif_output \
  --output-dir data/processed/semantic_epochs/zuna_full5s_backaligned_sub01_runs01_08

echo "=== [5/7] Generating Embeddings ==="
python scripts/generate_clip_embeddings.py \
    --metadata data/raw/nod/derivatives/detailed_events/sub-01_events.csv \
    --output data/processed/clip_embeddings/sub01_image_embeddings.pt

python scripts/generate_image_semantics.py \
  --metadata data/processed/semantic_epochs/zuna_full5s_backaligned_sub01_runs01_08/all_runs_metadata.csv \
  --image-root data/raw/nod/stimuli/ImageNet \
  --output data/processed/clip_embeddings/image_semantics.jsonl

python scripts/generate_text_embeddings.py \
  --source image_semantics \
  --semantics-jsonl data/processed/clip_embeddings/image_semantics.jsonl \
  --output data/processed/clip_embeddings/image_semantic_text_embeddings.pt

python scripts/generate_text_embeddings.py \
  --source labels \
  --metadata data/processed/semantic_epochs/zuna_full5s_backaligned_sub01_runs01_08/all_runs_metadata.csv \
  --output data/processed/clip_embeddings/imagenet_text_embeddings.pt

python scripts/build_common_embeddings.py \
  --image-embeddings data/processed/clip_embeddings/sub01_image_embeddings.pt \
  --semantic-embeddings data/processed/clip_embeddings/image_semantic_text_embeddings.pt \
  --label-embeddings data/processed/clip_embeddings/imagenet_text_embeddings.pt \
  --metadata data/processed/semantic_epochs/zuna_full5s_backaligned_sub01_runs01_08/all_runs_metadata.csv \
  --w-img 0.25 --w-sem 0.65 --w-lbl 0.10 \
  --output data/processed/clip_embeddings/common_embeddings.pt

echo "=== [6/7] Smoke Test (2 epochs) ==="
python scripts/train_eeg_clip.py \
  --metadata data/processed/semantic_epochs/zuna_full5s_backaligned_sub01_runs01_08/all_runs_metadata.csv \
  --epochs-dir data/processed/semantic_epochs/zuna_full5s_backaligned_sub01_runs01_08 \
  --common-embeddings data/processed/clip_embeddings/common_embeddings.pt \
  --input-domain zuna \
  --window-mode full5s_backaligned \
  --target-space common \
  --target-mode real \
  --model temporal_attn_small \
  --val-runs 8 \
  --epochs 2 \
  --batch-size 16 \
  --device cuda \
  --slug smoke_full5s \
  --add-event-marker \
  --augment-eeg

echo "=== [7/7] Launching Full Recovery Matrix ==="
python scripts/run_baseline_matrix.py \
  --metadata data/processed/semantic_epochs/zuna_full5s_backaligned_sub01_runs01_08/all_runs_metadata.csv \
  --epochs-dir data/processed/semantic_epochs/zuna_full5s_backaligned_sub01_runs01_08 \
  --common-embeddings data/processed/clip_embeddings/common_embeddings.pt \
  --val-runs 8 \
  --window-mode full5s_backaligned \
  --target-space common \
  --model temporal_attn_small \
  --epochs 50 \
  --batch-size 64 \
  --device cuda \
  --slug full5s_recovery \
  --add-event-marker \
  --augment-eeg \
  --conditions zuna_real zuna_shuffled zuna_random

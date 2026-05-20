#!/bin/bash
set -e

# Run Baseline Combined
echo "=== Running Baseline Combined ==="
PYTHONPATH=src python3 scripts/train_eeg_clip.py \
  --metadata data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv,data/processed/semantic_epochs/zuna_tight1s_sub02_runs01_32/all_runs_metadata.csv \
  --epochs-dir data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40,data/processed/semantic_epochs/zuna_tight1s_sub02_runs01_32 \
  --common-embeddings data/processed/clip_embeddings/combined_common_embeddings.pt \
  --input-domain zuna \
  --target-mode real \
  --target-space common \
  --output-dir outputs/phase6_compare \
  --slug phase6_baseline_combined_recovery \
  --window-mode tight1s \
  --add-event-marker \
  --model temporal_attn_small \
  --augment-eeg \
  --epochs 50 \
  --batch-size 64 \
  --device cuda > phase6_baseline_combined_recovery.log 2>&1

echo "=== Running Multitask Combined ==="
PYTHONPATH=src python3 scripts/train_eeg_clip.py \
  --metadata data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv,data/processed/semantic_epochs/zuna_tight1s_sub02_runs01_32/all_runs_metadata.csv \
  --epochs-dir data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40,data/processed/semantic_epochs/zuna_tight1s_sub02_runs01_32 \
  --common-embeddings data/processed/clip_embeddings/combined_common_embeddings.pt \
  --input-domain zuna \
  --target-mode real \
  --target-space common \
  --output-dir outputs/phase6_compare \
  --slug phase6_multitask_combined_recovery \
  --window-mode tight1s \
  --add-event-marker \
  --model temporal_attn_small \
  --augment-eeg \
  --epochs 80 \
  --batch-size 64 \
  --vlm-attributes data/processed/vlm_attributes.json \
  --aux-warmup-epochs 20 \
  --device cuda > phase6_multitask_combined_recovery.log 2>&1

echo "=== Done ==="

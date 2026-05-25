#!/bin/bash
set -e

cd /workspace/mindeye
export PYTHONPATH=src
export HF_HOME=/workspace/hf_cache
export TMPDIR=/workspace/tmp
mkdir -p /workspace/hf_cache /workspace/tmp

echo "=== Starting Matrix B (Dual-head, Fixed Mean Norm) ==="
python scripts/train_eeg_clip.py \
  --common-embeddings data/processed/clip_embeddings/decode_common_embeddings.pt \
  --loss contrastive \
  --model temporal_attn_small \
  --target-space decode_unit \
  --add-event-marker \
  --augment-eeg \
  --dual-head \
  --use-fixed-mean-norm \
  --metadata data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv \
  --epochs-dir data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40 \
  --window-mode tight1s \
  --epochs 15 \
  --target-mode real \
  --slug B_real

python scripts/train_eeg_clip.py \
  --common-embeddings data/processed/clip_embeddings/decode_common_embeddings.pt \
  --loss contrastive \
  --model temporal_attn_small \
  --target-space decode_unit \
  --add-event-marker \
  --augment-eeg \
  --dual-head \
  --use-fixed-mean-norm \
  --metadata data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv \
  --epochs-dir data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40 \
  --window-mode tight1s \
  --epochs 15 \
  --target-mode shuffled \
  --slug B_shuffled

python scripts/train_eeg_clip.py \
  --common-embeddings data/processed/clip_embeddings/decode_common_embeddings.pt \
  --loss contrastive \
  --model temporal_attn_small \
  --target-space decode_unit \
  --add-event-marker \
  --augment-eeg \
  --dual-head \
  --use-fixed-mean-norm \
  --metadata data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv \
  --epochs-dir data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40 \
  --window-mode tight1s \
  --epochs 15 \
  --target-mode random \
  --slug B_random

echo "=== Starting Matrix C (Dual-head, Learned Norm) ==="
python scripts/train_eeg_clip.py \
  --common-embeddings data/processed/clip_embeddings/decode_common_embeddings.pt \
  --loss contrastive \
  --model temporal_attn_small \
  --target-space decode_unit \
  --add-event-marker \
  --augment-eeg \
  --dual-head \
  --metadata data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv \
  --epochs-dir data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40 \
  --window-mode tight1s \
  --epochs 15 \
  --target-mode real \
  --slug C_real

python scripts/train_eeg_clip.py \
  --common-embeddings data/processed/clip_embeddings/decode_common_embeddings.pt \
  --loss contrastive \
  --model temporal_attn_small \
  --target-space decode_unit \
  --add-event-marker \
  --augment-eeg \
  --dual-head \
  --metadata data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv \
  --epochs-dir data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40 \
  --window-mode tight1s \
  --epochs 15 \
  --target-mode shuffled \
  --slug C_shuffled

python scripts/train_eeg_clip.py \
  --common-embeddings data/processed/clip_embeddings/decode_common_embeddings.pt \
  --loss contrastive \
  --model temporal_attn_small \
  --target-space decode_unit \
  --add-event-marker \
  --augment-eeg \
  --dual-head \
  --metadata data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv \
  --epochs-dir data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40 \
  --window-mode tight1s \
  --epochs 15 \
  --target-mode random \
  --slug C_random

echo "=== All runs completed successfully ==="

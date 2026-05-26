#!/bin/bash
set -e

cd /workspace/mindeye
source venv/bin/activate
export PYTHONPATH=/workspace/mindeye/src

echo "=== [1/3] Waiting for generate_vlm_attributes.py to complete ==="
while ps aux | grep -v grep | grep -q "generate_vlm_attributes.py"; do
    echo "VLM generation still running... checking again in 60s"
    sleep 60
done

echo "VLM generation process has completed!"

echo "=== [2/3] Auditing final VLM attribute coverage ==="
python3 scripts/analyze_vlm_attributes.py

echo "=== [3/3] Launching multi-subject training with active probe regularization ==="
python3 -u scripts/train_eeg_clip.py \
  --metadata "data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv,data/processed/semantic_epochs/zuna_tight1s_sub02_runs01_40/all_runs_metadata.csv,data/processed/semantic_epochs/zuna_tight1s_sub03_runs01_40/all_runs_metadata.csv,data/processed/semantic_epochs/zuna_tight1s_sub04_runs01_40/all_runs_metadata.csv" \
  --epochs-dir "data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40,data/processed/semantic_epochs/zuna_tight1s_sub02_runs01_40,data/processed/semantic_epochs/zuna_tight1s_sub03_runs01_40,data/processed/semantic_epochs/zuna_tight1s_sub04_runs01_40" \
  --common-embeddings data/processed/clip_embeddings/decode_common_embeddings.pt \
  --model temporal_attn_small \
  --window-mode tight1s \
  --target-space decode_unit \
  --target-mode real \
  --epochs 30 \
  --lr 0.0003 \
  --batch-size 64 \
  --stem-dropout1d 0.15 \
  --dropout 0.35 \
  --augment-eeg \
  --slug 14_multisubj_probe_active \
  --calibration-weight 0.05 \
  --common-probe outputs/decode_probe_v2/common_probe.pt \
  --probe-weight 0.01 \
  --vlm-attributes data/processed/clip_embeddings/vlm_attributes.json \
  > /tmp/train_multisubj_probe_active.log 2>&1

echo "Training completed successfully!"

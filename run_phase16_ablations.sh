#!/bin/bash
set -e
cd /workspace/mindeye
. venv/bin/activate
export PYTHONPATH=/workspace/mindeye/src

METADATA="data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv,data/processed/semantic_epochs/zuna_tight1s_sub02_runs01_40/all_runs_metadata.csv,data/processed/semantic_epochs/zuna_tight1s_sub03_runs01_40/all_runs_metadata.csv,data/processed/semantic_epochs/zuna_tight1s_sub04_runs01_40/all_runs_metadata.csv"
EPOCHS_DIR="data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40,data/processed/semantic_epochs/zuna_tight1s_sub02_runs01_40,data/processed/semantic_epochs/zuna_tight1s_sub03_runs01_40,data/processed/semantic_epochs/zuna_tight1s_sub04_runs01_40"
INIT_FROM="outputs/runs/20260527_013743_zuna_real_14_multisubj_probe_active/best.pt"
COMMON="data/processed/clip_embeddings/decode_common_embeddings.pt"
PROBE="outputs/decode_probe_v2/common_probe.pt"
VLM="data/processed/clip_embeddings/vlm_attributes.json"

echo "=============================="
echo "VARIANT A: Shared baseline (no adapters, init from Phase 15)"
echo "=============================="
python3 scripts/train_eeg_clip.py \
  --metadata "$METADATA" \
  --epochs-dir "$EPOCHS_DIR" \
  --common-embeddings "$COMMON" \
  --common-probe "$PROBE" \
  --vlm-attributes "$VLM" \
  --target-space decode_unit \
  --model temporal_attn_small \
  --window-mode tight1s \
  --epochs 30 --batch-size 64 --lr 3e-4 \
  --dropout 0.35 --stem-dropout1d 0.15 \
  --augment-eeg --probe-weight 0.01 \
  --probe-start-epoch 1 --aux-start-epoch 1 \
  --aux-warmup-epochs 20 --warmup-epochs 5 \
  --min-lr 1e-6 --patience 15 --seed 13 \
  --init-from "$INIT_FROM" \
  --no-film --no-subject-heads --head-reg-weight 0.0 \
  --slug 16a_shared_baseline \
  2>&1 | tee /workspace/mindeye/logs_16a.txt
echo "Variant A done"

echo "=============================="
echo "VARIANT B: Subject heads only (no FiLM, init from Phase 15)"
echo "=============================="
python3 scripts/train_eeg_clip.py \
  --metadata "$METADATA" \
  --epochs-dir "$EPOCHS_DIR" \
  --common-embeddings "$COMMON" \
  --common-probe "$PROBE" \
  --vlm-attributes "$VLM" \
  --target-space decode_unit \
  --model temporal_attn_small \
  --window-mode tight1s \
  --epochs 30 --batch-size 64 --lr 3e-4 \
  --dropout 0.35 --stem-dropout1d 0.15 \
  --augment-eeg --probe-weight 0.01 \
  --probe-start-epoch 1 --aux-start-epoch 1 \
  --aux-warmup-epochs 20 --warmup-epochs 5 \
  --min-lr 1e-6 --patience 15 --seed 13 \
  --init-from "$INIT_FROM" \
  --no-film --head-reg-weight 1e-4 \
  --slug 16b_heads_only \
  2>&1 | tee /workspace/mindeye/logs_16b.txt
echo "Variant B done"

echo "=============================="
echo "VARIANT C: FiLM + subject heads (init from Phase 15)"
echo "=============================="
python3 scripts/train_eeg_clip.py \
  --metadata "$METADATA" \
  --epochs-dir "$EPOCHS_DIR" \
  --common-embeddings "$COMMON" \
  --common-probe "$PROBE" \
  --vlm-attributes "$VLM" \
  --target-space decode_unit \
  --model temporal_attn_small \
  --window-mode tight1s \
  --epochs 30 --batch-size 64 --lr 3e-4 \
  --dropout 0.35 --stem-dropout1d 0.15 \
  --augment-eeg --probe-weight 0.01 \
  --probe-start-epoch 1 --aux-start-epoch 1 \
  --aux-warmup-epochs 20 --warmup-epochs 5 \
  --min-lr 1e-6 --patience 15 --seed 13 \
  --init-from "$INIT_FROM" \
  --head-reg-weight 1e-4 \
  --slug 16c_film_heads \
  2>&1 | tee /workspace/mindeye/logs_16c.txt
echo "Variant C done"

echo "ALL ABLATIONS COMPLETE"

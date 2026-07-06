#!/usr/bin/env bash
# ZUNA -> vision QFormer control matrix: real vs shuffled vs random.
# Uses the full cohort9 ZUNA cache. Window tc 15-31 matches this cache's recorded
# onset (onset_tc=19); this is exactly how ZUNA was being used in prior sweeps. The gate is whether target-mode=real beats
# the shuffled/random controls on val retrieval (val_mrr_full / top-k). If real
# only matches the controls, the bridge is not extracting stimulus signal.
# Usage: bash scripts/sweep_cos_negbank.sh 2>&1 | tee /workspace/matrix.log
set -euo pipefail

LATENTS="data/processed/zuna_latents/cohort9_runs01_32"
RAE="data/processed/rae_embeddings/rae_dinov2_base_all.pt"
TSPACE="DINO-CLS-768"
OUT_ROOT="outputs/qformer_control_matrix"

mkdir -p "$OUT_ROOT"

for MODE in real shuffled random; do
  CELL="$TSPACE-$MODE"
  echo "======================================================================"
  echo "  CONTROL CELL: target-mode=$MODE  target-space=$TSPACE"
  echo "======================================================================"
  # negative bank is real-mode only (queue ids must align with targets).
  NEGBANK=0
  PYTHONPATH=src python -u scripts/train_zuna_to_vision.py \
    --latents-pt "$LATENTS" \
    --targets-pt "$RAE" \
    --target-space "$TSPACE" \
    --target-mode "$MODE" \
    --layer-name post_mmd \
    --num-subjects 9 \
    --temporal-window --latent-tc-start 15 --latent-tc-end 31 \
    --batch-size 256 \
    --epochs 20 \
    --patience 20 \
    --train-runs "1-6" --val-runs "7-8" \
    --hidden-dim 1024 \
    --temperature 0.05 \
    --nce-weight 1.0 \
    --cos-weight 0.2 \
    --var-weight 0.05 \
    --spread-weight 0.3 \
    --negative-bank-size "$NEGBANK" \
    --out-dir "$OUT_ROOT/$CELL" \
    --slug "$CELL"
  # Free the heavy checkpoints; keep the tiny metrics/history for analysis.
  find "$OUT_ROOT/$CELL" -name '*.pt' -delete
done

echo "======================================================================"
echo "  CONTROL MATRIX COMPLETE"
echo "======================================================================"

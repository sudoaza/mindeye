#!/usr/bin/env bash
# Fast diagnostic sweep (real-mode only, sequential, unbuffered): does raising
# InfoNCE negatives (batch 256 + neg-bank) let us lower cos-weight without
# inducing directional collapse?
#
# We only need the real run's collapse_pct / StdRatio to judge health, so we
# skip the shuffled/random controls (needed only for the final gate) and run one
# process at a time to avoid GPU contention. Live per-epoch logs via python -u.
#
# Usage: bash scripts/sweep_cos_negbank.sh 2>&1 | tee /workspace/sweep.log
set -euo pipefail

LATENTS="data/processed/zuna_latents/cohort9_runs01_32"
RAE="data/processed/rae_embeddings/rae_dinov2_base_all.pt"
OUT_ROOT="outputs/qformer_cos_sweep"

mkdir -p "$OUT_ROOT"

for COS in 1.0 0.5 0.2; do
  for NEGBANK in 0 16384; do
    CELL="cos${COS}_neg${NEGBANK}"
    echo "======================================================================"
    echo "  SWEEP CELL: cos-weight=$COS  negative-bank=$NEGBANK  (real only)"
    echo "======================================================================"
    PYTHONPATH=src python -u scripts/train_zuna_to_vision.py \
      --latents-pt "$LATENTS" \
      --targets-pt "$RAE" \
      --target-space "DINO-Unit-768" \
      --target-mode real \
      --layer-name post_mmd \
      --num-subjects 9 \
      --temporal-window --latent-tc-start 15 --latent-tc-end 31 \
      --batch-size 256 \
      --epochs 10 \
      --patience 10 \
      --train-runs "1-6" --val-runs "7-8" \
      --hidden-dim 1024 \
      --temperature 0.05 \
      --nce-weight 1.0 \
      --cos-weight "$COS" \
      --var-weight 0.05 \
      --negative-bank-size "$NEGBANK" \
      --out-dir "$OUT_ROOT/$CELL" \
      --slug "$CELL"
  done
done

echo "======================================================================"
echo "  SWEEP COMPLETE"
echo "======================================================================"

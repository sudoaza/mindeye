#!/usr/bin/env bash
# Anti-collapse fix validation (real-mode only, sequential, unbuffered):
# does the VICReg-style cross-sample spread term break hub collapse?
#
# The cos-weight sweep showed hub collapse persists even at cos=1.0 (all preds
# converge to the mean-target direction; the old variance-floor is blind to this
# under force_unit_output). This sweep tests the new --spread-weight term.
# spread=0 already reproduced the collapse; here we test spread>0.
#
# NOTE: /workspace overlay disk is small; each checkpoint is ~830MB. We delete
# each cell's heavy checkpoints right after the run (history.csv/metrics.json,
# which hold collapse_pct/StdRatio, are tiny and kept).
#
# Usage: bash scripts/sweep_cos_negbank.sh 2>&1 | tee /workspace/sweep.log
set -euo pipefail

LATENTS="data/processed/zuna_latents/cohort9_runs01_32"
RAE="data/processed/rae_embeddings/rae_dinov2_base_all.pt"
OUT_ROOT="outputs/qformer_spread_sweep"

mkdir -p "$OUT_ROOT"

for SPREAD in 1.0 5.0; do
  CELL="spread${SPREAD}"
  echo "======================================================================"
  echo "  SWEEP CELL: spread-weight=$SPREAD  (cos=0.2, real only)"
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
    --epochs 12 \
    --patience 12 \
    --train-runs "1-6" --val-runs "7-8" \
    --hidden-dim 1024 \
    --temperature 0.05 \
    --nce-weight 1.0 \
    --cos-weight 0.2 \
    --var-weight 0.05 \
    --spread-weight "$SPREAD" \
    --negative-bank-size 0 \
    --out-dir "$OUT_ROOT/$CELL" \
    --slug "$CELL"
  # Free the heavy checkpoints; keep the tiny metrics/history for analysis.
  find "$OUT_ROOT/$CELL" -name '*.pt' -delete
done

echo "======================================================================"
echo "  SWEEP COMPLETE — per-cell final-epoch collapse/StdRatio:"
echo "======================================================================"
for h in "$OUT_ROOT"/*/*/history.csv; do
  echo "--- $h ---"
  head -1 "$h"; tail -1 "$h"
done

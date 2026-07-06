#!/usr/bin/env bash
# Target-space sweep: with the loss now healthy (no collapse), does ANY visual
# target space give the EEG->vision bridge above-chance *generalization* on val?
#
# The spread-weight sweep showed the loss is fixed (collapse 100%->0%, StdRatio in
# band) but EEG->DINO-Unit-768 overfits: train loss falls while val MRR stays at
# chance. This sweep varies the target space smaller<->richer to locate signal:
#   DINO-PCA-128-Unit  (128-d, easiest)
#   DINO-PCA-256-Unit  (256-d)
#   DINO-Unit-768      (768-d mean-pool, baseline)
#   DINO-CLS-768       (768-d CLS token, richer/different)
# (CLIP-Common-512 is NOT in this bank; would need re-caching, skipped.)
#
# real-only, spread=0.3 (best balance), 20 epochs. Watch val_mrr_norm / mrr_full.
# Usage: bash scripts/sweep_cos_negbank.sh 2>&1 | tee /workspace/sweep.log
set -euo pipefail

LATENTS="data/processed/zuna_latents/cohort9_runs01_32"
RAE="data/processed/rae_embeddings/rae_dinov2_base_all.pt"
OUT_ROOT="outputs/qformer_target_sweep"

mkdir -p "$OUT_ROOT"

for TSPACE in DINO-PCA-128-Unit DINO-PCA-256-Unit DINO-Unit-768 DINO-CLS-768; do
  CELL="$TSPACE"
  echo "======================================================================"
  echo "  SWEEP CELL: target-space=$TSPACE  (spread=0.3, cos=0.2, real only)"
  echo "======================================================================"
  PYTHONPATH=src python -u scripts/train_zuna_to_vision.py \
    --latents-pt "$LATENTS" \
    --targets-pt "$RAE" \
    --target-space "$TSPACE" \
    --target-mode real \
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
    --negative-bank-size 0 \
    --out-dir "$OUT_ROOT/$CELL" \
    --slug "$CELL"
  # Free the heavy checkpoints; keep the tiny metrics/history for analysis.
  find "$OUT_ROOT/$CELL" -name '*.pt' -delete
done

echo "======================================================================"
echo "  SWEEP COMPLETE"
echo "======================================================================"

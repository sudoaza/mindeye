#!/usr/bin/env bash
# Fast diagnostic sweep: does raising InfoNCE negatives (batch 256 + neg-bank)
# let us lower cos-weight without inducing directional collapse?
#
# Each cell runs a short DINO-Unit-768 grid (real + shuffled + random controls,
# controls auto-capped at 8 epochs by run_qformer_grid.py). We read collapse_pct
# / StdRatio from each real run to see which (cos, negbank) combos stay healthy.
#
# Usage: bash scripts/sweep_cos_negbank.sh 2>&1 | tee /workspace/sweep.log
set -euo pipefail

LATENTS="data/processed/zuna_latents/cohort9_runs01_32"
RAE="data/processed/rae_embeddings/rae_dinov2_base_all.pt"
OUT_ROOT="outputs/qformer_cos_sweep"
COMMON=(
  --latents-pt "$LATENTS"
  --rae-pt "$RAE"
  --target-spaces "DINO-Unit-768"
  --num-subjects 9
  --latent-tc-start 15 --latent-tc-end 31
  --batch-size 256
  --epochs 8
  --patience 8
  --train-runs "1-6" --val-runs "7-8"
  --hidden-dim 1024
  --temperature 0.05
  --nce-weight 1.0
  --var-weight 0.05
)

mkdir -p "$OUT_ROOT"

for COS in 1.0 0.5 0.2; do
  for NEGBANK in 0 16384; do
    CELL="cos${COS}_neg${NEGBANK}"
    echo "======================================================================"
    echo "  SWEEP CELL: cos-weight=$COS  negative-bank=$NEGBANK"
    echo "======================================================================"
    PYTHONPATH=src python scripts/run_qformer_grid.py \
      "${COMMON[@]}" \
      --cos-weight "$COS" \
      --negative-bank-size "$NEGBANK" \
      --out-dir "$OUT_ROOT/$CELL"
  done
done

echo "======================================================================"
echo "  SWEEP COMPLETE — collapse/StdRatio per cell:"
echo "======================================================================"
for d in "$OUT_ROOT"/*/grid_*/runs_summary.csv; do
  echo "--- $d ---"
  head -2 "$d"
done

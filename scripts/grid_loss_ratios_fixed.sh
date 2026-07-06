#!/usr/bin/env bash
# Loss-ratio hyperparameter grid on the FIXED ZUNA cache (global-norm extractor,
# onset_tc=24). Now that the extractor bug is fixed, re-sweep the loss balance and
# judge signal by vector distance (val_cosine_norm), not the near-chance full-bank
# image-id retrieval. Controls (shuffled/random) run only 8 epochs.
#
# The gate: does target-mode=real reach a higher val_cosine_norm than the
# shuffled/random controls for any loss config? If real == controls, no signal.
# Usage: bash scripts/grid_loss_ratios_fixed.sh 2>&1 | tee /workspace/grid.log
set -euo pipefail

LATENTS="data/processed/zuna_latents/sub01_runs01_32_fixed"
RAE="data/processed/rae_embeddings/rae_dinov2_base_all.pt"
TSPACE="DINO-CLS-768"
OUT_ROOT="outputs/qformer_loss_grid_fixed"
mkdir -p "$OUT_ROOT"

# Shared config. Window brackets the corrected onset (onset_tc=24).
COMMON=(
  --latents-pt "$LATENTS"
  --targets-pt "$RAE"
  --target-space "$TSPACE"
  --layer-name post_mmd
  --num-subjects 1
  --temporal-window --latent-tc-start 20 --latent-tc-end 36
  --batch-size 256
  --patience 20
  --train-runs "1-24" --val-runs "25-28" --test-runs "29-32"
  --hidden-dim 1024
  --nce-weight 1.0
  --var-weight 0.05
  --negative-bank-size 0
  --select-metric cosine
  --full-bank-eval final
)

run_cell () {
  local mode="$1" temp="$2" cos="$3" spread="$4" epochs="$5"
  local cell="${mode}_t${temp}_cos${cos}_sp${spread}"
  echo "======================================================================"
  echo "  CELL: mode=$mode temp=$temp cos=$cos spread=$spread epochs=$epochs"
  echo "======================================================================"
  PYTHONPATH=src python -u scripts/train_zuna_to_vision.py \
    "${COMMON[@]}" \
    --target-mode "$mode" \
    --temperature "$temp" \
    --cos-weight "$cos" \
    --spread-weight "$spread" \
    --epochs "$epochs" \
    --out-dir "$OUT_ROOT/$cell" \
    --slug "$cell"
  find "$OUT_ROOT/$cell" -name '*.pt' -delete
}

# --- Loss-ratio grid on REAL (20 epochs each) ---
for TEMP in 0.05 0.1; do
  for COS in 0.2 0.5 1.0; do
    for SPREAD in 0.3 0.5; do
      run_cell real "$TEMP" "$COS" "$SPREAD" 20
    done
  done
done

# --- Controls (8 epochs each) at the mid config for a chance baseline ---
run_cell shuffled 0.05 0.5 0.3 8
run_cell random   0.05 0.5 0.3 8

echo "======================================================================"
echo "  LOSS GRID COMPLETE"
echo "======================================================================"

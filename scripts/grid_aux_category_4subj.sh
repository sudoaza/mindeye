#!/usr/bin/env bash
# Auxiliary coarse-category grounding test on the FIXED 4-subject ZUNA cache.
# Hypothesis (user): the old ViT-CLIP pipeline "worked" partly because auxiliary
# semantic tasks grounded the embedding. Keep DINO as the retrieval target but add
# a jointly-trained ~20-way WordNet-lexname category head (--cat-weight) and test
# whether the honest full-bank real signal finally clears the shuffled control.
#
# Protocol: identical bridge/loss config as grid_loss_ratios_4subj.sh (best cell:
# temp0.05 cos1.0 spread0.3). Two real arms (no-aux baseline vs aux-grounded) and a
# shuffled control for each, judged by val_cosine_norm with a final full-bank eval.
# Usage: bash scripts/grid_aux_category_4subj.sh 2>&1 | tee /workspace/grid_aux.log
set -euo pipefail

LATENTS="data/processed/zuna_latents/sub01_04_runs01_32_fixed"
RAE="data/processed/rae_embeddings/rae_dinov2_base_all.pt"
TSPACE="DINO-CLS-768"
OUT_ROOT="outputs/qformer_aux_cat_4subj"
mkdir -p "$OUT_ROOT"

COMMON=(
  --latents-pt "$LATENTS"
  --targets-pt "$RAE"
  --target-space "$TSPACE"
  --layer-name post_mmd
  --num-subjects 4
  --temporal-window --latent-tc-start 20 --latent-tc-end 36
  --batch-size 256
  --patience 20
  --train-runs "1-24" --val-runs "25-28" --test-runs "29-32"
  --hidden-dim 1024
  --nce-weight 1.0
  --cos-weight 1.0
  --var-weight 0.05
  --spread-weight 0.3
  --temperature 0.05
  --negative-bank-size 0
  --select-metric cosine
  --full-bank-eval final
)

run_cell () {
  local mode="$1" catw="$2" epochs="$3"
  local cell="${mode}_cat${catw}"
  echo "======================================================================"
  echo "  CELL: mode=$mode cat_weight=$catw epochs=$epochs"
  echo "======================================================================"
  PYTHONPATH=src python -u scripts/train_zuna_to_vision.py \
    "${COMMON[@]}" \
    --target-mode "$mode" \
    --cat-weight "$catw" \
    --epochs "$epochs" \
    --out-dir "$OUT_ROOT/$cell" \
    --slug "$cell"
  find "$OUT_ROOT/$cell" -name '*.pt' -delete
}

# --- Real arms: no-aux baseline vs aux-grounded (two cat weights) ---
run_cell real 0.0 20
run_cell real 0.5 20
run_cell real 1.0 20

# --- Shuffled controls at matching configs (8 epochs, per project convention) ---
run_cell shuffled 0.0 8
run_cell shuffled 1.0 8

echo "======================================================================"
echo "  AUX-CATEGORY GRID COMPLETE"
echo "======================================================================"

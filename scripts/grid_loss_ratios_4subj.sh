#!/usr/bin/env bash
# 4-subject loss grid on the FIXED ZUNA cache (global-norm extractor, onset_tc=24).
# More data (sub01-04) — the raw-EEG coarse probe improved with more runs, so test
# whether the ZUNA->DINO bridge shows above-control vector-distance signal at 4x
# the data. Judged by val_cosine_norm. Controls (shuffled/random) run 8 epochs at
# the best real loss config so the comparison is apples-to-apples.
# Usage: bash scripts/grid_loss_ratios_4subj.sh 2>&1 | tee /workspace/grid4.log
set -euo pipefail

LATENTS="data/processed/zuna_latents/sub01_04_runs01_32_fixed"
RAE="data/processed/rae_embeddings/rae_dinov2_base_all.pt"
TSPACE="DINO-CLS-768"
OUT_ROOT="outputs/qformer_loss_grid_4subj"
mkdir -p "$OUT_ROOT"

# Shared config. Window brackets the corrected onset (onset_tc=24). Splits are by
# run and apply within every subject (run ids repeat across the 4 subjects).
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

# --- Focused real grid (cos-weight was the only lever last time) ---
run_cell real 0.05 0.5 0.3 20
run_cell real 0.05 1.0 0.3 20
run_cell real 0.10 1.0 0.3 20

# --- Controls at the best real config (cos=1.0) for a fair baseline ---
run_cell shuffled 0.05 1.0 0.3 8
run_cell random   0.05 1.0 0.3 8

echo "======================================================================"
echo "  4-SUBJECT LOSS GRID COMPLETE"
echo "======================================================================"

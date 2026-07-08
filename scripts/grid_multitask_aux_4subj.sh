#!/usr/bin/env bash
# Full auxiliary-task grounding suite vs controls on the FIXED 4-subject ZUNA cache.
# The old ViT-CLIP pipeline paired InfoNCE with a rich grounding stack (semantic probe
# + 29-way VLM multitask + dual-head raw MSE). We reproduce the SPIRIT with tasks
# derivable on-pod (no VLM regen): coarse_category, fine_category, is_animate (WordNet),
# dominant_color, brightness (from stimulus images), plus a dual-head raw-target MSE.
# DINO stays the retrieval target; the honest gate is full-bank real vs shuffled.
#
# Arms: real no-aux baseline, real full-suite (aux+raw-mse), real aux-only, real raw-mse-only,
# and shuffled controls for baseline + full-suite. Judged by val_cosine_norm with a final
# full-bank eval (kept in val_eval_preds.pt for honest real-shuffled deltas).
# Usage: bash scripts/grid_multitask_aux_4subj.sh 2>&1 | tee /workspace/grid_mt.log
set -euo pipefail

LATENTS="data/processed/zuna_latents/sub01_04_runs01_32_fixed"
RAE="data/processed/rae_embeddings/rae_dinov2_base_all.pt"
TSPACE="DINO-CLS-768"
OUT_ROOT="outputs/qformer_multitask_aux_4subj"
ALL_TASKS="coarse_category,fine_category,is_animate,dominant_color,brightness"
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

# args: mode auxw rawmsew epochs slug [extra flags...]
run_cell () {
  local mode="$1" auxw="$2" rawmsew="$3" epochs="$4" slug="$5"; shift 5
  echo "======================================================================"
  echo "  CELL: mode=$mode aux_w=$auxw raw_mse_w=$rawmsew epochs=$epochs slug=$slug"
  echo "======================================================================"
  PYTHONPATH=src python -u scripts/train_zuna_to_vision.py \
    "${COMMON[@]}" \
    --target-mode "$mode" \
    --aux-tasks "$ALL_TASKS" \
    --aux-weight "$auxw" \
    --raw-mse-weight "$rawmsew" \
    --epochs "$epochs" \
    --out-dir "$OUT_ROOT/$slug" \
    --slug "$slug" \
    "$@"
  # Keep val/test eval preds (hold the honest full-bank ranks); drop only heavy checkpoints.
  find "$OUT_ROOT/$slug" -name 'checkpoint_best.pt' -delete
  find "$OUT_ROOT/$slug" -name 'model_final.pt' -delete
}

# --- Real arms ---
run_cell real 0.0 0.0 20 real_baseline
run_cell real 0.5 0.1 20 real_fullsuite
run_cell real 0.5 0.0 20 real_auxonly
run_cell real 0.0 0.1 20 real_rawmseonly

# --- Shuffled controls (8 epochs) at baseline + full-suite ---
run_cell shuffled 0.0 0.0 8 shuf_baseline
run_cell shuffled 0.5 0.1 8 shuf_fullsuite

echo "======================================================================"
echo "  MULTITASK AUX GRID COMPLETE"
echo "======================================================================"

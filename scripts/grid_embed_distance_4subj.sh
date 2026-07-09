#!/usr/bin/env bash
# Embedding-distance gate (NOT exact-image retrieval) on the FIXED 4-subject ZUNA cache.
#
# Reframing: per-image top-k on a 34k bank is near-impossible by construction and the
# wrong success criterion for EEG. The honest question is whether the predicted embedding
# lands *near the true DINO embedding* and in the *right semantic neighborhood* — measured
# by cos_margin (cos_true - cos_rand) and neighbor category purity, judged real vs shuffled.
#
# metrics.json now carries an "embedding_distance" block on the best checkpoint:
#   val_cos_true, val_cos_rand, val_cos_margin, val_rank_percentile,
#   val_neighbor_cat_acc@{10,50}, val_neighbor_cat_chance
# The gate is: real cos_margin > shuffled cos_margin (Δ>0, and neighbor purity > chance).
#
# Arms: real + shuffled control at (a) plain contrastive+cos and (b) soft-DINO neighborhood
# loss, which directly optimizes the embedding-distance objective we now gate on.
# Usage: bash scripts/grid_embed_distance_4subj.sh 2>&1 | tee /workspace/grid_embed.log
set -euo pipefail

LATENTS="data/processed/zuna_latents/sub01_04_runs01_32_fixed"
RAE="data/processed/rae_embeddings/rae_dinov2_base_all.pt"
TSPACE="DINO-CLS-768"
OUT_ROOT="outputs/qformer_embed_distance_4subj"
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
  --temperature 0.05
  --negative-bank-size 0
  --select-metric cosine
  --full-bank-eval final
)

# args: mode epochs slug [extra loss flags...]
run_cell () {
  local mode="$1" epochs="$2" slug="$3"; shift 3
  echo "======================================================================"
  echo "  CELL: mode=$mode epochs=$epochs slug=$slug  extra: $*"
  echo "======================================================================"
  PYTHONPATH=src python -u scripts/train_zuna_to_vision.py \
    "${COMMON[@]}" \
    --target-mode "$mode" \
    --epochs "$epochs" \
    --out-dir "$OUT_ROOT/$slug" \
    --slug "$slug" \
    "$@"
  # Keep val/test eval preds (honest embedding-distance ranks); drop heavy checkpoints.
  find "$OUT_ROOT/$slug" -name 'checkpoint_best.pt' -delete
  find "$OUT_ROOT/$slug" -name 'model_final.pt' -delete
}

# --- Arm A: plain contrastive + cosine (baseline embedding objective) ---
A_LOSS=(--nce-weight 1.0 --cos-weight 1.0 --var-weight 0.05 --spread-weight 0.3 --soft-dino-weight 0.0)
run_cell real     20 real_contrastive "${A_LOSS[@]}"
run_cell shuffled 8  shuf_contrastive "${A_LOSS[@]}"

# --- Arm B: soft-DINO neighborhood loss (directly optimizes embedding distance) ---
B_LOSS=(--nce-weight 0.0 --cos-weight 1.0 --var-weight 0.05 --spread-weight 0.3 \
        --soft-dino-weight 1.0 --soft-dino-teacher-temp 0.1 --soft-dino-rkd-weight 0.5)
run_cell real     20 real_softdino "${B_LOSS[@]}"
run_cell shuffled 8  shuf_softdino "${B_LOSS[@]}"

echo "======================================================================"
echo "  EMBEDDING-DISTANCE GRID COMPLETE — compare metrics.json embedding_distance blocks"
echo "  gate: real val_cos_margin > shuffled, and val_neighbor_cat_acc@10 > chance"
echo "======================================================================"

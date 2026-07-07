#!/usr/bin/env bash
# Arm comparison: where is the EEG->vision signal lost? Run the SAME controlled
# bridge protocol (real vs shuffled vs random, judged by val_cosine_norm) on
# different input representations of the SAME sub-01 data:
#   raw       : globally-normed EEG chopped to tf tokens, NO ZUNA encoder (32-d)
#   post_mmd  : ZUNA final bottleneck latent (32-d)          [existing full cache]
#   layer_8   : ZUNA mid encoder hidden state (1024-d)       [windowed cache]
#   layer_12  : ZUNA deeper encoder hidden state (1024-d)    [windowed cache]
#
# Interpretation:
#   raw >> post_mmd  => ZUNA's encoder is destroying signal (bottleneck/collapse)
#   layer_k > post_mmd => richer layers retain signal the bottleneck drops
#   all ~= shuffled  => no per-image signal decodable at this scale, regardless of arm
#
# Best loss config from the fixed grid: temp=0.1 cos=1.0 spread=0.3.
# Usage: bash scripts/grid_arms_sub01.sh <arm> 2>&1 | tee /workspace/arms_<arm>.log
set -euo pipefail

ARM="${1:?usage: grid_arms_sub01.sh <raw|post_mmd|layer_8|layer_12>}"
RAE="data/processed/rae_embeddings/rae_dinov2_base_all.pt"
TSPACE="DINO-CLS-768"
OUT_ROOT="outputs/qformer_arms_sub01/$ARM"
mkdir -p "$OUT_ROOT"

# Per-arm cache + windowing. Windowed caches are pre-cropped => --no-temporal-window.
case "$ARM" in
  post_mmd)
    LATENTS="data/processed/zuna_latents/sub01_runs01_32_fixed"
    LAYER="post_mmd"
    WINDOW=(--temporal-window --latent-tc-start 20 --latent-tc-end 36)
    ;;
  raw)
    LATENTS="data/processed/zuna_latents/sub01_layer8raw_win"
    LAYER="raw"
    WINDOW=(--no-temporal-window)
    ;;
  layer_8)
    LATENTS="data/processed/zuna_latents/sub01_layer8raw_win"
    LAYER="layer_8"
    WINDOW=(--no-temporal-window)
    ;;
  layer_12)
    LATENTS="data/processed/zuna_latents/sub01_layer12_win"
    LAYER="layer_12"
    WINDOW=(--no-temporal-window)
    ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac

COMMON=(
  --latents-pt "$LATENTS"
  --targets-pt "$RAE"
  --target-space "$TSPACE"
  --layer-name "$LAYER"
  --num-subjects 1
  "${WINDOW[@]}"
  --batch-size 256
  --patience 20
  --train-runs "1-24" --val-runs "25-28" --test-runs "29-32"
  --hidden-dim 1024
  --nce-weight 1.0
  --var-weight 0.05
  --cos-weight 1.0
  --temperature 0.1
  --spread-weight 0.3
  --negative-bank-size 0
  --select-metric cosine
  --full-bank-eval final
)

run_cell () {
  local mode="$1" epochs="$2"
  local cell="${ARM}_${mode}"
  echo "======================================================================"
  echo "  ARM=$ARM  mode=$mode  epochs=$epochs  (layer=$LAYER)"
  echo "======================================================================"
  PYTHONPATH=src python -u scripts/train_zuna_to_vision.py \
    "${COMMON[@]}" \
    --target-mode "$mode" \
    --epochs "$epochs" \
    --out-dir "$OUT_ROOT/$cell" \
    --slug "$cell"
  find "$OUT_ROOT/$cell" -name '*.pt' -delete
}

run_cell real     20
run_cell shuffled 8
run_cell random   8

echo "======================================================================"
echo "  ARM $ARM COMPLETE"
echo "======================================================================"

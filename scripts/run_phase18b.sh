#!/bin/bash
# Phase 18B: EEG → RAE spatial bottleneck code training + evaluation
# Target: spatial_768x3x3 codes  (6,912 values per image)
#
# Run this on the RunPod:
#   bash scripts/run_phase18b.sh

set -e
source venv/bin/activate

export PYTHONPATH=src
export HF_HOME=/workspace/hf_cache
export TMPDIR=/workspace/tmp

METADATA="data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv,data/processed/semantic_epochs/zuna_tight1s_sub02_runs01_40/all_runs_metadata.csv,data/processed/semantic_epochs/zuna_tight1s_sub03_runs01_40/all_runs_metadata.csv,data/processed/semantic_epochs/zuna_tight1s_sub04_runs01_40/all_runs_metadata.csv"
EPOCHS_DIR="data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40,data/processed/semantic_epochs/zuna_tight1s_sub02_runs01_40,data/processed/semantic_epochs/zuna_tight1s_sub03_runs01_40,data/processed/semantic_epochs/zuna_tight1s_sub04_runs01_40"

CODES_BANK="data/processed/rae_embeddings/rae_bottleneck_codes_3x3.pt"
RAE_BANK="data/processed/rae_embeddings/rae_dinov2_base_sub01_04_runs01_40.pt"
BOTTLENECK_CKPT="outputs/rae_bottleneck/spatial_768x3x3/best.pt"
SLUG="phase18b_eeg_to_rae_3x3"

# -----------------------------------------------------------------------
# 1. Train EEG → spatial code (spatial_768x3x3)
# -----------------------------------------------------------------------
echo ""
echo "=== Phase 18B: Training EEG → spatial_768x3x3 codes ==="
echo ""

python3 scripts/train_eeg_clip.py \
    --metadata "$METADATA" \
    --epochs-dir "$EPOCHS_DIR" \
    --common-embeddings "$CODES_BANK" \
    --target-space rae_code \
    --target-key image_id_to_rae_code \
    --window-mode tight1s \
    --augment-eeg \
    --no-target-centering \
    --model temporal_attn_small \
    --epochs 100 \
    --patience 20 \
    --batch-size 128 \
    --loss spatial_cosine \
    --probe-weight 0 \
    --slug "$SLUG" \
    --output-dir outputs \
    --device cuda

echo "Training complete."

# -----------------------------------------------------------------------
# 2. Resolve run directory
# -----------------------------------------------------------------------
RUN_DIR=$(find outputs -maxdepth 1 -name "*_${SLUG}" -type d | sort | tail -n 1)
echo "Resolved run directory: $RUN_DIR"

if [ -z "$RUN_DIR" ]; then
    echo "ERROR: Could not find run directory for slug '$SLUG'"
    exit 1
fi

# -----------------------------------------------------------------------
# 3. Run Phase 18B spatial evaluation
# -----------------------------------------------------------------------
echo ""
echo "=== Phase 18B: RAE Spatial Evaluation ==="
echo ""

python3 scripts/evaluate_rae_spatial.py \
    --run-dir "$RUN_DIR" \
    --bottleneck-checkpoint "$BOTTLENECK_CKPT" \
    --rae-bank "$RAE_BANK" \
    --codes-bank "$CODES_BANK" \
    --stimuli-dir data/raw/nod/stimuli/ImageNet \
    --num-samples 200 \
    --output-dir outputs/phase18b_rae_spatial_eval \
    --device cuda

echo ""
echo "=== Phase 18B complete! ==="
echo "  Run dir: $RUN_DIR"
echo "  Metrics: outputs/phase18b_rae_spatial_eval/generation_metrics.json"
echo "  Grid:    outputs/phase18b_rae_spatial_eval/generation_grid.png"

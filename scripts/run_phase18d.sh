#!/usr/bin/env bash
# Phase 18D: EEG → spatial_768x4x4 code, probe-on, warm-started from Phase 16c with --loss spatial_cosine_norm
#
# Run on RunPod: bash scripts/run_phase18d.sh

set -e
cd /workspace/mindeye
source venv/bin/activate
export PYTHONPATH=src
export HF_HOME=/workspace/hf_cache
export TMPDIR=/workspace/tmp
export PYTHONUNBUFFERED=1

METADATA="data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv,data/processed/semantic_epochs/zuna_tight1s_sub02_runs01_40/all_runs_metadata.csv,data/processed/semantic_epochs/zuna_tight1s_sub03_runs01_40/all_runs_metadata.csv,data/processed/semantic_epochs/zuna_tight1s_sub04_runs01_40/all_runs_metadata.csv"
EPOCHS_DIR="data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40,data/processed/semantic_epochs/zuna_tight1s_sub02_runs01_40,data/processed/semantic_epochs/zuna_tight1s_sub03_runs01_40,data/processed/semantic_epochs/zuna_tight1s_sub04_runs01_40"

RAE_BANK="data/processed/rae_embeddings/rae_dinov2_base_sub01_04_runs01_40.pt"
CODES_BANK_4x4="data/processed/rae_embeddings/rae_bottleneck_codes_4x4.pt"
BOTTLENECK_CKPT_4x4="outputs/rae_bottleneck/spatial_768x4x4/best.pt"
PROBE_DIR="outputs/rae_code_probe_4x4"
SLUG="phase18d_eeg_to_rae_4x4_loss_cleanup"

# Phase 16c checkpoint — temporal_attn_small + subject FiLM adapters (decode_unit target)
INIT_FROM="outputs/runs/20260527_083218_zuna_real_16c_film_heads/best.pt"

# VLM attributes — prefer runs01_40 version
if [ -f "outputs/common_probe/vlm_attributes_runs01_40.json" ]; then
    VLM_ATTRIBUTES="outputs/common_probe/vlm_attributes_runs01_40.json"
elif [ -f "data/processed/vlm_attributes_runs01_40.json" ]; then
    VLM_ATTRIBUTES="data/processed/vlm_attributes_runs01_40.json"
else
    VLM_ATTRIBUTES=$(find data outputs -name 'vlm_attributes*.json' | head -1)
fi
echo "VLM attributes: $VLM_ATTRIBUTES"
echo "Init from: $INIT_FROM"

# ============================================================
# Step 1: Extract 4×4 bottleneck codes (skip if already exists)
# ============================================================
if [ -f "$CODES_BANK_4x4" ]; then
    echo ""
    echo "=== Step 1: 4×4 codes already exist — skipping extraction ==="
else
    echo ""
    echo "=== Step 1: Extracting 4×4 bottleneck codes ==="
    python3 scripts/build_rae_bottleneck_codes.py \
        --rae-bank "$RAE_BANK" \
        --checkpoint "$BOTTLENECK_CKPT_4x4" \
        --output "$CODES_BANK_4x4" \
        --batch-size 128 \
        --device cuda
fi

# ============================================================
# Step 2: Train probe (skip if already exists)
# ============================================================
if [ -f "$PROBE_DIR/common_probe.pt" ]; then
    echo ""
    echo "=== Step 2: Probe already trained — skipping ==="
else
    echo ""
    echo "=== Step 2: Training RAE-code probe on 4×4 pooled codes ==="
    python3 scripts/pretrain_common_probe.py \
        --metadata "$METADATA" \
        --common-embeddings "$CODES_BANK_4x4" \
        --vlm-attributes "$VLM_ATTRIBUTES" \
        --target-key rae_code \
        --spatial-pool \
        --output-dir "$PROBE_DIR" \
        --epochs 30 \
        --batch-size 128 \
        --lr 1e-4 \
        --device cuda
fi

# ============================================================
# Step 3: Train EEG — warm-started from Phase 16c with spatial_cosine_norm
# ============================================================
echo ""
echo "=== Step 3: Training EEG encoder — Phase 18D (warm-start from 16c, spatial_cosine_norm loss) ==="
python3 scripts/train_eeg_clip.py \
    --metadata "$METADATA" \
    --epochs-dir "$EPOCHS_DIR" \
    --common-embeddings "$CODES_BANK_4x4" \
    --target-space rae_code \
    --target-key image_id_to_rae_code \
    --window-mode tight1s \
    --augment-eeg \
    --no-target-centering \
    --model temporal_attn_small \
    --epochs 40 \
    --patience 8 \
    --batch-size 128 \
    --loss spatial_cosine_norm \
    --probe-weight 0.01 \
    --probe-start-epoch 5 \
    --common-probe "$PROBE_DIR/common_probe.pt" \
    --vlm-attributes "$VLM_ATTRIBUTES" \
    --init-from "$INIT_FROM" \
    --init-skip-heads \
    --slug "$SLUG" \
    --output-dir outputs \
    --device cuda

echo "Training complete."

# ============================================================
# Step 4: Resolve run dir and evaluate
# ============================================================
RUN_DIR=$(find outputs -maxdepth 1 -name "*${SLUG}" -type d | sort | tail -n 1)
echo "Resolved run directory: $RUN_DIR"

if [ -z "$RUN_DIR" ]; then
    echo "ERROR: Could not find run directory for slug '$SLUG'"
    exit 1
fi

echo ""
echo "=== Step 4: Evaluating Phase 18D ==="
python3 scripts/evaluate_rae_spatial.py \
    --run-dir "$RUN_DIR" \
    --bottleneck-checkpoint "$BOTTLENECK_CKPT_4x4" \
    --rae-bank "$RAE_BANK" \
    --codes-bank "$CODES_BANK_4x4" \
    --stimuli-dir data/raw/nod/stimuli/ImageNet \
    --num-samples 200 \
    --output-dir outputs/phase18d_rae_spatial_eval \
    --device cuda

echo ""
echo "=== Phase 18D complete! ==="
echo "  Run dir: $RUN_DIR"
echo "  Metrics: outputs/phase18d_rae_spatial_eval/generation_metrics.json"
echo "  Grid:    outputs/phase18d_rae_spatial_eval/generation_grid.png"

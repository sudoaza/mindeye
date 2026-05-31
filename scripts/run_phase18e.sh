#!/usr/bin/env bash
# Phase 18E: Expander-aligned RAE code training with z-scored code prediction
#
# loss = cosine(expand(pred_code), expand(target_code)) + 0.1 * SmoothL1(z_pred, target_z) + 0.01 * probe
# checkpoint: val_expanded_to_bottleneck_cosine
# final gate: paired bootstrap EEG − shuffled (expanded-to-full-token) in evaluate_rae_spatial.py
#
# Run on RunPod: bash scripts/run_phase18e.sh

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
CODE_STATS_4x4="data/processed/rae_embeddings/rae_code_stats_4x4.pt"
BOTTLENECK_CKPT_4x4="outputs/rae_bottleneck/spatial_768x4x4/best.pt"
PROBE_DIR="outputs/rae_code_probe_4x4"
SLUG="phase18e_expander_aligned_eeg_to_rae_4x4"

INIT_FROM="outputs/runs/20260527_083218_zuna_real_16c_film_heads/best.pt"

if [ -f "outputs/common_probe/vlm_attributes_runs01_40.json" ]; then
    VLM_ATTRIBUTES="outputs/common_probe/vlm_attributes_runs01_40.json"
elif [ -f "data/processed/vlm_attributes_runs01_40.json" ]; then
    VLM_ATTRIBUTES="data/processed/vlm_attributes_runs01_40.json"
else
    VLM_ATTRIBUTES=$(find data outputs -name 'vlm_attributes*.json' | head -1)
fi
echo "VLM attributes: $VLM_ATTRIBUTES"
echo "Init from: $INIT_FROM"

# Step 1: bottleneck codes
if [ -f "$CODES_BANK_4x4" ]; then
    echo "=== Step 1: 4×4 codes exist — skip ==="
else
    echo "=== Step 1: Extract 4×4 bottleneck codes ==="
    python3 scripts/build_rae_bottleneck_codes.py \
        --rae-bank "$RAE_BANK" \
        --checkpoint "$BOTTLENECK_CKPT_4x4" \
        --output "$CODES_BANK_4x4" \
        --batch-size 128 \
        --device cuda
fi

# Step 2: train-split code mean/std for z-scoring
if [ -f "$CODE_STATS_4x4" ]; then
    echo "=== Step 2: code stats exist — skip ==="
else
    echo "=== Step 2: Build train-split code_mean / code_std ==="
    python3 scripts/build_rae_code_stats.py \
        --codes-bank "$CODES_BANK_4x4" \
        --metadata "$METADATA" \
        --epochs-dir "$EPOCHS_DIR" \
        --output "$CODE_STATS_4x4" \
        --val-fraction 0.15 \
        --seed 13
fi

# Step 3: probe
if [ -f "$PROBE_DIR/common_probe.pt" ]; then
    echo "=== Step 3: probe exists — skip ==="
else
    echo "=== Step 3: Train RAE-code probe ==="
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

# Step 4: train EEG (batch 64 — expander forward is memory-heavy)
echo ""
echo "=== Step 4: Train EEG — Phase 18E (expander_aligned + z-score) ==="
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
    --batch-size 64 \
    --loss expander_aligned \
    --bottleneck-checkpoint "$BOTTLENECK_CKPT_4x4" \
    --code-stats "$CODE_STATS_4x4" \
    --rae-bank "$RAE_BANK" \
    --loss-z-weight 0.1 \
    --loss-var-weight 0.0 \
    --probe-weight 0.01 \
    --probe-start-epoch 5 \
    --common-probe "$PROBE_DIR/common_probe.pt" \
    --vlm-attributes "$VLM_ATTRIBUTES" \
    --init-from "$INIT_FROM" \
    --init-skip-heads \
    --slug "$SLUG" \
    --output-dir outputs \
    --device cuda

RUN_DIR=$(find outputs -maxdepth 1 -name "*${SLUG}" -type d | sort | tail -n 1)
echo "Resolved run directory: $RUN_DIR"
if [ -z "$RUN_DIR" ]; then
    echo "ERROR: Could not find run directory for slug '$SLUG'"
    exit 1
fi

echo ""
echo "=== Step 5: Evaluate Phase 18E (paired bootstrap gate) ==="
python3 scripts/evaluate_rae_spatial.py \
    --run-dir "$RUN_DIR" \
    --bottleneck-checkpoint "$BOTTLENECK_CKPT_4x4" \
    --rae-bank "$RAE_BANK" \
    --codes-bank "$CODES_BANK_4x4" \
    --stimuli-dir data/raw/nod/stimuli/ImageNet \
    --num-samples 200 \
    --output-dir outputs/phase18e_rae_spatial_eval \
    --device cuda

echo ""
echo "=== Phase 18E complete! ==="
echo "  Run dir: $RUN_DIR"
echo "  Metrics: outputs/phase18e_rae_spatial_eval/generation_metrics.json"
echo "  Grid:    outputs/phase18e_rae_spatial_eval/generation_grid.png"

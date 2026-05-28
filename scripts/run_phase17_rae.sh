#!/bin/bash
# Phase 17: RAE target/decoder swap on top of current best EEG architecture (16c_film_heads)

source venv/bin/activate

# Use multi-subject tight1s epochs
METADATA="data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv,data/processed/semantic_epochs/zuna_tight1s_sub02_runs01_40/all_runs_metadata.csv,data/processed/semantic_epochs/zuna_tight1s_sub03_runs01_40/all_runs_metadata.csv,data/processed/semantic_epochs/zuna_tight1s_sub04_runs01_40/all_runs_metadata.csv"
EPOCHS_DIR="data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40,data/processed/semantic_epochs/zuna_tight1s_sub02_runs01_40,data/processed/semantic_epochs/zuna_tight1s_sub03_runs01_40,data/processed/semantic_epochs/zuna_tight1s_sub04_runs01_40"

COMMON_EMB="data/processed/rae_embeddings/rae_dinov2_base_sub01_04_runs01_40.pt"

SLUG="phase17_6_rae_centered_16c_init_augmented"

# Architecture: 16c_film_heads with Phase 16 initialization and augment_eeg active
python3 scripts/train_eeg_clip.py \
    --metadata "$METADATA" \
    --epochs-dir "$EPOCHS_DIR" \
    --common-embeddings "$COMMON_EMB" \
    --target-space rae_centered_unit \
    --target-key image_id_to_rae_centered_unit \
    --window-mode tight1s \
    --augment-eeg \
    --model temporal_attn_small \
    --epochs 50 \
    --patience 15 \
    --batch-size 128 \
    --loss contrastive \
    --temperature 0.07 \
    --slug "$SLUG" \
    --output-dir outputs \
    --vlm-attributes data/processed/clip_embeddings/vlm_attributes.json \
    --common-probe outputs/rae_probe/common_probe.pt \
    --probe-weight 0.01 \
    --probe-start-epoch 5 \
    --head-reg-weight 0.01 \
    --init-from outputs/runs/20260527_083218_zuna_real_16c_film_heads/best.pt \
    --init-skip-heads \
    --lr 1e-4 \
    --device cuda

echo "Training complete. Extracting test metrics..."

# Dynamically resolve the newly created run directory
RUN_DIR=$(find outputs -maxdepth 1 -name "*_${SLUG}" -type d | sort | tail -n 1)
echo "Resolved training run directory: $RUN_DIR"

# Run evaluation script with RAE-native metric computation and retrieval
python3 scripts/evaluate_rae_generation.py \
    --run-dir "$RUN_DIR" \
    --num-samples 500 \
    --batch-size 25 \
    --k 5 \
    --target-key image_id_to_rae_centered_unit \
    --temperature 0.05 \
    --common-probe outputs/rae_probe/common_probe.pt \
    --stimuli-dir data/raw/nod/stimuli/ImageNet \
    --output-dir outputs/phase17_6_rae_eval

echo "Phase 17.6 complete!"

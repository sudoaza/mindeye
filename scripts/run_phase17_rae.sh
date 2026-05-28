#!/bin/bash
# Phase 17: RAE target/decoder swap on top of current best EEG architecture (16c_film_heads)

source venv/bin/activate

# Use multi-subject tight1s epochs
METADATA="data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv,data/processed/semantic_epochs/zuna_tight1s_sub02_runs01_40/all_runs_metadata.csv,data/processed/semantic_epochs/zuna_tight1s_sub03_runs01_40/all_runs_metadata.csv,data/processed/semantic_epochs/zuna_tight1s_sub04_runs01_40/all_runs_metadata.csv"
EPOCHS_DIR="data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40,data/processed/semantic_epochs/zuna_tight1s_sub02_runs01_40,data/processed/semantic_epochs/zuna_tight1s_sub03_runs01_40,data/processed/semantic_epochs/zuna_tight1s_sub04_runs01_40"

COMMON_EMB="data/processed/rae_embeddings/rae_dinov2_base_sub01_04_runs01_40.pt"

SLUG="phase17_rae_centered"

# Architecture: 16c_film_heads (temporal_attn_small, dual-head false, film true)
python3 scripts/train_eeg_clip.py \
    --metadata "$METADATA" \
    --epochs-dir "$EPOCHS_DIR" \
    --common-embeddings "$COMMON_EMB" \
    --target-space rae_unit \
    --target-key image_id_to_rae_centered_unit \
    --window-mode tight1s \
    --add-event-marker \
    --model temporal_attn_small \
    --hidden-dim 256 \
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
    --head-reg-weight 0.01

echo "Training complete. Extracting test metrics..."

# Run evaluation script with RAE-native metric computation and retrieval
python3 scripts/evaluate_rae_generation.py \
    --run-dir outputs/$SLUG \
    --num-samples 100 \
    --batch-size 25 \
    --k 5 \
    --target-key image_id_to_rae_centered_unit \
    --temperature 0.05 \
    --common-probe outputs/rae_probe/common_probe.pt \
    --stimuli-dir data/raw/nod/stimuli/ImageNet \
    --output-dir outputs/${SLUG}_eval

echo "Phase 17 complete!"

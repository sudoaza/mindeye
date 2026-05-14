#!/bin/bash
# Comprehensive Recovery Execution Script
set -e

source venv/bin/activate

REQUESTED_RUNS=40
AVAILABLE_RUNS=$(python - <<'PY_AVAIL'
import pandas as pd
from pathlib import Path
p = Path("data/raw/nod/derivatives/detailed_events/sub-01_events.csv")
if not p.exists():
    print(40)
else:
    df = pd.read_csv(p)
    pairs = df[["session", "run"]].drop_duplicates().sort_values(["session", "run"])
    print(len(pairs))
PY_AVAIL
)
# The current sub-01 detailed-events file has 32 ImageNet session-runs; still
# request 40 from OpenNeuro so a future ImageNet05 appears automatically, but
# train/validate only on runs with event metadata.
if [ "$AVAILABLE_RUNS" -lt "$REQUESTED_RUNS" ]; then
  echo "[WARN] Requested $REQUESTED_RUNS global runs, but events metadata exposes only $AVAILABLE_RUNS for sub-01."
  echo "[WARN] Will attempt download for 1-$REQUESTED_RUNS, then crop/train/evaluate 1-$AVAILABLE_RUNS."
fi
RUN_ARGS=$(seq 1 "$AVAILABLE_RUNS")
VAL_RUN="$AVAILABLE_RUNS"


echo "=== [1/7] Downloading NOD global runs 1-$REQUESTED_RUNS ==="
python scripts/download_nod.py --subject sub-01 --runs 1-$REQUESTED_RUNS

echo "=== [2/7] Syncing Stimuli from S3 ==="
python scripts/generate_clip_embeddings.py \
    --metadata data/raw/nod/derivatives/detailed_events/sub-01_events.csv \
    --write-openneuro-include-list data/raw/nod/stimuli_include.txt

python scripts/sync_stimuli_s3_targeted.py

echo "=== [3/7] Running ZUNA Batch Pipeline (15 steps) ==="
python scripts/run_zuna_batch.py --diffusion-steps 15

echo "=== [4/7] Cropping Full 5s Back-aligned Windows ==="
python scripts/run_cropper.py \
  --mode zuna \
  --tmin -0.2 --tmax 1.0 \
  --add-event-marker \
  --runs $RUN_ARGS \
  --zuna-dir data/processed/zuna_real/4_fif_output \
  --output-dir data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40

echo "=== [5/7] Generating Embeddings ==="
python scripts/generate_clip_embeddings.py \
    --metadata data/raw/nod/derivatives/detailed_events/sub-01_events.csv \
    --output data/processed/clip_embeddings/sub01_image_embeddings.pt

python scripts/generate_image_semantics.py \
  --metadata data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv \
  --image-root data/raw/nod/stimuli/ImageNet \
  --output data/processed/clip_embeddings/image_semantics.jsonl

python scripts/generate_text_embeddings.py \
  --source image_semantics \
  --semantics-jsonl data/processed/clip_embeddings/image_semantics.jsonl \
  --output data/processed/clip_embeddings/image_semantic_text_embeddings.pt

python scripts/generate_text_embeddings.py \
  --source templates \
  --metadata data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv \
  --output data/processed/clip_embeddings/imagenet_text_embeddings.pt

python scripts/build_common_embeddings.py \
  --image-embeddings data/processed/clip_embeddings/sub01_image_embeddings.pt \
  --semantic-embeddings data/processed/clip_embeddings/image_semantic_text_embeddings.pt \
  --label-embeddings data/processed/clip_embeddings/imagenet_text_embeddings.pt \
  --metadata data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv \
  --w-img 0.25 --w-sem 0.75 \
  --output data/processed/clip_embeddings/common_embeddings.pt

echo "=== [6/7] Smoke Test (2 epochs) ==="
python scripts/train_eeg_clip.py \
  --metadata data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv \
  --epochs-dir data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40 \
  --common-embeddings data/processed/clip_embeddings/common_embeddings.pt \
  --input-domain zuna \
  --window-mode tight1s \
  --target-space common \
  --target-mode real \
  --model temporal_attn_small \
  --val-runs "$VAL_RUN" \
  --epochs 2 \
  --batch-size 16 \
  --device cuda \
  --slug smoke_tight1s \
  --add-event-marker \
  --augment-eeg

echo "=== [7/7] Launching Full Recovery Matrix ==="
python scripts/run_baseline_matrix.py \
  --metadata data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv \
  --epochs-dir data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40 \
  --common-embeddings data/processed/clip_embeddings/common_embeddings.pt \
  --val-runs "$VAL_RUN" \
  --window-mode tight1s \
  --target-space common \
  --model temporal_attn_small \
  --epochs 50 \
  --batch-size 64 \
  --device cuda \
  --slug tight1s_recovery \
  --add-event-marker \
  --augment-eeg \
  --conditions zuna_real zuna_shuffled zuna_random

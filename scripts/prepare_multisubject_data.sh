#!/usr/bin/env bash
# ==============================================================================
# MindEye multi-subject cohort pipeline (ZUNA -> QFormer -> RAE)
# ==============================================================================
# Full-cohort run: sub-01..sub-09, each with 32 ImageNet runs (4 sessions x 8).
# NOD (ds005811) only has 32-run coverage for sub-01..sub-09; sub-10+ have 16.
#
# Runs end-to-end: download -> ZUNA denoise -> per-subject crop -> stimulus sync
# -> single RAE bank -> merged latent cache -> QFormer grid (combined cohort).
#
# Env: run on a RunPod pod with deps already installed (system Python). This
# script does NOT create a venv. Override the cohort/paths via the vars below.
# ==============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."          # repo root
export PYTHONPATH=src
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/triton_cache}"
mkdir -p "$TRITON_CACHE_DIR"

# --- Cohort configuration (override via env) ---
SUBJECTS="${SUBJECTS:-sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08 sub-09}"
RUNS_SPEC="${RUNS_SPEC:-1-32}"                 # download/split spec (global run ids)
RUNS_SEQ="${RUNS_SEQ:-$(seq 1 32)}"            # cropper expects space-separated ints
RUN_TAG="${RUN_TAG:-runs01_32}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-15}"
DEVICE="${DEVICE:-cuda}"
# ZUNA packs a whole batch into one flex-attention document -> dense (B*2480)^2 mask,
# so caching cost is O(B^2). B=4 fits an 80GB A100; B=32 OOMs (187GiB). Keep small.
ZUNA_CACHE_BATCH="${ZUNA_CACHE_BATCH:-4}"
RAE_BANK="${RAE_BANK:-data/processed/rae_embeddings/rae_dinov2_base_all.pt}"
MERGED_LATENTS="${MERGED_LATENTS:-data/processed/zuna_latents/cohort9_${RUN_TAG}}"
OUT_DIR="${OUT_DIR:-outputs/qformer_cohort9_grid}"
TRAIN_RUNS="${TRAIN_RUNS:-1-24}"
VAL_RUNS="${VAL_RUNS:-25-28}"
TEST_RUNS="${TEST_RUNS:-29-32}"

# Count subjects for FiLM.
NUM_SUBJECTS="$(echo "$SUBJECTS" | wc -w)"

echo "=== Cohort: [$SUBJECTS] | runs $RUNS_SPEC | num_subjects=$NUM_SUBJECTS ==="

mkdir -p data/raw/nod data/processed/rae_embeddings data/processed/zuna_latents "$OUT_DIR"

echo "=== [1/7] Downloading raw NOD-EEG (ds005811) ==="
for sub in $SUBJECTS; do
  echo ">>> Downloading $sub (runs $RUNS_SPEC)..."
  python scripts/download_nod.py --subject "$sub" --runs "$RUNS_SPEC"
done

echo "=== [2/7] ZUNA diffusion denoising (all subjects, single pass) ==="
python scripts/run_zuna_batch.py --diffusion-steps "$DIFFUSION_STEPS" --gpu-device 0

echo "=== [3/7] Cropping semantic epochs (per subject) ==="
EPOCHS_DIRS=()
for sub in $SUBJECTS; do
  sub_clean=${sub//-/}
  out="data/processed/semantic_epochs/zuna_full5s_backaligned_${sub_clean}_${RUN_TAG}"
  echo ">>> Cropping $sub -> $out"
  python scripts/run_cropper.py --mode zuna --full5s-backaligned --add-event-marker \
    --runs $RUNS_SEQ \
    --subject "$sub" \
    --zuna-dir data/processed/zuna_real/4_fif_output \
    --output-dir "$out"
  EPOCHS_DIRS+=("$out")
done

echo "=== [4/7] Building stimulus include-list (union) and syncing images ==="
: > data/raw/nod/stimuli_include_all.txt
for dir in "${EPOCHS_DIRS[@]}"; do
  python scripts/generate_clip_embeddings.py \
    --metadata "$dir/all_runs_metadata.csv" \
    --write-openneuro-include-list "$dir/stimuli_include.txt"
  cat "$dir/stimuli_include.txt" >> data/raw/nod/stimuli_include_all.txt
done
sort -u data/raw/nod/stimuli_include_all.txt > data/raw/nod/stimuli_include.txt
python scripts/sync_stimuli_s3_targeted.py

echo "=== [5/7] Building single RAE/DINOv2 target bank (keyed by image_id) ==="
python scripts/build_rae_latent_bank.py \
  --image-dir data/raw/nod/stimuli/ImageNet \
  --output "$RAE_BANK"

echo "=== [6/7] Caching ZUNA latents into ONE merged cohort dir ==="
python scripts/cache_zuna_latents.py \
  --epochs-dir "${EPOCHS_DIRS[@]}" \
  --output-dir "$MERGED_LATENTS" \
  --batch-size "$ZUNA_CACHE_BATCH" \
  --layers post_mmd --device "$DEVICE"

echo "=== [7/7] QFormer grid over the combined cohort (real/shuffled/random) ==="
python scripts/run_qformer_grid.py \
  --latents-pt "$MERGED_LATENTS" \
  --rae-pt "$RAE_BANK" \
  --num-subjects "$NUM_SUBJECTS" \
  --train-runs "$TRAIN_RUNS" --val-runs "$VAL_RUNS" --test-runs "$TEST_RUNS" \
  --epochs 40 --patience 8 --batch-size 64 --lr 3e-4 \
  --device "$DEVICE" --out-dir "$OUT_DIR"

echo "=== Multi-subject cohort pipeline completed successfully! ==="

#!/bin/bash
set -e

# Make sure we are in the correct directory and using the virtual environment
cd /workspace/mindeye
source venv/bin/activate
export PYTHONPATH=src

# Use persistent Triton compilation cache to bypass autotune overhead
export TRITON_CACHE_DIR=/workspace/triton_cache
mkdir -p $TRITON_CACHE_DIR

echo "=== [1/3] Downloading sub-02, sub-03, sub-04 (Runs 1-32) ==="
for sub in sub-02 sub-03 sub-04; do
  echo ">>> Downloading $sub..."
  python scripts/download_nod.py --subject $sub --runs 1-32
done

echo "=== [2/3] Running ZUNA Batch Denoising Pipeline (15 steps) ==="
# Denoises new files and automatically skips already processed sub-01 runs
python scripts/run_zuna_batch.py --diffusion-steps 15

echo "=== [3/3] Cropping tight1s semantic epochs ==="
for sub in sub-02 sub-03 sub-04; do
  # Strip hyphen from subject name (sub-02 -> sub02) to match config paths
  sub_clean=${sub//-/}
  echo ">>> Cropping $sub -> $sub_clean..."
  python scripts/run_cropper.py --mode zuna --tmin -0.2 --tmax 1.0 --add-event-marker \
    --runs $(seq 1 32) \
    --subject $sub \
    --zuna-dir data/processed/zuna_real/4_fif_output \
    --output-dir data/processed/semantic_epochs/zuna_tight1s_${sub_clean}_runs01_40
done

echo "=== Data Preparation Completed Successfully ==="

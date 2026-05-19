#!/bin/bash
set -e

# Use the number of available runs (we know from previous steps it's 32 for sub-01)
RUNS=$(seq 1 32)
VAL_RUN=32

echo "=== [1/6] Simulating EPOC-14 Low-Channel Recordings ==="
export PYTHONPATH=src
python scripts/simulate_low_channel_zuna.py --runs $RUNS

echo "=== [2/6] Running ZUNA Upscaling on Simulated Recordings ==="
# Offline pipeline uses configs/zuna/zuna_epoc14_sim.yaml
python src/mindseye/zuna/offline_pipeline.py configs/zuna/zuna_epoc14_sim.yaml

echo "=== [3/6] Cropping EPOC-14 ZUNA-Upscaled Epochs ==="
python scripts/run_cropper.py \
  --mode zuna \
  --raw-dir data/processed/simulated_epoc14 \
  --zuna-dir data/processed/zuna_epoc14_sim/4_fif_output \
  --output-dir data/processed/semantic_epochs/zuna_epoc14_tight1s_sub01_runs01_32 \
  --tmin -0.2 --tmax 1.0 \
  --add-event-marker \
  --runs $RUNS

echo "=== [4/6] Cropping Raw EPOC-14 Epochs (No ZUNA) ==="
python scripts/run_cropper.py \
  --mode raw \
  --raw-dir data/processed/simulated_epoc14 \
  --output-dir data/processed/semantic_epochs/raw_epoc14_tight1s_sub01_runs01_32 \
  --tmin -0.2 --tmax 1.0 \
  --add-event-marker \
  --runs $RUNS

echo "=== [5/6] Training Baseline Matrix on EPOC-14 ZUNA vs Raw ==="
nohup python scripts/run_baseline_matrix.py \
  --metadata data/processed/semantic_epochs/zuna_epoc14_tight1s_sub01_runs01_32/all_runs_metadata.csv \
  --epochs-dir data/processed/semantic_epochs/zuna_epoc14_tight1s_sub01_runs01_32 \
  --epochs-dir-raw data/processed/semantic_epochs/raw_epoc14_tight1s_sub01_runs01_32 \
  --common-embeddings data/processed/clip_embeddings/common_embeddings.pt \
  --val-runs $VAL_RUN \
  --window-mode tight1s \
  --target-space common \
  --model temporal_attn_small \
  --epochs 50 \
  --batch-size 64 \
  --device cuda \
  --slug epoc14_sim \
  --add-event-marker \
  --augment-eeg \
  --conditions zuna_real zuna_shuffled raw_real > matrix_epoc14.log 2>&1 &

echo "=== [6/6] Matrix Training Started! ==="
echo "You can tail the logs with: ssh -p 33396 -i ~/.ssh/id_ed25519 root@157.157.221.29 \"tail -f /workspace/mindeye/matrix_epoc14.log\""

#!/bin/bash
set -e
echo "Killing previous processes..."
pkill -f 'python.*simulate_low_channel_zuna.py' || true
pkill -f 'python.*offline_pipeline.py' || true
pkill -f 'python.*run_cropper.py' || true
pkill -f 'python.*run_baseline_matrix.py' || true

echo "Cleaning up bad processed data..."
rm -rf /workspace/mindeye/data/processed/simulated_epoc14
rm -rf /workspace/mindeye/data/processed/zuna_epoc14_sim/4_fif_output
rm -rf /workspace/mindeye/data/processed/semantic_epochs/zuna_epoc14_tight1s_sub01_runs01_32
rm -rf /workspace/mindeye/data/processed/semantic_epochs/raw_epoc14_tight1s_sub01_runs01_32

echo "Starting fresh simulation..."
rm -f /workspace/mindeye/epoc14.log
nohup bash run_epoc_simulation.sh > epoc14.log 2>&1 &
echo "Started."

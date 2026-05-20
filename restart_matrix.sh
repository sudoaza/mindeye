#!/bin/bash
ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_ed25519 -p 33396 root@157.157.221.29 '
cd /workspace/mindeye
git init
git remote add origin git@github.com:sudoaza/mindeye.git
git config user.email "agent@example.com"
git config user.name "Agent"
git add .
git commit -m "sync from local" || true

# Kill the currently running baseline matrix script
pkill -f "python scripts/run_baseline_matrix.py" || true

# Start it with the correct val-runs value
export PYTHONPATH=src
nohup python scripts/run_baseline_matrix.py \
  --metadata data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv \
  --epochs-dir data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40 \
  --common-embeddings data/processed/clip_embeddings/common_embeddings.pt \
  --val-runs 32 \
  --window-mode tight1s \
  --target-space common \
  --model temporal_attn_small \
  --epochs 50 \
  --batch-size 64 \
  --device cuda \
  --slug tight1s_recovery \
  --add-event-marker \
  --augment-eeg \
  --conditions zuna_real zuna_shuffled zuna_random > matrix_v2.log 2>&1 &

echo "Restarted matrix! You can tail the logs with: ssh -p 33396 -i ~/.ssh/id_ed25519 root@157.157.221.29 \"tail -f /workspace/mindeye/matrix_v2.log\""
'

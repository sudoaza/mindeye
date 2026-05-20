#!/bin/bash
set -e

POD_IP="157.157.221.29"
POD_PORT="33396"
KEY="~/.ssh/id_ed25519"

echo "Syncing code to pod..."
rsync -avz -e "ssh -i $KEY -p $POD_PORT -o StrictHostKeyChecking=no" \
  --exclude '.git' --exclude 'data' --exclude 'outputs' --exclude 'venv' --exclude '__pycache__' \
  ./ root@$POD_IP:/workspace/mindeye/

echo "Starting EPOC-14 simulation on pod..."
ssh -o StrictHostKeyChecking=no -i $KEY -p $POD_PORT root@$POD_IP "cd /workspace/mindeye && git add . && git commit -m 'start epoc14 sim' || true && nohup bash run_epoc_simulation.sh > epoc14.log 2>&1 &"

echo "Done! You can monitor the progress on the pod with:"
echo "ssh -p $POD_PORT -i $KEY root@$POD_IP 'tail -f /workspace/mindeye/epoc14.log'"

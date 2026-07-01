---
name: mindeye-pod
description: Run MindEye GPU work on a RunPod pod via the runpod MCP with a persistent network volume. Use when the user wants to provision/start/stop a pod, attach or move the network volume between pods, sync MindEye code to the pod, or launch/monitor a training or evaluation run (e.g. the QFormer grid). Covers the full lifecycle: pod up → code sync → run → pull outputs → pod down.
disable-model-invocation: true
---

# MindEye Pod Workflow

All MindEye GPU work runs on a **RunPod** pod (managed via the `runpod` MCP); the local machine
has no GPU and no data. **Remote-only: never run training/inference/eval locally** — always spin up
or start a pod and run over SSH. Authoritative infra doc: `docs/INFRA.md`. Architecture/plan:
`docs/PLAN.md`, `docs/HANDOVER.md`.

## Lifecycle checklist

```
- [ ] 1. Find or provision the pod (runpod MCP)
- [ ] 2. SSH in; pull code + install deps
- [ ] 3. Run the job in background (nohup) with PYTHONPATH=src
- [ ] 4. Monitor the log / gate metrics
- [ ] 5. Pull outputs (or confirm they're on the volume)
- [ ] 6. Stop the pod to save cost
```

## 1. Pod up

```
mcp_runpod_list-pods                                  # find an existing pod
mcp_runpod_start-pod  {"podId": "<POD_ID>"}           # if EXITED
mcp_runpod_get-pod    {"podId": "<POD_ID>", "includeMachine": true}   # get IP + SSH port
```

To provision a new pod, use the JSON in `docs/INFRA.md` §3 (`volumeInGb: 200`, `containerDiskInGb: 100`,
`PUBLIC_KEY` = contents of `~/.ssh/id_ed25519.pub`).

> **Network volumes are datacenter-local.** Create the pod in the same DC as the volume, or it
> cannot mount at `/workspace`. See `docs/INFRA.md` §2 for the detach/reattach procedure.

## 2. Code + deps (on pod)

```bash
ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no \
  "cd /workspace/mindeye && git pull origin master && pip install -r requirements.txt"
```

Prefer `git pull` over rsync. **If you must rsync, pull `outputs/` first** — local copies silently
overwrite pod checkpoints (`docs/CHEAT.md`).

Always set before any script (system Python, no venv):
```bash
export PYTHONPATH=/workspace/mindeye/src HF_HOME=/workspace/hf_cache TMPDIR=/workspace/tmp
```

## 3. Run (background, survives SSH drop)

Launch the current pipeline (QFormer grid) via `nohup` so it survives disconnection:
```bash
ssh root@<IP> -p <PORT> \
  "cd /workspace/mindeye && export PYTHONPATH=src && nohup python scripts/run_qformer_grid.py \
     --latents-pt data/processed/zuna_latents/sub01_runs01_32 \
     --clip-pt data/processed/clip_embeddings/common_embeddings.pt \
     --rae-pt data/processed/rae_embeddings/rae_dinov2_base_sub01_04_runs01_40.pt \
     --train-runs 1-24 --val-runs 25-28 --test-runs 29-32 \
     --device cuda --out-dir outputs/qformer_aligned_grid \
     > qformer_grid.log 2>&1 &"
```

## 4. Monitor

```bash
ssh root@<IP> -p <PORT> "tail -n 50 /workspace/mindeye/qformer_grid.log; echo '---'; ps aux | grep -c '[p]ython'"
```
Gather multiple checks in one SSH call — connection overhead is significant. The **gate** is the
paired-bootstrap table: real − shuffled Δ > +0.005, 95% CI excludes 0, `collapse_pct` < 20%.

## 5. Outputs

Outputs land under `outputs/qformer_aligned_grid/grid_<timestamp>/` on the network volume, so they
persist without copying. To inspect locally, pull (never push over them):
```bash
rsync -avz --no-o --no-g --no-perms -e "ssh -p <PORT>" \
  root@<IP>:/workspace/mindeye/outputs/qformer_aligned_grid/ outputs/qformer_aligned_grid/
```

## 6. Pod down

```
mcp_runpod_stop-pod {"podId": "<POD_ID>"}   # preserves volume; stops billing for GPU
```

## Notes
- **Empty/new volume or lost volume?** Nothing to push from the laptop — raw data (OpenNeuro
  `ds005811`) and model weights (`Zyphra/ZUNA`, `nyu-visionx/RAE-...`) are public. Rebuild the whole
  volume with `bash cold_start.sh` (`export PYTHONPATH=src HF_HOME=/workspace/hf_cache` first). See
  `docs/INFRA.md` §4b–4d for model sources, from-scratch bootstrap, and disaster recovery.
- **Only `outputs/` is not trivially reproducible** — pull it to the laptop after any run worth
  keeping, before deleting the volume/pod. There is no automatic volume backup.
- The pod uses **system Python + `PYTHONPATH=src`**, not a venv. A pod-created venv can have
  zero-byte binaries after a container restart — delete it and use system Python.
- Deprecated run paths (`make matrix`, `train_rae_token_bottleneck.py`, `run_phase18*.sh`) are not
  the live plan; see `docs/PLAN.md` §6.

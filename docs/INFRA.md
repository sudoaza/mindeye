# MindEye Infrastructure — RunPod + Network Volume

> All GPU work runs on a **RunPod Secure Cloud** pod, managed via the **runpod MCP**
> (see [`RunPod_SKILL.md`](RunPod_SKILL.md)). The local dev machine has no GPU and no data.
>
> **Remote-only rule.** Never run training, ZUNA inference, embedding builds, or evaluation
> locally — always spin up / start a pod and run there over SSH. There is no local venv for
> pipeline steps.

## 1. Storage model — what lives where

RunPod gives each pod two kinds of storage:

| Storage | Lifetime | Speed | Use for |
|---|---|---|---|
| **Container disk** | Destroyed when the pod is deleted | Fast, local | OS, ephemeral scratch |
| **Network volume** (mounted at `/workspace`) | **Persists across pods**; can be detached from one pod and attached to another | Network-backed | Everything we want to keep |

### Current setup (what we do today)

The **network volume holds the data and big weights**; **code and the Python env are rebuilt on each pod**:

| Path | Storage | Rebuilt per pod? |
|---|---|---|
| `/workspace/mindeye/data/` | Network volume | No — persists |
| `/workspace/mindeye/outputs/` (checkpoints, metrics) | Network volume | No — persists |
| `/workspace/hf_cache/` (HF model weights) | Network volume | No — persists |
| `/workspace/mindeye/` code | Rebuilt | Yes — `git pull` |
| Python packages | Rebuilt | Yes — `pip install -r requirements.txt` (system Python, no venv) |

### Target setup (where we are moving)

**One persistent network volume that holds everything** — data, outputs, HF cache, **and** the code + a stable environment — so a fresh pod only needs the volume attached and `PYTHONPATH` set, with no reinstall. Migration is incremental; until it lands, treat the "current setup" table above as authoritative.

> **Rule of thumb (both setups):** anything expensive to recreate (downloaded EEG, ZUNA outputs, embedding banks, trained checkpoints, HF weights) **must** be on the network volume, never on container disk.

## 2. Moving the volume between pods

The volume is the durable artifact; pods are disposable.

1. **Stop or delete** the old pod (`mcp_runpod_stop-pod` / `mcp_runpod_delete-pod`). Stopping preserves the pod shell; deleting frees it entirely — the volume survives either way.
2. **Create/start** the new pod **in the same datacenter as the volume**, attaching the existing network volume at `/workspace`.
3. On the new pod: `cd /workspace/mindeye && git pull` (SSH git URL, agent-forwarded key) → install non-torch deps (§4) → `export PYTHONPATH=src`. Data/outputs/cache are already present from the volume.

> [!IMPORTANT]
> **Network volumes are datacenter-local.** A pod in `EUR-NO-2` cannot mount a volume created in `EU-RO-1`. Always check the volume's `dataCenterId` and create the pod in the same DC. This is the #1 cause of "my data is gone" — it isn't gone, the pod just can't reach the volume.

> [!CAUTION]
> **Never `rsync` local → remote without pulling `outputs/` first.** Pod-side checkpoints and probe models will be silently overwritten by stale local copies. Prefer `git pull` on the pod over rsync entirely.

## 3. Recommended pod configuration

| Parameter | Value | Why |
|---|---|---|
| **GPU** | RTX 4090 / A100 / H100 | Fits the model stack comfortably; A100/H100 speed iteration |
| **VRAM** | ≥ 16 GB | Room for batching + any decoder stack |
| **CPU RAM** | ≥ 32 GB | Dataloaders and dataset preloading |
| **Container disk** | 60 GB | Ephemeral scratch, pip build cache, torch tmp |
| **Network volume** | 200 GB | Persistent `/workspace`: raw EEG + processed data + outputs + hf_cache. Ample for the full 9-subject cohort. |
| **Image** | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (Runpod PyTorch 2.8) | Ships torch 2.8.0+cu128, ready to use. **Do not force the `torch==2.6.0+cu124` pin from `requirements.txt` onto this image** — install only the non-torch deps (see §4), then **re-pin torch back to 2.8.0** because a transitive dep upgrades it to 2.12.1 and breaks the CUDA stack. |

> Canonical sizes: **containerDiskInGb = 60**, **volumeInGb = 200**. Do **not** request larger disks
> — A100 hosts reject 500–700 GB requests with *"no instances available with enough disk space"*.
> NOD data is tiny (~0.6 GB/subject); the RAE bank (~14 GB) + hf_cache (~20 GB) are the real consumers.
> **GPU choice**: A100 80GB is recommended for full-cohort runs (fast ZUNA batch + grid). A40 works for
> single-subject dev. v2 `create-pod` accepts only **one** GPU type per pod (extra `gpuTypeIds` ignored).

### Provisioning JSON (runpod MCP `create-pod`)

```json
{
  "cloudType": "SECURE",
  "computeType": "GPU",
  "gpuCount": 1,
  "gpuTypeIds": ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"],
  "volumeInGb": 200,
  "containerDiskInGb": 60,
  "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
  "name": "mindeye-cohort9",
  "ports": ["22/tcp"],
  "volumeMountPath": "/workspace",
  "env": {"PUBLIC_KEY": "<contents of ~/.ssh/runpod.pub>"}
}
```

> [!WARNING]
> Pods not provisioned with your `~/.ssh/runpod.pub` as `PUBLIC_KEY` will prompt for a password and
> block SSH. Always pass the ed25519 pub key. SSH to the pod with `-i ~/.ssh/runpod`.

> [!IMPORTANT]
> **The runpod MCP `create-pod` / `update-pod` cannot attach an existing network volume** — there is
> no `networkVolumeId` field in the MCP schema, and MCP-created pods may land in a different
> datacenter than the volume. In practice, **an MCP-created pod gets a fresh, empty volume** and must
> be bootstrapped from scratch (§4c). To reuse a persistent volume (e.g. `mindeye-netvol-100gb` in
> `EU-RO-1`), create the pod in the **RunPod web UI**, selecting that volume, then drive it over SSH.
> Choose your path per run: MCP = fast + disposable; UI = persistent data.

## 4. Fresh-pod setup

Two distinct SSH keys are in play:
- **`~/.ssh/runpod`** — laptop → pod access (the pod's `PUBLIC_KEY`). Use with `ssh -i ~/.ssh/runpod`.
- **`~/.ssh/id_ed25519`** — GitHub auth. **Always use the SSH git URL** (`git@github.com:sudoaza/mindeye.git`).
  Locally this "just works" via `~/.ssh/config`. On the **pod** the GitHub key is not present, so either
  **forward the agent** (`ssh -A`, with the key added via `ssh-add ~/.ssh/id_ed25519`) or copy the key
  to the pod. Agent forwarding is preferred — no private key ever lands on the pod.

```bash
# From the laptop: make the GitHub key available for forwarding, then SSH with -A
ssh-add ~/.ssh/id_ed25519
ssh -A root@<IP> -p <PORT> -i ~/.ssh/runpod -o StrictHostKeyChecking=no

# On the pod: git over SSH needs relaxed host-key checking on a fresh pod — a bare
# `git clone` fails with "Host key verification failed" even after `ssh -T` succeeded.
export GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null'
cd /workspace && git clone git@github.com:sudoaza/mindeye.git mindeye        # fresh volume
# or: cd /workspace/mindeye && git fetch origin && git reset --hard origin/master  # existing volume
# (existing HTTPS remote? switch it: git remote set-url origin git@github.com:sudoaza/mindeye.git)

# Deps: the image already ships torch 2.8.0+cu128. Install only the non-torch deps
# (skipping the torch/cu124 pin lines) with --break-system-packages (Ubuntu 24.04 PEP 668):
cd /workspace/mindeye
grep -vE '^(--extra-index-url|torch==|torchvision==|torchaudio==|nvidia-cudnn|$)' requirements.txt > /tmp/reqs_notorch.txt
pip install --break-system-packages -r /tmp/reqs_notorch.txt

# ⚠️ A transitive dep pulls torch 2.12.1 and breaks torchvision/torchaudio + cu128. Re-pin:
pip install --break-system-packages torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128
python3 -c "import torch,torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"

# Always set before any script:
export PYTHONPATH=/workspace/mindeye/src
export HF_HOME=/workspace/hf_cache
export TMPDIR=/workspace/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

> The pod uses **system Python with `PYTHONPATH=src`** — no venv. A venv created on the pod can end up with zero-byte binaries after a container restart; if you hit that, delete it and use system Python.

### Driving the pod without SSH-approval prompts — `pod-exec` MCP

A small custom MCP server (`~/.cursor/mcp-servers/pod-exec/server.py`, registered in `~/.cursor/mcp.json` as `pod-exec`, tools namespaced `user-pod-exec`) execs arbitrary commands on the pod over SSH, so the agent runs pod commands without a local approval prompt each time. Update the pod host/port in `~/.cursor/mcp-servers/pod-exec/pod.json` (or via the `pod_config` tool) when the pod changes. Background long runs on the pod with `nohup ... &`; they survive Cursor/backend disconnects. See `docs/HANDOVER.md` §2.

## 4b. Model weights (downloaded, not stored in git)

All model weights are pulled from their sources on first use and cached under `HF_HOME`
(`/workspace/hf_cache`, on the volume) so they download **once per volume**, not once per pod.

| Model | Source | How it's fetched | Cache |
|---|---|---|---|
| **ZUNA** (EEG foundation) | HF `Zyphra/ZUNA` | `hf_hub_download` in `src/mindseye/zuna/latent_extractor.py` (config.json + `model-00001-of-00001.safetensors`) | `HF_HOME` |
| **RAE decoder / DINOv2** | HF `nyu-visionx/RAE-dinov2-wReg-base-ViTXL-n08` | `AutoencoderRAE.from_pretrained` in `src/mindseye/generation/rae_backend.py` | `HF_HOME` |

To pre-warm the cache on a fresh volume (optional — otherwise they download lazily on first run):
```bash
export HF_HOME=/workspace/hf_cache
python -c "from huggingface_hub import hf_hub_download; \
  hf_hub_download('Zyphra/ZUNA','config.json'); \
  hf_hub_download('Zyphra/ZUNA','model-00001-of-00001.safetensors')"
python -c "from diffusers import AutoencoderRAE; \
  AutoencoderRAE.from_pretrained('nyu-visionx/RAE-dinov2-wReg-base-ViTXL-n08')"
```

## 4c. Bootstrapping a fresh (empty) volume from scratch

When the volume is brand new / empty, nothing above the code exists yet. Rebuild the durable data
in order — this is exactly what `cold_start.sh` automates. All raw inputs come from public sources,
so **no data needs to be pushed from the laptop**.

```
- [ ] Code + deps (§4), torch re-pinned to 2.8.0
- [ ] Model weights warm to HF_HOME (§4b, or lazy on first run)
- [ ] Raw EEG + stimuli   → data/raw/nod/        (OpenNeuro ds005811, public)
- [ ] ZUNA denoise        → data/processed/zuna_real/4_fif_output/
- [ ] 5s back-aligned crop → data/processed/semantic_epochs/zuna_full5s_backaligned_*/  (--full5s-backaligned)
- [ ] Target bank         → data/processed/rae_embeddings/rae_dinov2_base_all.pt   (RAE/DINO; CLIP dropped)
- [ ] Cache ZUNA latents  → data/processed/zuna_latents/cohort9_runs01_32/
```

```bash
cd /workspace/mindeye && export PYTHONPATH=src HF_HOME=/workspace/hf_cache
# Full 9-subject cohort (recommended — positive results need scale):
SKIP_ENV=1 bash scripts/prepare_multisubject_data.sh
# Single-subject dev slice:
SKIP_ENV=1 bash cold_start.sh
```

Raw data sources (used by `cold_start.sh`, safe to run standalone):
- **EEG (ds005811)** via `scripts/download_nod.py` / `download_nod_s3.py` — OpenNeuro public dataset.
- **ImageNet stimuli** via `scripts/sync_stimuli_s3_targeted.py` — unsigned (public) S3 bucket, only the referenced images (from an include-list).

## 4d. Disaster recovery — lost volume or lost pod

Everything durable is either **public** (raw EEG, stimuli, model weights) or **reproducible** (all
processed data + embeddings + cached latents are deterministic outputs of the pipeline). The only
artifacts that are *not* trivially reproducible are **trained checkpoints** under `outputs/`.

| What was lost | Recovery |
|---|---|
| **Pod** (volume intact) | Create a new pod in the volume's datacenter, attach the volume, `git pull` + `pip install`. Data/outputs/cache are all present. (§2) |
| **Volume** (data gone) | Rebuild from scratch via `cold_start.sh` (§4c). Model weights + raw data re-download from public sources. Regenerating all processed data is a full pipeline run. |
| **`outputs/` only** | Re-run training. This is why outputs are the one thing worth periodically pulling to the laptop (`rsync` pull, §5) or copying to a second volume. |

> **Backup guidance**: raw and processed data are recoverable, so they don't need backup. **Trained
> checkpoints do** — pull `outputs/` to the laptop after any run you care about, before deleting the
> volume or pod. There is no automatic backup of a RunPod network volume.

## 5. SSH shortcut (laptop `~/.ssh/config`)

```
Host mindeye-pod
  HostName <PUBLIC_IP>
  Port <SSH_PORT>
  User root
  IdentityFile ~/.ssh/runpod
  StrictHostKeyChecking no
```

Then: `ssh mindeye-pod`.

## 6. Disk budget

Measured on the 2026-07-03 9-subject run — NOD EEG is **much smaller** than earlier estimates:

| Path (on network volume) | Size | Notes |
|---|---|---|
| `data/raw/nod` | **~0.6 GB / subject** (~6 GB for 9) | Raw NOD EEG (run FIF ~11 MB, subject epoch FIF ~215 MB) + referenced ImageNet stimuli |
| `data/processed/` | tens of GB | ZUNA outputs, 5s epochs, cached latents |
| `data/processed/rae_embeddings/rae_dinov2_base_all.pt` | **~14 GB** | The single biggest data artifact |
| `hf_cache` | ~20 GB | ZUNA + RAE/DINOv2 weights |
| `outputs/` | grows | Checkpoints, metrics, grids |

- The old "~20 GB / subject" figure was wrong by ~30×. The **RAE bank (~14 GB) and hf_cache (~20 GB)** dominate, not raw EEG.
- **200 GB volume is ample** even for the full 9-subject cohort. Do **not** over-request volume/container disk: A100 hosts reject large disk requests with *"no instances available with enough disk space"* (500–700 GB failed; 200 GB + 60 GB container succeeded).
- RunPod volumes can be grown but not shrunk.

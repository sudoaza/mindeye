# MindEye Infrastructure — RunPod + Network Volume

> All GPU work runs on a **RunPod Secure Cloud** pod, managed via the **runpod MCP**
> (see [`RunPod_SKILL.md`](RunPod_SKILL.md)). The local dev machine has no GPU.

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
3. On the new pod: `cd /workspace/mindeye && git pull` → `pip install -r requirements.txt` → `export PYTHONPATH=src`. Data/outputs/cache are already present from the volume.

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
| **Container disk** | 100 GB | Ephemeral scratch, pip build cache, torch tmp |
| **Network volume** | 200 GB | Persistent `/workspace`: raw EEG + processed data + outputs + hf_cache |
| **Image** | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` | Matches pinned torch/CUDA (`torch==2.6.0+cu124`) in `requirements.txt` |

> Canonical sizes: **containerDiskInGb = 100**, **volumeInGb = 200**. Earlier docs quoted 50/80–100;
> those are too tight once all 4 subjects at full runs (~20 GB raw each) plus ZUNA outputs, embedding
> banks, HF weights (~20 GB), and checkpoints coexist. Size up before pulling the full dataset; a
> single-subject dev volume can be smaller (100 GB).

### Provisioning JSON (runpod MCP `create-pod`)

```json
{
  "cloudType": "SECURE",
  "gpuCount": 1,
  "volumeInGb": 200,
  "containerDiskInGb": 100,
  "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
  "name": "mindeye-qformer",
  "ports": ["22/tcp"],
  "volumeMountPath": "/workspace",
  "env": {"PUBLIC_KEY": "<contents of ~/.ssh/id_ed25519.pub>"}
}
```

> [!WARNING]
> Pods not provisioned with your `~/.ssh/id_ed25519.pub` as `PUBLIC_KEY` will prompt for a password and block SSH. Always pass the ed25519 pub key.

## 4. Fresh-pod setup

```bash
ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no

cd /workspace/mindeye && git pull origin master     # code (volume or fresh clone)
pip install -r requirements.txt                     # system Python, no venv

# Always set before any script:
export PYTHONPATH=/workspace/mindeye/src
export HF_HOME=/workspace/hf_cache
export TMPDIR=/workspace/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

> The pod uses **system Python with `PYTHONPATH=src`** — no venv. A venv created on the pod can end up with zero-byte binaries after a container restart; if you hit that, delete it and use system Python.

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
- [ ] Code + deps (§4)
- [ ] Model weights warm to HF_HOME (§4b, or lazy on first run)
- [ ] Raw EEG + stimuli   → data/raw/nod/        (OpenNeuro ds005811, public)
- [ ] ZUNA denoise        → data/processed/zuna_real/
- [ ] Onset-aligned crop  → data/processed/semantic_epochs/
- [ ] Target banks        → data/processed/{clip_embeddings,rae_embeddings}/
- [ ] Cache ZUNA latents  → data/processed/zuna_latents/
```

```bash
cd /workspace/mindeye && export PYTHONPATH=src HF_HOME=/workspace/hf_cache
bash cold_start.sh            # end-to-end; edit subject/run range at the top for full vs dev
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
  IdentityFile ~/.ssh/id_ed25519
  StrictHostKeyChecking no
```

Then: `ssh mindeye-pod`.

## 6. Disk budget

| Path (on network volume) | Size | Notes |
|---|---|---|
| `data/raw/nod` | ~20 GB / subject (~80 GB for 4) | Raw NOD EEG + referenced ImageNet stimuli |
| `data/processed/` | tens of GB | ZUNA outputs, epochs, embedding banks, cached latents |
| `hf_cache` | ~20 GB | ZUNA + RAE/DINOv2 weights |
| `outputs/` | grows | Checkpoints, metrics, grids |

- **Single-subject dev**: ~100 GB volume is comfortable.
- **Full 4-subject**: use the canonical **200 GB** volume — raw (~80 GB) + processed + `hf_cache`
  (~20 GB) + outputs leave little headroom below that. Size up the volume before pulling all
  subjects; RunPod volumes can be grown but not shrunk.

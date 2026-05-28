# MindEye Dev Cheat Sheet

> IMPORTANT: All GPU work runs on the **remote RunPod pod**. The dev machine has no GPU.

## Index
- [RunPod Management](#runpod-management)
- [SSH & File Transfer](#ssh--file-transfer)
- [Running the Pipeline](#running-the-pipeline)
- [Background Execution & Logging](#background-execution--logging)
- [Common Pitfalls & Troubleshooting](#common-pitfalls--troubleshooting)
  - [EEG Model Architecture](#eeg-model-architecture--training)
  - [Pipeline, Data & Scripts](#pipeline-data--script-execution)
  - [RunPod & Infrastructure](#runpod--infrastructure)
  - [Phase 12 / Large Model (Qwen-Image)](#phase-12--large-model-infrastructure-qwen-image)


---

## RunPod Management

Use the `runpod` MCP tools. See [`docs/RunPod_SKILL.md`](RunPod_SKILL.md) for full details.

```bash
mcp_runpod_list-pods
mcp_runpod_get-pod    {"podId": "<POD_ID>", "includeMachine": true}
mcp_runpod_start-pod  {"podId": "<POD_ID>"}
mcp_runpod_stop-pod   {"podId": "<POD_ID>"}
```

### Create a New Pod
```json
{
  "cloudType": "SECURE",
  "gpuCount": 1,
  "volumeInGb": 80,
  "containerDiskInGb": 50,
  "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
  "name": "mindeye-zuna",
  "ports": ["22/tcp"],
  "volumeMountPath": "/workspace",
  "env": {"PUBLIC_KEY": "ssh-ed25519 AAA... user@example.com"}
}
```

---

## SSH & File Transfer

Always use `-i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no`.

### Preferred: git pull on pod (no file transfer needed)
```bash
ssh -p <PORT> root@<IP> "cd /workspace/mindeye && git pull origin master"
```

### Rsync (last resort — debugging only)
```bash
# Install rsync first if needed:
ssh -p <PORT> root@<IP> "apt-get update && apt-get install -y rsync"

# Push code (exclude data and outputs to avoid clobbering remote checkpoints):
rsync -avz --no-o --no-g --no-perms \
  --exclude 'venv' --exclude '.git' --exclude '__pycache__' \
  --exclude 'outputs/' --exclude 'data/' \
  -e "ssh -p <PORT>" . root@<IP>:/workspace/mindeye/

# Pull remote outputs/checkpoints first:
rsync -avz --no-o --no-g --no-perms \
  -e "ssh -p <PORT>" \
  root@<IP>:/workspace/mindeye/outputs/ outputs/
```

> [!CAUTION]
> **Never rsync local → remote without pulling outputs first.** Any `outputs/` generated on the pod (checkpoints, probe models, matrix results) will be silently overwritten by stale local copies.

### Install Dependencies
```bash
# Install requirements (uses system Python — no venv needed on pod):
ssh -p <PORT> root@<IP> "cd /workspace/mindeye && pip install -r requirements.txt"

# IMPORTANT: Upgrade PyTorch to 2.6.0 for flex_attention support on CUDA 12.4:
ssh -p <PORT> root@<IP> "pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
```

---

## Running the Pipeline

For step-by-step details see [`scripts/README.md`](../scripts/README.md). The canonical entry points are via `make`:

```bash
make pipeline    # Full end-to-end: download → ZUNA → crop → embeddings → matrix
make matrix      # Matrix only (assumes prior steps done) — canonical training run
make ablation    # no-probe vs probe comparison (faster, ~2-3h)
make probe_sweep # Sweep probe weights 0 / 0.01 / 0.03 / 0.05 / 0.10
make simulate    # EPOC-14 low-channel simulation
```

### Phase 12A / 13 (Decode-Unit unCLIP Branch) — run on pod manually
```bash
# Step 0: Extract z_decode_common targets (one-time)
python scripts/build_decode_common_embeddings.py

# Step 1: Retrain decode probe on decode_unit space (one-time)
python scripts/pretrain_common_probe.py \
  --target-key decode_unit \
  --common-embeddings data/processed/clip_embeddings/decode_common_embeddings.pt \
  --metadata data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv \
  --vlm-attributes data/processed/clip_embeddings/vlm_attributes.json \
  --output-dir outputs/decode_probe_v2 \
  --epochs 50 --lr 1e-4

# Step 2: Train EEG -> decode_unit (canonical Phase 13 config)
# Uses --mode contrastive_only, probe_weight=0.01, probe_start_epoch=5
python scripts/train_eeg_decode_common.py \
  --mode contrastive_only \
  --common-probe outputs/decode_probe_v2/common_probe.pt \
  --vlm-attributes data/processed/clip_embeddings/vlm_attributes.json \
  --probe-weight 0.01 \
  --probe-start-epoch 5 \
  --epochs 30 \
  --slug 13_probe001

# Step 3: Evaluate generations (Oracle, EEG kNN, Shuffled, Random)
python scripts/evaluate_clip_native_decoder.py \
  --run-dir outputs/runs/<YOUR_RUN_DIR>

### Phase 17 (DINOv2-RAE Target & Decoder Swap) — run on pod manually

DINOv2-RAE is a 768-D target space. It uses a frozen DINOv2-RAE encoder and a trained decoder. The RAE-native pipeline uses target centering (Option B) to reduce conic crowd density.

# Step 0: Pretrain the RAE-native CommonProbeModel (one-time)
python scripts/train_rae_probe.py \
  --target-key image_id_to_rae_centered_unit \
  --common-embeddings data/processed/rae_embeddings/rae_dinov2_base_sub01_04_runs01_40.pt \
  --metadata data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv \
  --vlm-attributes data/processed/clip_embeddings/vlm_attributes.json \
  --output-dir outputs/rae_probe \
  --epochs 80 --lr 1e-4

# Step 1: Train EEG -> rae_centered_unit (Option B target centering)
# Initialized from the Phase 16 16c_film_heads checkpoint, skipping heads
python scripts/train_eeg_clip.py \
  --metadata "$METADATA" \
  --epochs-dir "$EPOCHS_DIR" \
  --common-embeddings data/processed/rae_embeddings/rae_dinov2_base_sub01_04_runs01_40.pt \
  --target-space rae_centered_unit \
  --target-key image_id_to_rae_centered_unit \
  --window-mode tight1s \
  --augment-eeg \
  --model temporal_attn_small \
  --epochs 50 \
  --patience 15 \
  --batch-size 128 \
  --loss contrastive \
  --temperature 0.07 \
  --slug phase17_6_rae_centered_16c_init_augmented \
  --output-dir outputs \
  --vlm-attributes data/processed/clip_embeddings/vlm_attributes.json \
  --common-probe outputs/rae_probe/common_probe.pt \
  --probe-weight 0.01 \
  --probe-start-epoch 5 \
  --head-reg-weight 0.01 \
  --init-from outputs/runs/20260527_083218_zuna_real_16c_film_heads/best.pt \
  --init-skip-heads \
  --lr 1e-4 \
  --device cuda

# Step 2: Evaluate generations (Oracle, EEG kNN, Shuffled, Random)
# Re-encodes generated images, centers, normalizes, and runs bootstrap + RAE probe
python scripts/evaluate_rae_generation.py \
  --run-dir outputs/<YOUR_RUN_DIR> \
  --num-samples 500 \
  --batch-size 25 \
  --k 5 \
  --target-key image_id_to_rae_centered_unit \
  --temperature 0.05 \
  --common-probe outputs/rae_probe/common_probe.pt \
  --stimuli-dir data/raw/nod/stimuli/ImageNet \
  --output-dir outputs/phase17_6_rae_eval
```

On the pod, prefix with `export PYTHONPATH=src &&` when running scripts directly:
```bash
ssh -p <PORT> root@<IP> "cd /workspace/mindeye && export PYTHONPATH=src && nohup make matrix > matrix.log 2>&1 &"
```

---

## Background Execution & Logging

Long jobs **must** use `nohup` to survive SSH disconnection:

```bash
nohup make matrix > matrix.log 2>&1 &
nohup bash run_epoc_simulation.sh > epoc14.log 2>&1 &
```

### Process Management
```bash
ps aux | grep python               # Check running jobs
tail -f /workspace/mindeye/matrix.log  # Follow log live
pkill -f 'python.*run_baseline_matrix' # Kill matrix run
```

### Output Locations
| Artifact | Location |
|---|---|
| Console logs | `/workspace/mindeye/*.log` (e.g. `matrix.log`, `epoc14.log`) |
| Per-run metrics | `outputs/runs/<timestamp>_<slug>/metrics.json` |
| Per-epoch CSV | `outputs/runs/<timestamp>_<slug>/train_log.csv` |
| Matrix summary | `outputs/baseline_matrix/<timestamp>_matrix/matrix_summary.csv` |
| Probe model (decode space v2) | `outputs/decode_probe_v2/common_probe.pt` |
| Probe model (old z_common)    | `outputs/common_probe/common_probe.pt` |

**Key `metrics.json` fields**: `top1`, `top5`, `top10`, `mrr`, `median_rank`, `mean_diag_cosine`, `collapse_score`, `best_epoch`, `full_bank_top10`, `full_bank_mrr`.

> **Primary gate metric is `full_bank_top10`** (predictions ranked against all 4000 images, expected random = 0.0025). `within_val_top10` inflates signal by ~20–4× and is diagnostic only. `collapse_score` must be **> 0.1** to pass baseline gates.

---

## Common Pitfalls & Troubleshooting

### EEG Model Architecture & Training

- **BatchNorm → GroupNorm**: `BatchNorm1d` in temporal/spatial stems causes train/eval discrepancy on small-batch EEG, collapsing validation performance. Use `GroupNorm` instead — it computes statistics per-sample.
- **Augmentation on marker channel**: The event marker channel (last channel, e.g. index 63) must be **excluded** from noise/masking augmentations. Corrupting it destroys event-EEG alignment.
- **Control label leakage**: In `shuffled`/`random` conditions, auxiliary probe labels must be shuffled/randomized *identically* to the main contrastive target. Otherwise control scores are artificially inflated.
- **Collapse guard in checkpoint saving**: Never select checkpoints purely on MRR/Top-10 — the model may have collapsed to a constant vector. Enforce `collapse_score > 0.1` (ideally > 1.0) as a hard gate before saving.
- **Channel count mismatch after checkpoint load**: A model trained with N channels (e.g. 63 + event marker = 64) will crash at inference if the evaluation dataset returns N-1 channels. Ensure `--add-event-marker` is passed consistently between training and evaluation. Error: `RuntimeError: expected input to have 63 channels, but got 62`.
- **`return_features` API signature**: Models used with downstream multitask heads must support `return_features: bool = False` in `forward()`, returning `(output, features)` when `True`.

### Pipeline, Data & Script Execution

- **VLM output field name mismatch**: If the VLM generates field `structured_embedding_text` but the parser reads `embedding_text`, the fallback becomes `"empty"` for every image — collapsing the entire target embedding space to a single point. Always verify JSON key names end-to-end.
- **CLIP model output type**: `CLIPModel.get_image_features()` returns a plain tensor, but `CLIPModel()` (full forward pass) returns a `BaseModelOutputWithPooling` object. Calling `.norm()` on the latter raises `AttributeError`. Use `.image_embeds` or call `get_image_features()` directly.
- **Missing `argparse` attribute at runtime**: Adding a new CLI flag to a script but forgetting to add it to a downstream caller (e.g. `evaluate_retrieved_priors.py` missing `--num-grid`) causes `AttributeError: 'Namespace' object has no attribute 'num_grid'`. Always add defaults to `parse_args()` and keep CLI consistent across callers.
- **CSV fieldname drift in training loop**: If new probe metrics (e.g. `calib_probe_class_label_top10_acc`) are added to the row dict but not to the `fieldnames` list passed to `csv.DictWriter`, training crashes at the end of the first epoch with `ValueError: dict contains fields not in fieldnames`. Keep `log_fields` and the metrics dict in sync.
- **Stimulus overlap in RSVP / wide windows**: NOD presents images every ~1.3–1.7s. Wide windows (e.g. `[-3s, +2s]` or `[-1s, +4s]`) capture responses to multiple stimuli simultaneously. Use tight windows (`-0.2s` to `+1.0s`, 1.2s total) to isolate the target response.
- **Off-by-one sample rounding**: Window clipping at 256Hz can produce 1281 instead of 1280 samples due to floating-point rounding. Accept both lengths in the dataset validator and slice/truncate to the expected size rather than raising a hard error.
- **Pipeline execution order**: VLM attribute generation must complete before CLIP/text embedding generation. Use cached `common_embeddings.pt` to avoid regenerating large embeddings on every run.
- **VAL_RUN / metadata alignment**: Dynamically read the actual number of downloaded runs (e.g. 32) rather than hardcoding a higher expected value (e.g. 40). Mismatched run counts produce an empty validation split crash: `ValueError: Invalid run split: train=3974 val=0`.

### RunPod & Infrastructure

- **SSH key mismatch on pod creation**: Pods not provisioned with `~/.ssh/id_ed25519.pub` as `PUBLIC_KEY` will prompt for a password and block SSH. Always explicitly pass the ed25519 pub key.
- **rsync overwrites remote checkpoints**: See the [rsync note above](#ssh--file-transfer). Pull before push.
- **FlexAttention OOM / missing features**: PyTorch < 2.6.0 on the base RunPod image causes `flex_attention` CUDA OOMs. Always upgrade to `torch==2.6.0+cu124` after pod creation.
- **`source venv/bin/activate` on pod**: The pod uses system Python with no venv. Replace `source venv/bin/activate` with `export PYTHONPATH=src` in any scripts synced to the pod.
- **Venv corruption on pod**: If a venv was created on the pod and then the container stopped/restarted, the venv may have zero-byte binaries (`python3` is 0 bytes, 000 permissions). Delete and recreate, or just use system Python with `PYTHONPATH=src`.
- **Multiple debug commands**: Gather all needed info in one SSH call rather than running separate commands for each check. SSH connection overhead is significant.

- **Within-val vs full-bank retrieval gap**: `within_val_top10` compares EEG predictions against only the val batch (n≈596). This inflates numbers 2–4× vs. the honest `full_bank_top10` metric (predictions vs all 4000 image embeddings). Always use `full_bank_top10` and `full_bank_mrr` as primary gate metrics. A model can achieve within-val Top-10 = 0.042 while being BELOW random on full-bank (as A_real_repro demonstrated).
- **Target extraction compatibility**: Use `StableUnCLIPImg2ImgPipeline` instead of `StableUnCLIPPipeline` for image-to-image extraction.
- **Unnormalized targets**: When building `z_decode_common`, use `extract_teacher_embeds(normalize=False)`. Unnormalized embeds retain essential structural information for image reconstruction; L2-normalized embeddings fall back to random-level chance.
- **Relative evaluation gating**: Absolute cosine scores vary. Always evaluate relative performance against random embedding baselines: `Oracle Cosine > Random Cosine + 0.05`.

### Phase 17 (DINOv2-RAE)
- **Attribute Probe Mismatch**: Never feed RAE-space target vectors into `outputs/decode_probe_v2/common_probe.pt` (which was trained on Stable unCLIP decode_unit space). Use the RAE-native probe `outputs/rae_probe/common_probe.pt`.
- **Target Centering in Evaluation**: Generated images must be encoded using the RAE backend, subtracted by the training set mean (`rae_center_mean`), and then L2-normalized. Raw unit comparison will hide retrieval performance.
- **Option B Coordinate Alignment**: Make sure `pred_for_loss` is defined as `normalize(pred - target_center)` during training, and the auxiliary probe model gets `pred_for_loss` as input (instead of uncentered prediction).

### Phase 18A (RAE Token Bottleneck Autoencoder)

The bottleneck decouples EEG↔RAE into two independent systems:
1. **Image-only bridge**: compress [768,16,16] RAE tokens → [C,4,4] code → expand back → decode image.
2. **EEG→code**: EEG encoder predicts the compact code (Phase 18B, not yet started).

```bash
# Step 0: Train the bottleneck (on pod, tokens already in RAE bank)
python scripts/train_rae_token_bottleneck.py \
    --rae-bank data/processed/rae_embeddings/rae_dinov2_base_sub01_04_runs01_40.pt \
    --arch conv_256x4x4 \
    --epochs 50 \
    --batch-size 64 \
    --lr 1e-3 \
    --output-dir outputs/rae_bottleneck/conv_256x4x4 \
    --device cuda

# To run all three archs in sequence (background):
for arch in spatial_768x4x4 conv_256x4x4 conv_128x4x4; do
    python scripts/train_rae_token_bottleneck.py \
        --rae-bank data/processed/rae_embeddings/rae_dinov2_base_sub01_04_runs01_40.pt \
        --arch $arch \
        --epochs 50 \
        --output-dir outputs/rae_bottleneck/$arch \
        --device cuda
done

# Step 1: Check gate metrics from outputs/rae_bottleneck/<arch>/metrics.json
#   gate_token_cosine_pass: true   (mean_token_cosine > 0.90)
#   gate_collapse_pass:     true   (pct_collapsed_channels < 5%)

# Step 2: Extract codes for all images (after bottleneck is validated)
python scripts/build_rae_bottleneck_codes.py \
    --rae-bank data/processed/rae_embeddings/rae_dinov2_base_sub01_04_runs01_40.pt \
    --checkpoint outputs/rae_bottleneck/conv_256x4x4/best.pt \
    --output data/processed/rae_embeddings/rae_bottleneck_codes_conv256.pt \
    --device cuda
```

**Gate criteria (must pass before Phase 18B):**
- `mean_token_cosine > 0.90` — bottleneck faithfully reconstructs token geometry
- `pct_collapsed_channels < 5%` — no mode collapse in the code
- If `conv_256x4x4` fails, try `conv_128x4x4` before falling back to `spatial_768x4x4` (upper bound).

**Architecture summary:**
| Arch | Code shape | Values | Notes |
|---|---|---|---|
| `spatial_768x4x4` | [768,4,4] | 12,288 | Parameter-free compressor; learned expander — reconstruction upper bound |
| `conv_256x4x4` | [256,4,4] | 4,096 | **Primary EEG target** |
| `conv_128x4x4` | [128,4,4] | 2,048 | Leaner — compression tolerance test |
| `conv_256x8x8` | [256,8,8] | 16,384 | Fallback if 4x4 fails visually |

**Gate criteria (aspirational — compare all archs before deciding):**
- `mean_token_cosine > 0.90` — bottleneck faithfully reconstructs token channel geometry
- `pct_collapsed_channels < 5%` — relative std_ratio > 0.2 per channel (code_std / tok_std)
- These are aspirational targets, not hard fails. RAE latents are high-dimensional; compare across architectures before deciding.
- If `conv_256x4x4` fails visually, try `conv_256x8x8` before falling back to `spatial_768x4x4`.

**Evaluation visual comparison (3-way):**
- Target image
- Full-token RAE oracle (image → full tokens → decode)
- Bottleneck reconstruction (image → compress → expand → decode)

This 3-way comparison is the ground truth for selecting the bottleneck. Run all three architectures before extracting codes.

**Code extraction** — only run for the **selected** architecture (not mandatory for all three).


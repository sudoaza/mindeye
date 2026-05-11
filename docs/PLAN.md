# Mind's Eye Technical Plan

## Objective
Build an EEG→semantic→image system using **ZUNA** as the signal normalization layer. Train on **NOD-EEG** (continuous visual EEG with stimulus images), pivoting away from the Alljoined short-epoch approach.

## Core Architecture
1. **Dataset**: NOD-EEG (ds005811)
   - Continuous visual EEG recordings (.fif files) with 62 channels and 3D montages.
   - Rich metadata with ImageNet synsets and event timings.
2. **Signal Normalization**: ZUNA
   - ZUNA takes 5s continuous EEG windows at 256Hz.
   - It outputs denoised, reconstructed, standardized EEG signals.
3. **Event Aligned Cropping**:
   - Extract shorter task-relevant windows (e.g., 1.25s around stimulus) from ZUNA's 5s reconstructed output.
4. **Semantic Embedding**:
   - Train an EEG→CLIP semantic encoder mapping the standardized EEG latent space to CLIP embeddings of the visual stimuli.
5. **Image Generation**:
   - A frozen diffusion img2img loop conditioned on the predicted CLIP embeddings to reconstruct what the subject is seeing.

## Key Findings (Latest Pipeline Test)
- **Data Availability**: `sub-01` tested successfully.
- **Pre-existing Epochs**: The existing dataset epochs are 0.9s (-0.1s to +0.8s), which are too short for ZUNA. ZUNA strictly requires 5s continuous windows to function.
- **Channels**: The data has 62 channels with a pre-set 3D montage, making it directly compatible with ZUNA.
- **Rich Metadata**: We have 4,000 trials for `sub-01`, rich with ImageNet synsets, `class`, `super_class` (like 'canine', 'device', 'artifact'), and `face_score`.


## Current Status — 2026-05-11
- Existing RunPod pod `vm7hhvxx1mx40s` is stopped (`desiredStatus=EXITED`); no training is currently running. A restart attempt on 2026-05-11 failed because RunPod reported no free GPUs on that host.
- Latest baseline evidence: 5-epoch cosine+MSE EEG→CLIP run on `sub-01` runs 01-05 improved loss but retrieval was near chance and visually collapsed toward repeated CLIP hub images.
- Code now defaults to contrastive EEG↔image training and supports CLIP target centering plus run-level validation split. Next GPU step is to train the improved objective and regenerate the retrieval grid.

Recommended next command once a GPU pod is available:

```bash
python scripts/train_eeg_clip.py \
  --device cuda \
  --epochs 80 \
  --batch-size 128 \
  --loss contrastive \
  --temperature 0.07 \
  --center-clip \
  --split-mode run \
  --val-runs 5 \
  --output-dir outputs/eeg_clip_contrastive_sub01_runs01_05_ep080_run05
```

## Implementation Phases

### Phase 0: Cleanup & Restructure (Completed)
- [x] Removed all legacy Alljoined code and configurations.
- [x] Rebuilt `src/mindseye/` structure for `datasets`, `zuna`, `embeddings`, `models`, `train`, `inference`.
- [x] Integrated `zuna` and `openneuro-py` dependencies into `pyproject.toml` and `requirements.txt`.
- [x] Rewrote dataset loader and pipelines for NOD-EEG and ZUNA compatibility.

### Phase 1: NOD-EEG Ingestion (Completed)
- **Goal**: Download and load NOD-EEG raw continuous `.fif` files.
- **Current State**: Downloaded `sub-01` epoch file, events CSV, and continuous `.fif` runs. Tested loader and metadata mapping successfully.

### Phase 2: ZUNA Integration (Completed for sub-01 runs 01-05)
- **Goal**: Process continuous `.fif` files through the ZUNA normalization model.
- **Actions**:
  - [x] Run `offline_pipeline.py` / `run_zuna_batch.py` to batch process continuous `.fif` files through ZUNA's denoising and inference.
  - [x] Reconstruct the output into standardized `.fif` files (resampled to 256Hz).
  - [x] Use `cropper.py` to crop event-aligned 1.25s windows around stimulus onsets. Current implementation aligns from original raw FIF `stim_on` annotations because ZUNA output FIFs do not preserve annotations.

### Phase 3: CLIP Embedding Generation (Completed for sub-01 runs 01-05)
- **Goal**: Generate ground-truth CLIP embeddings for the visual stimuli.
- **Actions**:
  - [x] Add CLIP embedding utility/CLI and targeted OpenNeuro include-list generation for cropped metadata.
  - [x] Add targeted `download_nod.py --include-list` support so the CLIP stimulus images can be fetched without grabbing all ImageNet stimuli.
  - [x] Download stimulus images from OpenNeuro `stimuli/ImageNet/` for the cropped subset.
  - [x] Process images through a pre-trained CLIP vision encoder to generate `[image_id, embedding]` pairs.
  - [x] Save as a persistent embedding dictionary (`.pt`); current sub-01 runs 01-05 table is 618 unique images × 512 dims.

### Phase 4: EEG→CLIP Encoder Training (Baseline + Contrastive Upgrade)
- **Goal**: Train the initial projection model mapping ZUNA-normalized EEG crops to CLIP embeddings.
- **Actions**:
  - [x] Add dataset-pair loader for `(1.25s ZUNA-cleaned EEG crop, target CLIP embedding)` tables.
  - [x] Add a small baseline temporal-conv EEG→CLIP encoder and train/eval CLI. Initial cosine+MSE smoke run showed hub/collapse behavior.
  - [x] Add retrieval metrics scaffold (top-1, top-5 on validation target bank).
  - [x] Run 5-epoch smoke baseline after CLIP embeddings exist; val loss fell but retrieval stayed near chance and repeated the same hub images.
  - [x] Add CLIP-style symmetric contrastive / InfoNCE loss, optional train-set CLIP mean-centering, and run-heldout validation support.
  - [ ] Run improved contrastive training when RunPod capacity is available, e.g. 80 epochs with `--center-clip --split-mode run --val-runs 5`.

### Phase 5: Image Generation Diffusion Loop
- **Goal**: Hook the predicted semantic embeddings into a stable diffusion pipeline.
- **Actions**:
  - [ ] Use a frozen diffusion model (e.g., Stable Diffusion Image Variations or Versatile Diffusion).
  - [ ] Feed the predicted CLIP embedding to generate images.
  - [ ] Evaluate generated images against ground truth stimuli visually and via metrics (SSIM, FID, CLIP similarity).

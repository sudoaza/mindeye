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

### Phase 3: CLIP Embedding Generation (In Progress)
- **Goal**: Generate ground-truth CLIP embeddings for the visual stimuli.
- **Actions**:
  - [x] Add CLIP embedding utility/CLI and targeted OpenNeuro include-list generation for cropped metadata.
  - [ ] Download stimulus images from OpenNeuro `stimuli/ImageNet/`.
  - [ ] Process images through a pre-trained CLIP vision encoder to generate `[image_id, embedding]` pairs.
  - [ ] Save as a persistent embedding dictionary (`.pt` or `.zarr`).

### Phase 4: EEG→CLIP Encoder Training (Baseline)
- **Goal**: Train the initial projection model mapping ZUNA-normalized EEG crops to CLIP embeddings.
- **Actions**:
  - [ ] Construct dataset pairs: (1.25s ZUNA-cleaned EEG crop, target CLIP embedding).
  - [ ] Implement a contrastive loss or direct MSE training loop.
  - [ ] Evaluate retrieval metrics (top-1, top-5 retrieval accuracy on test set).

### Phase 5: Image Generation Diffusion Loop
- **Goal**: Hook the predicted semantic embeddings into a stable diffusion pipeline.
- **Actions**:
  - [ ] Use a frozen diffusion model (e.g., Stable Diffusion Image Variations or Versatile Diffusion).
  - [ ] Feed the predicted CLIP embedding to generate images.
  - [ ] Evaluate generated images against ground truth stimuli visually and via metrics (SSIM, FID, CLIP similarity).

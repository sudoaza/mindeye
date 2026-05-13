# MindEye Scripts

This directory contains the orchestration scripts for the MindEye EEG-to-CLIP pipeline.

## 1. Data Preparation
* `download_nod.py`: Downloads continuous NOD dataset recordings from OpenNeuro (Runs 1-8).
* `sync_stimuli_s3_targeted.py`: Syncs ImageNet stimuli images from S3.

## 2. Signal Processing
* `run_zuna_batch.py`: Orchestrates the continuous ZUNA diffusion-based denoising pipeline on the raw `.fif` files.
* `run_cropper.py`: Crops the continuous files into semantic epochs. 
  * *Canonical Usage*: `python scripts/run_cropper.py --mode zuna --full5s-backaligned --add-event-marker`

## 3. Multimodal Latent Space
* `generate_clip_embeddings.py`: Generates canonical 512-dim CLIP embeddings for stimuli images.
* `generate_image_semantics.py`: Uses a VLM (Qwen2.5-VL) to extract semantic captions from images.
* `generate_text_embeddings.py`: Generates text CLIP embeddings (from captions or ImageNet labels).
* `build_common_embeddings.py`: Fuses image, semantic, and label signals into a single `z_common` target space using $L_2$ normalization and weighted combinations.

## 4. Core Training Pipeline
* `train_eeg_clip.py`: The single-condition training loop. Uses the `ZunaClipPairDataset` and trains an EEG encoder (e.g. `temporal_attn_small` or `cnn`) against the multimodal target space.
* `run_baseline_matrix.py`: Orchestrates comparative matrix runs (e.g., `zuna_real` vs `zuna_shuffled` vs `zuna_random`). Outputs results to a timestamped matrix directory and streams progress to `matrix_run.log`.

## Canonical Execution Order (Sprint 2/3)
To reproduce the Phase 3 architecture from scratch, run the scripts in this sequence:

```bash
# 1. Fetch Data
python scripts/download_nod.py --subject sub-01 --runs 1-8
python scripts/sync_stimuli_s3_targeted.py

# 2. Process Signals
python scripts/run_zuna_batch.py --diffusion-steps 15
python scripts/run_cropper.py \
    --mode zuna \
    --full5s-backaligned \
    --add-event-marker \
    --runs 1 2 3 4 5 6 7 8 \
    --zuna-dir data/processed/zuna_real/4_fif_output \
    --output-dir data/processed/semantic_epochs/zuna_full5s_backaligned_sub01_runs01_08

# 3. Generate Multimodal Latent Space
python scripts/generate_clip_embeddings.py
python scripts/generate_text_embeddings.py
python scripts/generate_image_semantics.py
python scripts/build_common_embeddings.py \
    --image-embeddings data/processed/clip_embeddings/sub01_image_embeddings.pt \
    --semantic-embeddings data/processed/clip_embeddings/image_semantic_text_embeddings.pt \
    --label-embeddings data/processed/clip_embeddings/imagenet_text_embeddings.pt \
    --metadata data/raw/nod/derivatives/detailed_events/sub-01_events.csv \
    --w-img 0.25 --w-sem 0.65 --w-lbl 0.10 \
    --output data/processed/clip_embeddings/common_embeddings.pt

# 4. Train Model via Baseline Matrix
python scripts/run_baseline_matrix.py \
    --metadata data/processed/semantic_epochs/zuna_full5s_backaligned_sub01_runs01_08/all_runs_metadata.csv \
    --epochs-dir data/processed/semantic_epochs/zuna_full5s_backaligned_sub01_runs01_08 \
    --common-embeddings data/processed/clip_embeddings/common_embeddings.pt \
    --val-runs 8 \
    --window-mode full5s_backaligned \
    --target-space common \
    --model temporal_attn_small \
    --epochs 50 \
    --batch-size 64 \
    --device cuda \
    --slug common_space_sprint2 \
    --add-event-marker \
    --augment-eeg \
    --conditions zuna_real zuna_shuffled zuna_random
```

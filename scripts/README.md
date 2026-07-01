# MindEye Scripts

This directory contains the orchestration scripts for the MindEye **ZUNA → QFormer → RAE** pipeline.
Doc index: [`../docs/README.md`](../docs/README.md). Architecture/roadmap: [`../docs/PLAN.md`](../docs/PLAN.md).

## 1. Data Preparation
* `download_nod.py`: Downloads continuous NOD dataset recordings from OpenNeuro.
* `sync_stimuli_s3_targeted.py`: Syncs ImageNet stimuli images from S3.

## 2. Signal Processing
* `run_zuna_batch.py`: Orchestrates the continuous ZUNA diffusion-based denoising pipeline on the raw `.fif` files.
* `run_cropper.py`: Crops the continuous files into semantic epochs, onset-back-aligned.
  * *Canonical Usage*: `python scripts/run_cropper.py --mode zuna --tmin -0.2 --tmax 1.0 --add-event-marker`

## 3. Multimodal Target Spaces
* `generate_clip_embeddings.py`: Generates 512-dim CLIP embeddings for stimuli images.
* `generate_image_semantics.py`: Uses a VLM (Qwen2-VL) to extract semantic captions from images.
* `generate_text_embeddings.py`: Generates text CLIP embeddings (from captions or ImageNet labels).
* `build_common_embeddings.py`: Fuses image/semantic/label signals into the CLIP `common` target (semantic baseline).
* `build_rae_latent_bank.py`: Builds the **RAE / DINOv2 embedding bank** — the primary reconstruction target.
* `generate_vlm_attributes.py`: Qwen2-VL JSON labels for 29 semantic attributes. See [`../docs/VLM_ATTRIBUTES.md`](../docs/VLM_ATTRIBUTES.md).
* `analyze_vlm_attributes.py`: Audit image coverage, per-attribute unclear %, missing calibration keys.

## 4. Core Training Pipeline (current — QFormer bridge)
* `cache_zuna_latents.py`: Caches ZUNA `post_mmd` latents (the QFormer input) to `data/processed/zuna_latents/`.
* `train_zuna_to_vision.py`: Trains the `ZunaToVisionQFormer` bridge (ZUNA latents → vision target). Supports `real / shuffled / random` target modes and onset-crop windowing.
* `run_qformer_grid.py`: Orchestrates the QFormer grid over target spaces (CLIP-Common-512, DINO-Unit-768, DINO-PCA-256/128) × control modes, then runs the 10,000-iter paired bootstrap.

## Canonical Execution Order (ZUNA → QFormer → RAE)
Run on the pod with `export PYTHONPATH=src`. See [`../docs/CHEAT.md`](../docs/CHEAT.md) and `cold_start.sh`.

```bash
# 1. Fetch data
python scripts/download_nod.py --subject sub-01 --runs 1-8
python scripts/sync_stimuli_s3_targeted.py

# 2. Process signals (ZUNA denoise + onset-aligned crop)
python scripts/run_zuna_batch.py --diffusion-steps 15
python scripts/run_cropper.py --mode zuna --tmin -0.2 --tmax 1.0 --add-event-marker \
    --runs 1 2 3 4 5 6 7 8 \
    --zuna-dir data/processed/zuna_real/4_fif_output \
    --output-dir data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_08

# 3. Build target spaces
python scripts/build_common_embeddings.py ...   # CLIP common (semantic baseline)
python scripts/build_rae_latent_bank.py \
    --image-dir data/raw/nod/stimuli/ImageNet \
    --output data/processed/rae_embeddings/rae_dinov2_base_sub01_04_runs01_40.pt

# 4. Cache ZUNA latents (QFormer input)
python scripts/cache_zuna_latents.py \
    --epochs-dir data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_08 \
    --output-dir data/processed/zuna_latents/sub01_runs01_08 \
    --layers post_mmd

# 5. Train + evaluate the QFormer bridge grid (real / shuffled / random + paired bootstrap)
python scripts/run_qformer_grid.py \
    --latents-pt data/processed/zuna_latents/sub01_runs01_08 \
    --clip-pt    data/processed/clip_embeddings/common_embeddings.pt \
    --rae-pt     data/processed/rae_embeddings/rae_dinov2_base_sub01_04_runs01_40.pt \
    --epochs 40 --patience 8 --batch-size 64 --lr 3e-4 \
    --device cuda --out-dir outputs/qformer_aligned_grid
```

## ⛔ Deprecated scripts (kept for reference, not the live plan)
See [`../docs/PLAN.md`](../docs/PLAN.md) §6 for the post-mortem.

* Decode_unit / unCLIP branch: `train_eeg_decode_common.py`, `build_decode_common_embeddings.py`, `evaluate_clip_native_decoder.py`, `pretrain_common_probe.py`, `run_baseline_matrix.py` (`make matrix`).
* RAE code-bottleneck branch: `train_rae_token_bottleneck.py`, `build_rae_bottleneck_codes.py`, `build_rae_code_stats.py`, `run_phase17_rae.sh`, `run_phase18*.sh`, `train_eeg_clip.py` (`--loss expander_aligned` / `spatial_cosine*`).

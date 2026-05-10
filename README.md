# Mind's Eye — ZUNA-first EEG→Semantic→Image

EEG-driven image generation using ZUNA as the signal normalization layer.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Pipeline

```
NOD-EEG continuous .fif
→ ZUNA denoise/reconstruct (5s windows @ 256Hz)
→ event-aligned crop (1.25s around stimulus)
→ EEG→CLIP semantic encoder
→ frozen diffusion img2img loop
```

## Quick Start

```bash
# 1. Download NOD-EEG subset (one subject, run 01 by default)
python scripts/download_nod.py

# Download runs 01-05 for the current ZUNA/cropper path
python scripts/download_nod.py --runs 1-5

# Later, after generating a targeted stimulus include list for CLIP:
python scripts/download_nod.py \
  --runs 1-5 \
  --include-list data/processed/clip_embeddings/openneuro_image_includes_sub01_runs01_05.txt

# 2. Run smoke test
python scripts/test_pipeline.py
```

## Project Structure

```
mindseye/
  src/mindseye/
    datasets/nod.py          — NOD-EEG loader (.fif + events + images)
    zuna/offline_pipeline.py  — batch ZUNA processing
    zuna/cropper.py           — event-aligned crop extraction
    zuna/montage.py           — channel coordinate handling
    embeddings/clip.py        — CLIP embedding computation
    models/eeg_encoder.py     — EEG→embedding encoder (TODO)
    train/train_eeg_clip.py   — contrastive training (TODO)
  scripts/
    download_nod.py           — download NOD-EEG from OpenNeuro
    run_zuna_batch.py         — real ZUNA batch runner; explicit resample-only baseline mode
    run_cropper.py            — event-aligned crop CLI
    generate_clip_embeddings.py — CLIP embedding generation / OpenNeuro image include list
    test_pipeline.py          — end-to-end smoke test
  vendor/ENIGMA/              — reference codebase
```

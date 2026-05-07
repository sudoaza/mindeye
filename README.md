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
# 1. Download NOD-EEG subset (one subject)
python scripts/download_nod.py

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
    embeddings/clip_embed.py  — CLIP embedding computation (TODO)
    models/eeg_encoder.py     — EEG→embedding encoder (TODO)
    train/train_eeg_clip.py   — contrastive training (TODO)
  scripts/
    download_nod.py           — download NOD-EEG from OpenNeuro
    test_pipeline.py          — end-to-end smoke test
  vendor/ENIGMA/              — reference codebase
```

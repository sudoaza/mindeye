# MindEye — ZUNA-first EEG→Semantic→Image

## Project thesis
EEG-driven image generation using ZUNA as the signal normalization layer. The core objective is to map real continuous EEG, cleaned by ZUNA, into the visual latent space (CLIP) to reconstruct what a subject is seeing.

## Current status
**MindEye v0.1-dev**: We are currently in the baseline generation and testing phase. We have established the pipeline for downloading NOD-EEG, processing via ZUNA, extracting semantic crops, and generating ground-truth CLIP targets. We are strictly requiring that our ZUNA-cleaned EEG crops retrieve the correct visual/semantic targets above shuffled controls before introducing diffusion.

*Note: The EEG-to-image system is NOT complete. We are currently building the ZUNA-aligned EEG→CLIP retrieval scaffold.*

## Installation
```bash
make setup
```

## Data requirements
This pipeline uses the NOD-EEG dataset (ds005811) from OpenNeuro, which contains continuous visual EEG recordings.

## Reproduce the current NOD-ZUNA baseline
```bash
make nod
make zuna
make crop
make clip
```

## Run timing audit
*Before any training runs, you must ensure event timing has not been corrupted.*
```bash
make audit
```

## Train EEG→CLIP baseline
```bash
make train
```

## Run baseline matrix
(Pending Matrix Script Implementation)

## Generate retrieval grids
```bash
make grid
```

## Planned architecture
- **NOD-EEG continuous .fif**
- **ZUNA** 5s denoise/reconstruct @ 256Hz
- **Event-aligned crop** 1.25s around stimulus
- **EEG→CLIP semantic encoder** mapping EEG latent space to CLIP embeddings
- **Frozen diffusion img2img loop**

## Known limitations
- Diffusion is not yet integrated.
- Only testing on `sub-01` currently.

## Paper/model comparison
This project draws inspiration from **ZUNA** (EEG foundation model) and **ENIGMA** (EEG-to-Image decoding). MindEye's focus is on establishing strict timing-aligned continuous processing pipelines before evaluating performance against these benchmarks.

# MindEye — ZUNA-first EEG→Semantic→Image

## Project Thesis
EEG-driven image generation using ZUNA as the signal normalization layer. The core objective is to map real continuous EEG, cleaned by ZUNA, into the multimodal latent space (`z_common`) to reconstruct what a subject is seeing.

## Current Status: Phase 12A (CLIP-Native Diffusion Decoder) 🚧
**MindEye v0.2-dev**: We have successfully completed the baseline evaluation matrix, retrieval index setup, and frozen diffusion image reconstruction. The pipeline now enforces a single canonical `z_common` target space with multi-subject FiLM scale/shift adapters and multi-task frozen probe heads. We are currently in Phase 12A, extracting target embeddings via a CLIP-Native decoder (`sd2-community/stable-diffusion-2-1-unclip`) to enable direct, unprompted image generation from decoded EEG embeddings.

## Project Structure
* `configs/`: Configuration files for datasets, ZUNA pipeline, and training.
* `data/`: Local storage for `raw/` NOD-EEG data and `processed/` features, crops, and embeddings. (Not tracked in git)
* `docs/`: Project documentation, including the comprehensive [`docs/PLAN.md`](docs/PLAN.md) detailing the phased roadmap.
* `outputs/`: Timestamped tracking for all baseline matrix runs, training logs, and checkpoints.
* `scripts/`: Modular orchestration scripts. See [`scripts/README.md`](scripts/README.md) for the detailed canonical execution order.
* `src/mindseye/`: Core library code (models, data loaders, evaluation, ZUNA offline wrappers).

## Installation
```bash
make setup
# OR manually:
# python3 -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt
```

## Reproducing the Canonical Pipeline
The end-to-end pipeline is fully automated via orchestration scripts. For detailed explanations of each step, refer to [`scripts/README.md`](scripts/README.md).

To run the full recovery pipeline (Downloads → ZUNA Denoising → Cropping → Common Embeddings → Matrix Training):
```bash
make pipeline
# OR manually:
# bash scripts/execute_recovery_v2.sh
```

## Methodology & Non-Negotiable Principles
1. **ZUNA-First**: All models are trained on continuous EEG data that has passed through ZUNA.
2. **Strict Controls**: We never evaluate "absolute" performance. Every run includes strict baseline controls (`zuna_shuffled`, `zuna_random`) to guard against dimensional collapse and dataset biases.
3. **No Premature Diffusion**: Diffusion image generation is locked until the semantic retrieval branch consistently beats shuffled/random baselines.

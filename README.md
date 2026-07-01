# MindEye — ZUNA-first EEG→Semantic→Image

## Project Thesis
EEG-driven image reconstruction with three best-in-class components: **ZUNA** as the frozen EEG foundation embedding (signal feature recovery), a learned **QFormer** bridge, and **RAE (DINOv2)** as the reconstruction target (visual fidelity). The core objective is to map real continuous EEG, embedded by ZUNA, into RAE's visual latent space to reconstruct what a subject is seeing.

## Current Status: QFormer Bridge (ZUNA → QFormer → RAE) 🚧
**MindEye v0.3-dev**: The architecture is three best-in-class components — **ZUNA** as the frozen EEG foundation embedding (best feature recovery), **RAE (DINOv2)** as the reconstruction target (best visual fidelity; CLIP/ViT was semantically ok but visually imprecise), and a learned **QFormer** as the bridge between them (simple adapters did not work). We cache ZUNA `post_mmd` latents, onset-crop them, and train the QFormer to map them into the RAE/DINO embedding space, ranking against the full image bank with mandatory `real / shuffled / random` controls and a paired bootstrap gate. The earlier RAE code-bottleneck path (Phase 18B–18E) was abandoned because squeezing EEG through a tiny `768×4×4` code discarded the per-site fidelity RAE needs. See [`docs/HANDOVER.md`](docs/HANDOVER.md) for the current architecture and run instructions.

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
1. **ZUNA-First**: All models are trained on continuous EEG data that has passed through ZUNA (frozen EEG foundation model). ZUNA and RAE are frozen; only the QFormer bridge trains.
2. **Strict Controls**: We never evaluate "absolute" performance. Every run includes strict baseline controls (`shuffled`, `random`) and a paired bootstrap to guard against dimensional collapse and dataset leakage. This split has repeatedly caught data/pipeline bugs.
3. **Full-set retrieval is the only honest metric**; within-val ranking is diagnostic only (inflated).
4. **No Premature Reconstruction**: The RAE decoder / diffusion image generation is locked until the QFormer retrieval gate (paired bootstrap Δ > +0.005, 95% CI excluding 0) is consistently met.

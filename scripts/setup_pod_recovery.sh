#!/bin/bash
# Setup script for mindeye recovery pod
set -e

# 1. Clone repository
cd /workspace
git clone https://github.com/sudoaza/mindeye.git
cd mindeye

# 2. Setup environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install mne pandas torch torchvision torchaudio transformers accelerate pyyaml matplotlib scipy tqdm scikit-learn

# 3. Create directories
mkdir -p data/raw/nod/derivatives/preprocessed/raw
mkdir -p data/raw/nod/stimuli/ImageNet
mkdir -p data/processed/clip_embeddings
mkdir -p data/processed/semantic_epochs

# 4. Download NOD data for sub-01 runs 01-10
# Note: This uses the existing download script which handles OpenNeuro
python scripts/download_nod.py --subject sub-01 --runs {1..10}

# 5. Generate Text Embeddings
# We'll use the metadata from the first few runs to get the class list
# Actually, we need to generate CLIP image embeddings too if they are missing
# But let's start with text.
python scripts/generate_text_embeddings.py --metadata data/raw/nod/derivatives/detailed_events/sub-01_events.csv --output data/processed/clip_embeddings/imagenet_text_embeddings.pt

# 6. Generate CLIP Image Embeddings for sub-01 runs 1-10
# (Assuming images are downloaded by download_nod.py --include-stimuli)
python scripts/generate_clip_embeddings.py --metadata data/raw/nod/derivatives/detailed_events/sub-01_events.csv --output data/processed/clip_embeddings/sub01_image_embeddings.pt

#!/usr/bin/env python3
"""Generate CLIP text embeddings for ImageNet classes in the NOD dataset.

Uses a set of templates to create a robust semantic representation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


TEMPLATES = [
    "a photo of a {label}",
    "an image of a {label}",
    "a natural image containing a {label}",
    "a visual stimulus showing a {label}",
]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--metadata",
        default="data/processed/semantic_epochs/zuna_real_sub01_runs01_05/all_runs_metadata.csv",
        help="Metadata CSV containing 'class' or 'class_id' column",
    )
    p.add_argument(
        "--output",
        default="data/processed/clip_embeddings/imagenet_text_embeddings.pt",
        help="Output .pt embedding table",
    )
    p.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    
    # Use the metadata to find all unique classes
    df = pd.read_csv(args.metadata)
    if "class" not in df.columns:
        raise ValueError("Metadata must contain a 'class' column (label)")
    
    unique_classes = sorted(df["class"].unique())
    print(f"Found {len(unique_classes)} unique classes.")

    from transformers import CLIPModel, CLIPProcessor
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    
    class_to_embedding = {}
    
    with torch.inference_mode():
        for label in unique_classes:
            # Generate embeddings for each template and average them
            prompts = [t.format(label=label) for t in TEMPLATES]
            inputs = processor(text=prompts, padding=True, return_tensors="pt").to(device)
            text_features = model.get_text_features(**inputs)
            # Normalize and average
            text_features = torch.nn.functional.normalize(text_features, dim=-1)
            mean_embedding = text_features.mean(dim=0)
            mean_embedding = torch.nn.functional.normalize(mean_embedding, dim=0)
            class_to_embedding[label] = mean_embedding.cpu()

    table = {
        "model_name": args.model_name,
        "class_to_embedding": class_to_embedding,
        "templates": TEMPLATES,
    }
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(table, args.output)
    print(f"Saved {len(class_to_embedding)} text embeddings to {args.output}")

if __name__ == "__main__":
    main()

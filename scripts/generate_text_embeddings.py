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
    p.add_argument("--source", choices=("templates", "image_semantics"), default="templates")
    p.add_argument("--semantics-jsonl", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    
    from transformers import CLIPModel, CLIPProcessor
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    
    if args.source == "templates":
        # Use the metadata to find all unique classes
        df = pd.read_csv(args.metadata)
        if "class" not in df.columns:
            raise ValueError("Metadata must contain a 'class' column (label)")
        
        unique_classes = sorted(df["class"].unique())
        print(f"Found {len(unique_classes)} unique classes.")
        
        class_to_embedding = {}
        
        with torch.inference_mode():
            for label in unique_classes:
                prompts = [t.format(label=label) for t in TEMPLATES]
                inputs = processor(text=prompts, padding=True, return_tensors="pt").to(device)
                text_features = model.get_text_features(**inputs)
                if not isinstance(text_features, torch.Tensor):
                    if hasattr(text_features, "text_embeds"):
                        text_features = text_features.text_embeds
                    elif hasattr(text_features, "pooler_output"):
                        text_features = text_features.pooler_output
                    else:
                        text_features = text_features[0]
                text_features = torch.nn.functional.normalize(text_features, dim=-1)
                mean_embedding = text_features.mean(dim=0)
                mean_embedding = torch.nn.functional.normalize(mean_embedding, dim=0)
                class_to_embedding[label] = mean_embedding.cpu()

        table = {
            "model_name": args.model_name,
            "class_to_embedding": class_to_embedding,
            "templates": TEMPLATES,
        }
    else:
        # source == image_semantics
        target_dicts = {
            "caption_short": {},
            "caption_detailed": {},
            "caption_composition": {},
            "caption_attributes": {},
            "caption_core": {},
        }
        image_id_to_caption_fields = {}
        
        with open(args.semantics_jsonl, "r") as f:
            lines = f.readlines()
            
        print(f"Found {len(lines)} items in {args.semantics_jsonl}")
        
        with torch.inference_mode():
            for line in lines:
                row = json.loads(line)
                image_id = row["image_id"]
                
                texts = {
                    "caption_short": row.get("short_caption", ""),
                    "caption_detailed": row.get("short_caption", "") + " " + row.get("detailed_caption", ""),
                    "caption_composition": row.get("composition_caption", ""),
                    "caption_attributes": row.get("attribute_caption", ""),
                    "caption_core": f"{row.get('short_caption', '')} {row.get('detailed_caption', '')} Composition: {row.get('composition_caption', '')} Attributes: {row.get('attribute_caption', '')}.",
                }
                
                for target_name, text in texts.items():
                    if not text.strip():
                        text = "empty"
                        
                    inputs = processor(text=[text], padding=True, return_tensors="pt", truncation=True, max_length=77).to(device)
                    text_features = model.get_text_features(**inputs)
                    if not isinstance(text_features, torch.Tensor):
                        if hasattr(text_features, "text_embeds"):
                            text_features = text_features.text_embeds
                        elif hasattr(text_features, "pooler_output"):
                            text_features = text_features.pooler_output
                        else:
                            text_features = text_features[0]
                    text_features = torch.nn.functional.normalize(text_features, dim=-1)[0]
                    
                    target_dicts[target_name][image_id] = text_features.cpu()
                
                caption_fields = {k: v for k, v in row.items() if k.endswith("_caption") or k in ["objects", "scene", "setting", "spatial_layout", "dominant_colors", "materials_textures", "lighting", "viewpoint", "action_or_state", "mood", "uncertainties"]}
                image_id_to_caption_fields[image_id] = caption_fields
                
        table = {
            "model": args.model_name,
            "source": "image_semantics",
            "image_id_to_caption_fields": image_id_to_caption_fields,
        }
        for k, v in target_dicts.items():
            table[f"image_id_to_{k}"] = v
            # Backwards compatibility for scripts expecting image_id_to_text_embedding
            if k == "caption_core":
                table["image_id_to_text_embedding"] = v
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(table, args.output)
    print(f"Saved text embeddings to {args.output}")

if __name__ == "__main__":
    main()

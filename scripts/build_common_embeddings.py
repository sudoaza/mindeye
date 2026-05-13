#!/usr/bin/env python3
"""Build a common multimodal embedding space by fusing image, text, and label embeddings.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import torch
import torch.nn.functional as F

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image-embeddings", required=True,
                   help="Path to .pt containing CLIP image embeddings")
    p.add_argument("--semantic-embeddings", required=True,
                   help="Path to .pt containing VLM semantic text embeddings")
    p.add_argument("--label-embeddings", default=None,
                   help="Optional path to .pt containing label text embeddings")
    p.add_argument("--metadata", required=True,
                   help="Path to metadata CSV to map classes to labels if using label-embeddings")
    p.add_argument("--w-img", type=float, default=0.25, help="Weight for image embedding")
    p.add_argument("--w-sem", type=float, default=0.75, help="Weight for semantic text embedding")
    p.add_argument("--output", required=True, help="Output .pt path")
    return p.parse_args()


def main():
    args = parse_args()
    
    print(f"Loading image embeddings from {args.image_embeddings}...")
    img_table = torch.load(args.image_embeddings, map_location="cpu")
    # sub01_image_embeddings.pt typically has {"image_id": [...], "embedding": [...]}
    if "embedding" in img_table and "image_id" in img_table:
        image_id_to_image = {
            str(img_id): F.normalize(img_table["embedding"][i].float(), dim=-1)
            for i, img_id in enumerate(img_table["image_id"])
        }
    else:
        # Fallback if it's already a dict
        image_id_to_image = {str(k): F.normalize(v.float(), dim=-1) for k, v in img_table.items()}
        
    print(f"Loading semantic embeddings from {args.semantic_embeddings}...")
    sem_table = torch.load(args.semantic_embeddings, map_location="cpu")
    if "image_id_to_semantic" not in sem_table:
        raise KeyError("Key 'image_id_to_semantic' not found in semantic embeddings file.")
    
    image_id_to_semantic = {
        str(k): F.normalize(v.float(), dim=-1) 
        for k, v in sem_table["image_id_to_semantic"].items()
    }
    
    image_id_to_label = {}
    if args.label_embeddings:
        print(f"Loading label embeddings from {args.label_embeddings}...")
        lbl_table = torch.load(args.label_embeddings, map_location="cpu")
        class_to_embedding = lbl_table.get("class_to_embedding", lbl_table)
        class_to_embedding = {str(k): F.normalize(v.float(), dim=-1) for k, v in class_to_embedding.items()}
        
        import pandas as pd
        print(f"Loading metadata from {args.metadata}...")
        df = pd.read_csv(args.metadata)
        
        for _, row in df.iterrows():
            img_id = str(row["image_id"])
            cls_name = str(row["class"])
            if cls_name in class_to_embedding:
                image_id_to_label[img_id] = class_to_embedding[cls_name]
    
    # Intersect keys
    common_keys = set(image_id_to_image.keys()) & set(image_id_to_semantic.keys())
    if args.label_embeddings:
        common_keys &= set(image_id_to_label.keys())
        
    print(f"Found {len(common_keys)} shared images across modalities.")
    
    image_id_to_common = {}
    for k in sorted(common_keys):
        img_emb = image_id_to_image[k]
        sem_emb = image_id_to_semantic[k]
        
        fused = args.w_img * img_emb + args.w_sem * sem_emb
            
        image_id_to_common[k] = F.normalize(fused, dim=-1)
        
    out_dict = {
        "model": "openai/clip-vit-base-patch32",
        "space": "common_clip_fused",
        "weights": {
            "image": args.w_img,
            "semantic": args.w_sem,
        },
        "image_id_to_common": image_id_to_common,
        "image_id_to_image": {k: image_id_to_image[k] for k in common_keys},
        "image_id_to_semantic": {k: image_id_to_semantic[k] for k in common_keys},
    }
    
    if args.label_embeddings:
        out_dict["image_id_to_label"] = {k: image_id_to_label[k] for k in common_keys}
        
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_dict, args.output)
    print(f"Saved fused common embeddings to {args.output}")


if __name__ == "__main__":
    main()

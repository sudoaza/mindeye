#!/usr/bin/env python3
"""Rebuild fused common embeddings for both sub-01 and sub-02, using non-mocked VLM attributes.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

TEMPLATES = [
    "a photo of a {label}",
    "an image of a {label}",
    "a natural image containing a {label}",
    "a visual stimulus showing a {label}",
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vlm-attributes", default="data/processed/vlm_attributes.json")
    p.add_argument("--metadata-sub01", default="data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv")
    p.add_argument("--metadata-sub02", default="data/processed/semantic_epochs/zuna_tight1s_sub02_runs01_32/all_runs_metadata.csv")
    p.add_argument("--image-embeddings", default="data/processed/clip_embeddings/combined_image_embeddings.pt")
    p.add_argument("--output-text-embeddings", default="data/processed/clip_embeddings/combined_image_semantic_text_embeddings.pt")
    p.add_argument("--output-common-embeddings", default="data/processed/clip_embeddings/combined_common_embeddings.pt")
    p.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    p.add_argument("--w-img", type=float, default=0.25)
    p.add_argument("--w-sem", type=float, default=0.75)
    p.add_argument("--device", default=None)
    return p.parse_args()

def construct_sentence(label: str, attrs: dict) -> str:
    # Build a clean descriptive sentence using attributes
    items = []
    
    is_anim = attrs.get("is_animate", "unclear")
    if is_anim == "yes":
        items.append("animate")
    elif is_anim == "no":
        items.append("inanimate")
        
    io = attrs.get("indoor_outdoor", "unclear")
    if io in ("indoor", "outdoor", "mixed"):
        items.append(f"{io}")
        
    na = attrs.get("natural_artificial", "unclear")
    if na in ("natural", "artificial", "mixed"):
        items.append(f"{na}")
        
    sd = attrs.get("scene_dominance", "unclear")
    if sd in ("isolated_object", "object_with_background", "full_scene"):
        items.append(sd.replace("_", " "))
        
    sz = attrs.get("real_world_size", "unclear")
    if sz in ("tiny", "small", "medium", "large", "huge"):
        items.append(f"{sz} size")
        
    # Join items
    if items:
        attrs_str = ", ".join(items)
        return f"a photo of a {label}. Attributes: {attrs_str}."
    else:
        return f"a photo of a {label}."

def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading VLM attributes from {args.vlm_attributes}...")
    with open(args.vlm_attributes, "r") as f:
        vlm_attrs = json.load(f)
        
    print("Loading metadata CSVs...")
    df_s1 = pd.read_csv(args.metadata_sub01)
    df_s2 = pd.read_csv(args.metadata_sub02)
    df = pd.concat([df_s1, df_s2], ignore_index=True)
    
    unique_images = df[["image_id", "class"]].drop_duplicates().reset_index(drop=True)
    print(f"Found {len(unique_images)} unique images in combined metadata.")
    
    # Check coverage in vlm_attrs
    missing_count = 0
    for img_id in unique_images["image_id"]:
        if str(img_id) not in vlm_attrs:
            missing_count += 1
    if missing_count > 0:
        print(f"Warning: {missing_count} images missing from VLM attributes. Will use fallback attributes.")
        
    # Generate captions
    image_id_to_caption = {}
    image_id_to_caption_fields = {}
    for _, row in unique_images.iterrows():
        img_id = str(row["image_id"])
        label = str(row["class"])
        attrs = vlm_attrs.get(img_id, {})
        caption = construct_sentence(label, attrs)
        image_id_to_caption[img_id] = caption
        
        # Save caption fields for compatibility
        image_id_to_caption_fields[img_id] = {
            "image_path": f"data/raw/nod/stimuli/ImageNet/{img_id}.JPEG",
            "class_label": label,
            "vlm_model": "Qwen/Qwen2-VL-7B-Instruct",
            "embedding_text": caption,
            **attrs
        }
        
    print(f"Loading CLIP text encoder: {args.model_name}...")
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    
    image_id_to_semantic = {}
    print("Generating CLIP text embeddings for each visual stimulus...")
    with torch.inference_mode():
        for img_id, text in tqdm(image_id_to_caption.items()):
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
            image_id_to_semantic[img_id] = text_features.cpu()
            
    print(f"Saving text embeddings table to {args.output_text_embeddings}...")
    table = {
        "model": args.model_name,
        "source": "image_semantics",
        "image_id_to_semantic": image_id_to_semantic,
        "image_id_to_caption_fields": image_id_to_caption_fields,
    }
    Path(args.output_text_embeddings).parent.mkdir(parents=True, exist_ok=True)
    torch.save(table, args.output_text_embeddings)
    
    # Build fused common embeddings
    print(f"Loading image embeddings from {args.image_embeddings}...")
    img_table = torch.load(args.image_embeddings, map_location="cpu")
    if "embedding" in img_table and "image_id" in img_table:
        image_id_to_image = {
            str(img_id): F.normalize(img_table["embedding"][i].float(), dim=-1)
            for i, img_id in enumerate(img_table["image_id"])
        }
    else:
        image_id_to_image = {str(k): F.normalize(v.float(), dim=-1) for k, v in img_table.items()}
        
    # Class label embeddings using templates
    print("Generating template-based class label embeddings...")
    class_to_embedding = {}
    unique_classes = sorted(df["class"].unique())
    with torch.inference_mode():
        for label in tqdm(unique_classes):
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
            
    image_id_to_label = {}
    for _, row in df.iterrows():
        img_id = str(row["image_id"])
        cls_name = str(row["class"])
        if cls_name in class_to_embedding:
            image_id_to_label[img_id] = class_to_embedding[cls_name]
            
    # Intersect keys
    common_keys = set(image_id_to_image.keys()) & set(image_id_to_semantic.keys())
    print(f"Fusing {len(common_keys)} shared images into common space...")
    
    image_id_to_common = {}
    for k in sorted(common_keys):
        img_emb = image_id_to_image[k]
        sem_emb = image_id_to_semantic[k]
        fused = args.w_img * img_emb + args.w_sem * sem_emb
        image_id_to_common[k] = F.normalize(fused, dim=-1)
        
    out_dict = {
        "model": args.model_name,
        "space": "common_clip_fused",
        "weights": {
            "image": args.w_img,
            "semantic": args.w_sem,
        },
        "image_id_to_common": image_id_to_common,
        "image_id_to_image": {k: image_id_to_image[k] for k in common_keys},
        "image_id_to_semantic": {k: image_id_to_semantic[k] for k in common_keys},
        "image_id_to_label": {k: image_id_to_label[k] for k in common_keys},
    }
    
    torch.save(out_dict, args.output_common_embeddings)
    print(f"Successfully saved fused common embeddings to {args.output_common_embeddings}!")

if __name__ == "__main__":
    main()

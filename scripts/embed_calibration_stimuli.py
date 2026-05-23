#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image

def parse_args():
    p = argparse.ArgumentParser(description="Embed visual calibration stimuli into z_common")
    p.add_argument("--metadata", default="data/stimuli/calibration/calibration_metadata.csv", help="Path to calibration metadata CSV")
    p.add_argument("--stimuli-root", default="data/stimuli/calibration", help="Root folder of calibration stimuli")
    p.add_argument("--common-embeddings", default="data/processed/clip_embeddings/common_embeddings.pt", help="Path to common_embeddings.pt")
    p.add_argument("--model-name", default="openai/clip-vit-base-patch32", help="CLIP model name")
    p.add_argument("--device", default=None, help="cuda, cpu, or auto")
    return p.parse_args()

def build_prompt(row):
    stim_type = row["stimulus_type"]
    if stim_type == "color_patch":
        return f"A visual stimulus showing a solid {row['dominant_color']} patch on a gray background. Color is {row['warm_vs_cool']}."
    elif stim_type == "shape":
        return f"A visual stimulus showing a centered gray {row['shape']} shape on a gray background. Shape is {row['round_or_curved']} round and {row['angular_or_geometric']} angular."
    elif stim_type == "texture":
        desc = []
        if row["organic_texture"] == "yes":
            desc.append("organic texture")
        else:
            desc.append("synthetic texture")
        if row["soft_texture"] == "yes":
            desc.append("soft texture")
        if row["furry"] == "yes":
            desc.append("furry texture")
        if row["metallic"] == "yes":
            desc.append("metallic surface")
        if row["rough"] == "yes":
            desc.append("rough surface")
        if row["smooth"] == "yes":
            desc.append("smooth surface")
        return f"A visual stimulus showing a {', '.join(desc)}."
    elif stim_type == "spatial":
        return f"A visual stimulus showing a small white dot target located on the {row['main_subject_position_x']} side."
    elif stim_type == "animacy":
        desc = []
        if row["is_animate"] == "yes":
            desc.append("animate subject")
        else:
            desc.append("inanimate subject")
        if row["face_visible"] == "yes":
            if row["animal_visible"] == "yes":
                desc.append("animal face")
            else:
                desc.append("human face")
        elif row["animal_visible"] == "yes":
            desc.append("animal body")
        elif row["is_animate"] == "yes":
            desc.append("human body")
        else:
            desc.append("object")
        return f"A visual stimulus showing an image containing a {', '.join(desc)}."
    return "A visual calibration stimulus."

def coerce_features(features, model, is_text=False) -> torch.Tensor:
    if isinstance(features, torch.Tensor):
        return features
    if hasattr(features, "image_embeds") and isinstance(features.image_embeds, torch.Tensor):
        return features.image_embeds
    if hasattr(features, "text_embeds") and isinstance(features.text_embeds, torch.Tensor):
        return features.text_embeds
    if hasattr(features, "pooler_output") and isinstance(features.pooler_output, torch.Tensor):
        pooled = features.pooler_output
        projection = getattr(model, "text_projection" if is_text else "visual_projection", None)
        if projection is not None and pooled.shape[-1] == getattr(projection, "in_features", None):
            return projection(pooled)
        return pooled
    if isinstance(features, (list, tuple)) and len(features) > 0:
        return coerce_features(features[0], model, is_text)
    raise TypeError(f"Unsupported feature type: {type(features)}")

def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Embedder] Using device: {device}")
    
    metadata_path = Path(args.metadata)
    stim_root = Path(args.stimuli_root)
    embs_path = Path(args.common_embeddings)
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Calibration metadata CSV not found at {metadata_path}")
    if not embs_path.exists():
        raise FileNotFoundError(f"Common embeddings file not found at {embs_path}")
        
    print(f"[Embedder] Loading calibration metadata from {metadata_path}...")
    rows = []
    with open(metadata_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            
    print(f"[Embedder] Loading CLIP model {args.model_name}...")
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model.eval()
    
    subfolder_map = {
        "color_patch": "color_patches",
        "shape": "shapes",
        "texture": "textures",
        "spatial": "spatial",
        "animacy": "animacy"
    }
    
    print("[Embedder] Computing CLIP embeddings for calibration stimuli...")
    image_id_to_common = {}
    image_id_to_image = {}
    image_id_to_semantic = {}
    
    with torch.no_grad():
        for i, row in enumerate(rows, 1):
            img_id = row["image_id"]
            stim_type = row["stimulus_type"]
            subfolder = subfolder_map.get(stim_type)
            if not subfolder:
                print(f"[WARN] Unknown stimulus type '{stim_type}', skipping.")
                continue
                
            img_file = stim_root / subfolder / f"{img_id}.jpg"
            if not img_file.exists():
                print(f"[WARN] Image file {img_file} not found, skipping.")
                continue
                
            # Load and process image
            img = Image.open(img_file).convert("RGB")
            inputs_img = processor(images=img, return_tensors="pt").to(device)
            raw_img_emb = model.get_image_features(**inputs_img)
            img_emb = coerce_features(raw_img_emb, model, is_text=False)
            img_emb = F.normalize(img_emb, dim=-1)[0]
            
            # Construct and process prompt
            prompt = build_prompt(row)
            inputs_txt = processor(text=[prompt], padding=True, return_tensors="pt", truncation=True, max_length=77).to(device)
            raw_txt_emb = model.get_text_features(**inputs_txt)
            txt_emb = coerce_features(raw_txt_emb, model, is_text=True)
            txt_emb = F.normalize(txt_emb, dim=-1)[0]
            
            # Fuse embeddings (0.25 image, 0.75 text)
            fused = 0.25 * img_emb + 0.75 * txt_emb
            fused = F.normalize(fused, dim=-1)
            
            image_id_to_image[img_id] = img_emb.cpu()
            image_id_to_semantic[img_id] = txt_emb.cpu()
            image_id_to_common[img_id] = fused.cpu()
            
            if i % 10 == 0 or i == len(rows):
                print(f"  Processed {i}/{len(rows)} calibration stimuli.")

                
    print(f"[Embedder] Loading existing common embeddings from {embs_path}...")
    common_data = torch.load(embs_path, map_location="cpu")
    
    # Update dicts
    common_data["image_id_to_common"].update(image_id_to_common)
    common_data["image_id_to_image"].update(image_id_to_image)
    common_data["image_id_to_semantic"].update(image_id_to_semantic)
    
    # If image_id_to_label is present, add dummy entries for calibration image_ids to avoid KeyError
    if "image_id_to_label" in common_data:
        for img_id in image_id_to_common.keys():
            # Set dummy class embedding (zeros) or skip
            common_data["image_id_to_label"][img_id] = torch.zeros_like(next(iter(image_id_to_common.values())))
            
    print(f"[Embedder] Saving updated common embeddings back to {embs_path}...")
    torch.save(common_data, embs_path)
    print("[Embedder] Calibration stimuli successfully embedded into common space!")

if __name__ == "__main__":
    main()

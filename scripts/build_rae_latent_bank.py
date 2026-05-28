#!/usr/bin/env python3
import argparse
import sys
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindseye.generation.rae_backend import RaeDecoderBackend

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", default="data/raw/nod/stimuli/ImageNet", help="Path to ImageNet stimulus images")
    p.add_argument("--output", default="data/processed/rae_embeddings/rae_dinov2_base_sub01_04_runs01_40.pt", help="Output path")
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize backend
    backend = RaeDecoderBackend(device=device, apply_patch=True)
    backend.load()
    
    image_dir = Path(args.image_dir)
    image_paths = sorted(
        list(image_dir.rglob("*.png")) + 
        list(image_dir.rglob("*.jpg")) + 
        list(image_dir.rglob("*.JPEG")) +
        list(image_dir.rglob("*.jpeg"))
    )
    print(f"Found {len(image_paths)} images in {image_dir}")
    
    image_id_to_rae_tokens = {}
    image_id_to_rae_global = {}
    image_id_to_rae_unit = {}
    image_id_to_rae_global_norm = {}
    
    # Process in batches
    for i in tqdm(range(0, len(image_paths), args.batch_size), desc="Extracting RAE Embeddings"):
        batch_paths = image_paths[i:i+args.batch_size]
        
        images = []
        valid_paths = []
        for p in batch_paths:
            try:
                img = Image.open(p)
                img = img.convert("RGB")
                img.load()
                images.append(img)
                valid_paths.append(p)
            except Exception as e:
                print(f"\nWarning: Failed to load image {p}: {e}")
                
        if not images:
            continue
            
        # Extract RAE latents
        latents = backend.extract_rae_latent(images)
        
        tokens_batch = latents["tokens"]  # [B, 768, 16, 16]
        global_batch = latents["global"]  # [B, 768]
        unit_batch = latents["unit"]      # [B, 768]
        norm_batch = latents["norm"]      # [B]
        
        for path, tokens, glb, unt, nrm in zip(valid_paths, tokens_batch, global_batch, unit_batch, norm_batch):
            img_id = path.stem
            # Save tokens in float16 to save storage/memory, keeping others in float32
            image_id_to_rae_tokens[img_id] = tokens.half().cpu()
            image_id_to_rae_global[img_id] = glb.float().cpu()
            image_id_to_rae_unit[img_id] = unt.float().cpu()
            image_id_to_rae_global_norm[img_id] = nrm.float().cpu()

    out_dict = {
        "model": backend.model_id,
        "space": "rae_unit",
        "image_id_to_rae_tokens": image_id_to_rae_tokens,
        "image_id_to_rae_global": image_id_to_rae_global,
        "image_id_to_rae_unit": image_id_to_rae_unit,
        "image_id_to_rae_global_norm": image_id_to_rae_global_norm
    }
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_dict, out_path)
    
    print(f"Saved RAE latent bank with {len(image_id_to_rae_tokens)} images to {out_path}")

if __name__ == "__main__":
    main()

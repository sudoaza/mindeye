#!/usr/bin/env python3
"""
Build target embeddings using the exact Stable unCLIP image encoder.
Saves unnormalized embeddings to be used as targets for EEG decoding.
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindseye.generation.clip_native_backend import ClipNativeDecoderBackend

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", default="data/raw/nod/stimuli/ImageNet", help="Path to stimulus images")
    p.add_argument("--output", default="data/processed/clip_embeddings/decode_common_embeddings.pt", help="Output path")
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()

def main():
    args = parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading ClipNativeDecoderBackend...")
    backend = ClipNativeDecoderBackend(device=device)
    
    image_dir = Path(args.image_dir)
    image_paths = sorted(list(image_dir.rglob("*.png")) + list(image_dir.rglob("*.jpg")) + list(image_dir.rglob("*.JPEG")))
    print(f"Found {len(image_paths)} images in {image_dir}")
    
    image_id_to_common = {}
    
    # Process in batches
    with torch.inference_mode():
        for i in range(0, len(image_paths), args.batch_size):
            batch_paths = image_paths[i:i+args.batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            
            # Use normalize=False because Gate 1 showed raw embeddings work, normalized fall to chance.
            embeds = backend.extract_teacher_embeds(images, normalize=False)
            
            for path, embed in zip(batch_paths, embeds):
                img_id = path.stem
                # Move to CPU to save memory
                image_id_to_common[img_id] = embed.cpu()
                
            print(f"Processed {i+len(batch_paths)}/{len(image_paths)} images...")

    out_dict = {
        "model": backend.model_id,
        "space": "decode_common",
        "normalization": "unnormalized",
        "image_id_to_common": image_id_to_common
    }
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_dict, out_path)
    
    print(f"Saved {len(image_id_to_common)} embeddings to {out_path}")

if __name__ == "__main__":
    main()

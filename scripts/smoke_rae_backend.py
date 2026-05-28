#!/usr/bin/env python3
import sys
import os
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindseye.generation.rae_backend import RaeDecoderBackend

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize RAE Backend
    backend = RaeDecoderBackend(device=device, apply_patch=True)
    backend.load()
    
    # Create output directory
    output_dir = Path("outputs/rae_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find or create a test image
    stimuli_dir = Path("data/raw/nod/stimuli/ImageNet")
    test_img = None
    if stimuli_dir.exists():
        img_paths = list(stimuli_dir.rglob("*.JPEG")) + list(stimuli_dir.rglob("*.png")) + list(stimuli_dir.rglob("*.jpg"))
        if img_paths:
            print(f"Found test image: {img_paths[0]}")
            test_img = Image.open(img_paths[0]).convert("RGB").resize((256, 256))
            
    if test_img is None:
        print("No image found in stimuli. Creating a synthetic test image.")
        test_img = Image.new("RGB", (256, 256), color="teal")
        draw = ImageDraw.Draw(test_img)
        draw.ellipse([50, 50, 200, 200], fill="gold", outline="orange")
        draw.rectangle([100, 100, 150, 150], fill="crimson")
        
    test_img.save(output_dir / "target.png")
    print("Saved target.png")
    
    # Extract latents
    print("Extracting latents...")
    latents = backend.extract_rae_latent(test_img)
    tokens = latents["tokens"]
    global_embed = latents["global"]
    unit_embed = latents["unit"]
    norm_val = latents["norm"]
    
    print(f"RAE Latent Shape (tokens): {tokens.shape}")
    print(f"Global Embedding Shape: {global_embed.shape}")
    print(f"Unit Embedding Shape: {unit_embed.shape}")
    print(f"Norm Value: {norm_val.item()}")
    
    # Decode back to image (Oracle reconstruction)
    print("Decoding oracle reconstruction...")
    oracle_imgs = backend.generate_from_embeds(tokens)
    oracle_img = oracle_imgs[0]
    oracle_img.save(output_dir / "oracle_recon.png")
    print("Saved oracle_recon.png")
    
    # Decode random latent (Noise)
    print("Decoding random noise...")
    random_tokens = torch.randn_like(tokens)
    random_imgs = backend.generate_from_embeds(random_tokens)
    random_img = random_imgs[0]
    random_img.save(output_dir / "random_decode.png")
    print("Saved random_decode.png")
    
    # Extract latents from reconstruction to compute RAE-native oracle cosine similarity
    print("Extracting latents from oracle reconstruction...")
    oracle_latents = backend.extract_rae_latent(oracle_img)
    oracle_unit = oracle_latents["unit"]
    
    rae_native_cosine = F.cosine_similarity(unit_embed, oracle_unit, dim=-1).item()
    print(f"RAE-Native Oracle Cosine Similarity: {rae_native_cosine:.5f}")
    
    # Save smoke metrics
    metrics = {
        "rae_latent_shape": list(tokens.shape),
        "global_embedding_shape": list(global_embed.shape),
        "norm_value": norm_val.item(),
        "rae_native_oracle_cosine": rae_native_cosine
    }
    
    with open(output_dir / "smoke_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved smoke_metrics.json")
    print("Smoke test completed successfully!")

if __name__ == "__main__":
    main()

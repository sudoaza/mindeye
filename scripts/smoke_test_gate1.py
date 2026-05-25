#!/usr/bin/env python3
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindseye.generation.clip_native_backend import ClipNativeDecoderBackend

def main():
    print("Loading ClipNativeDecoderBackend (small)...")
    backend = ClipNativeDecoderBackend(
        model_id="sd2-community/stable-diffusion-2-1-unclip-small",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Load 8 targets
    stimuli_dir = Path("data/processed/stimuli")
    if not stimuli_dir.exists():
        print(f"Directory {stimuli_dir} does not exist. Using dummy images for testing.")
        images = [Image.new("RGB", (512, 512), color=(i*30, 255-i*30, i*15)) for i in range(8)]
    else:
        image_paths = list(stimuli_dir.glob("*.jpg")) + list(stimuli_dir.glob("*.png"))
        image_paths = image_paths[:8]
        if not image_paths:
             print("No images found in stimuli dir, using dummies.")
             images = [Image.new("RGB", (512, 512), color=(i*30, 255-i*30, i*15)) for i in range(8)]
        else:
             images = [Image.open(p).convert("RGB") for p in image_paths]
        
    print(f"Loaded {len(images)} images.")
    
    # 1. Extract raw and normalized embeddings
    print("Extracting teacher embeddings...")
    raw_embeds = backend.extract_teacher_embeds(images, normalize=False)
    norm_embeds = backend.extract_teacher_embeds(images, normalize=True)
    
    print(f"Extracted shape: {raw_embeds.shape}")
    
    # 2. Random embeddings
    torch.manual_seed(42)
    random_embeds_raw = torch.randn_like(raw_embeds)
    random_embeds_norm = F.normalize(torch.randn_like(raw_embeds), dim=-1)
    
    # Generate from oracle (raw)
    print("Generating from oracle (raw) embeds...")
    oracle_raw_images = backend.generate_from_embeds(raw_embeds, num_inference_steps=20)
    
    # Generate from oracle (norm)
    print("Generating from oracle (norm) embeds...")
    oracle_norm_images = backend.generate_from_embeds(norm_embeds, num_inference_steps=20)
    
    # Generate from random
    print("Generating from random embeds...")
    random_images = backend.generate_from_embeds(random_embeds_raw, num_inference_steps=20)
    
    # Evaluate via cosine similarity of the generated images back to the original targets
    print("Re-extracting generated images...")
    oracle_raw_reextracted = backend.extract_teacher_embeds(oracle_raw_images, normalize=True)
    oracle_norm_reextracted = backend.extract_teacher_embeds(oracle_norm_images, normalize=True)
    random_reextracted = backend.extract_teacher_embeds(random_images, normalize=True)
    
    target_embeds = backend.extract_teacher_embeds(images, normalize=True)
    
    # Compute Cosine Similarities
    def mean_cosine(a, b):
        return F.cosine_similarity(a, b, dim=-1).mean().item()
        
    cos_raw = mean_cosine(oracle_raw_reextracted, target_embeds)
    cos_norm = mean_cosine(oracle_norm_reextracted, target_embeds)
    cos_rand = mean_cosine(random_reextracted, target_embeds)
    
    print(f"\n--- Gate 1 Results ---")
    print(f"Oracle (Raw Embeds) Cosine: {cos_raw:.4f}")
    print(f"Oracle (Norm Embeds) Cosine: {cos_norm:.4f}")
    print(f"Random Embeds Cosine:       {cos_rand:.4f}")
    
    # Pass condition: Oracle > Random + 0.05
    best_oracle = max(cos_raw, cos_norm)
    
    if best_oracle > cos_rand + 0.05:
        print(f"\nSUCCESS: Oracle ({best_oracle:.4f}) > Random ({cos_rand:.4f}) + 0.05")
    else:
        print(f"\nFAILURE: Oracle ({best_oracle:.4f}) is NOT significantly better than Random ({cos_rand:.4f})")

if __name__ == "__main__":
    main()

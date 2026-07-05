#!/usr/bin/env python3
import argparse
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindseye.generation.rae_backend import RaeDecoderBackend

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", default="data/raw/nod/stimuli/ImageNet", help="Path to ImageNet stimulus images")
    p.add_argument("--output", default="data/processed/rae_embeddings/rae_dinov2_base_bank.pt", help="Output path")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dino-model-id", default="facebook/dinov2-base",
                   help="DINOv2 model id for the CLS-token target space (image-identity vector).")
    return p.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize backend
    backend = RaeDecoderBackend(device=device, apply_patch=True)
    backend.load()

    # DINOv2 CLS encoder: the RAE `unit` target is mean-pooled (spatial detail is
    # averaged away), which produces a diffuse retrieval signal. The DINOv2 CLS token
    # is the image-identity vector, kept as an additional target space (dino_cls).
    from transformers import AutoModel, AutoImageProcessor
    print(f"Loading DINOv2 CLS encoder {args.dino_model_id}...")
    dino_processor = AutoImageProcessor.from_pretrained(args.dino_model_id)
    dino_model = AutoModel.from_pretrained(args.dino_model_id).to(device).eval()
    for p_ in dino_model.parameters():
        p_.requires_grad = False
    
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
    image_id_to_dino_cls = {}
    
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

        # Extract DINOv2 CLS token (L2-normalized): last_hidden_state[:, 0]
        with torch.inference_mode():
            dino_inputs = dino_processor(images=images, return_tensors="pt").to(device)
            dino_out = dino_model(**dino_inputs)
            dino_cls_batch = dino_out.last_hidden_state[:, 0]  # [B, 768]
            dino_cls_batch = F.normalize(dino_cls_batch.float(), dim=-1)
        
        for path, tokens, glb, unt, nrm, dcls in zip(
            valid_paths, tokens_batch, global_batch, unit_batch, norm_batch, dino_cls_batch
        ):
            img_id = path.stem
            # Save tokens in float16 to save storage/memory, keeping others in float32
            image_id_to_rae_tokens[img_id] = tokens.half().cpu()
            image_id_to_rae_global[img_id] = glb.float().cpu()
            image_id_to_rae_unit[img_id] = unt.float().cpu()
            image_id_to_rae_global_norm[img_id] = nrm.float().cpu()
            image_id_to_dino_cls[img_id] = dcls.float().cpu()

    out_dict = {
        "model": backend.model_id,
        "dino_cls_model": args.dino_model_id,
        "space": "rae_unit",
        "image_id_to_rae_tokens": image_id_to_rae_tokens,
        "image_id_to_rae_global": image_id_to_rae_global,
        "image_id_to_rae_unit": image_id_to_rae_unit,
        "image_id_to_rae_global_norm": image_id_to_rae_global_norm,
        "image_id_to_dino_cls": image_id_to_dino_cls,
    }
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_dict, out_path)
    
    print(f"Saved RAE latent bank with {len(image_id_to_rae_tokens)} images "
          f"({len(image_id_to_dino_cls)} DINOv2 CLS vectors) to {out_path}")

if __name__ == "__main__":
    main()

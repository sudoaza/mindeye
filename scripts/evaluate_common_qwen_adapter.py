import os
import torch
import argparse
from PIL import Image
import pandas as pd
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from mindseye.models.common_to_qwen_adapter import CommonToQwenAdapter
from mindseye.generation.qwen_backend import QwenBackend

def build_grid(target_img, oracle_img, rand_img, nn_img):
    """
    Creates a simple horizontal grid of images.
    """
    w, h = target_img.size
    grid = Image.new('RGB', (w * 4, h))
    grid.paste(target_img, (0, 0))
    grid.paste(oracle_img, (w, 0))
    grid.paste(nn_img, (w * 2, 0))
    grid.paste(rand_img, (w * 3, 0))
    return grid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-ckpt", required=True)
    parser.add_argument("--common-embeddings", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--stimuli-root", required=True)
    parser.add_argument("--qwen-model", default="Qwen/Qwen-Image")
    parser.add_argument("--num-tokens", type=int, default=256)
    parser.add_argument("--adapter-dim", type=int, default=2048)
    parser.add_argument("--watermark", action="store_true", default=True, help="Watermark output images")
    parser.add_argument("--no-watermark", action="store_false", dest="watermark", help="Disable watermarking")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading Qwen Backend...")
    backend = QwenBackend(model_id=args.qwen_model, device="cuda")
    
    print("Loading CommonToQwenAdapter...")
    adapter = CommonToQwenAdapter(
        common_dim=512,
        adapter_dim=args.adapter_dim,
        num_tokens=args.num_tokens,
        qwen_hidden_dim=backend.qwen_hidden_dim
    ).to("cuda")
    adapter.load_state_dict(torch.load(args.adapter_ckpt, map_location="cuda"))
    adapter.eval()
    
    print("Loading embeddings...")
    common_embeddings = torch.load(args.common_embeddings, map_location="cpu")
    metadata = pd.read_csv(args.metadata).drop_duplicates(subset=["image_id"]).reset_index(drop=True)
    
    # Evaluate on the first 5 images
    metadata = metadata.head(5)
    
    all_z = torch.stack(list(common_embeddings.values()))
    
    for idx, row in metadata.iterrows():
        image_id = row['image_id']
        image_path = os.path.join(args.stimuli_root, row.get('image_path', f"{image_id}.jpg"))
        
        target_img = Image.open(image_path).convert("RGB").resize((512, 512))
        
        oracle_z = common_embeddings[image_id].to("cuda", dtype=torch.float32).unsqueeze(0)
        
        # random baseline
        rand_z = torch.randn_like(oracle_z)
        
        # nearest neighbor baseline
        dists = torch.cdist(oracle_z.cpu(), all_z)
        # 0 is itself, 1 is nearest neighbor
        nn_idx = dists.argsort(dim=1)[0][1].item()
        nn_z = all_z[nn_idx].to("cuda", dtype=torch.float32).unsqueeze(0)
        
        print(f"Generating for {image_id}...")
        
        with torch.no_grad():
            oracle_embeds = adapter(oracle_z)
            rand_embeds = adapter(rand_z)
            nn_embeds = adapter(nn_z)
            
            oracle_gen = backend.generate_from_embeds(oracle_embeds, height=512, width=512, watermark=args.watermark)[0]
            rand_gen = backend.generate_from_embeds(rand_embeds, height=512, width=512, watermark=args.watermark)[0]
            nn_gen = backend.generate_from_embeds(nn_embeds, height=512, width=512, watermark=args.watermark)[0]
            
        grid = build_grid(target_img, oracle_gen, rand_gen, nn_gen)
        grid_path = os.path.join(args.output_dir, f"grid_{image_id}.jpg")
        grid.save(grid_path)
        print(f"Saved {grid_path}")

if __name__ == "__main__":
    main()

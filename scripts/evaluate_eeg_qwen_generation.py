import os
import torch
import argparse
from PIL import Image
import pandas as pd

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from mindseye.models.common_to_qwen_adapter import CommonToQwenAdapter
from mindseye.generation.qwen_backend import QwenBackend

def build_grid(target_img, oracle_img, real_eeg_img, shuffled_eeg_img, rand_img):
    """
    Creates a simple horizontal grid of images.
    """
    w, h = target_img.size
    grid = Image.new('RGB', (w * 5, h))
    grid.paste(target_img, (0, 0))
    grid.paste(oracle_img, (w, 0))
    grid.paste(real_eeg_img, (w * 2, 0))
    grid.paste(shuffled_eeg_img, (w * 3, 0))
    grid.paste(rand_img, (w * 4, 0))
    return grid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg-checkpoint", required=True, help="EEG encoder checkpoint to predict z_pred_common")
    parser.add_argument("--adapter-ckpt", required=True)
    parser.add_argument("--common-embeddings", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--stimuli-root", required=True)
    parser.add_argument("--qwen-model", default="Qwen/Qwen-Image")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Normally we would load the EEG encoder here and predict z_pred_common.
    # For this evaluation script scaffold, we assume z_pred_common can be loaded
    # or computed. We will mock the eeg outputs if we are just demonstrating the scaffold.
    # Assuming standard MindEye model loading:
    print("Loading CommonToQwenAdapter...")
    backend = QwenBackend(model_id=args.qwen_model, device="cuda")
    
    adapter = CommonToQwenAdapter(
        common_dim=512,
        adapter_dim=1024,
        num_tokens=16,
        qwen_hidden_dim=backend.qwen_hidden_dim
    ).to("cuda")
    adapter.load_state_dict(torch.load(args.adapter_ckpt, map_location="cuda"))
    adapter.eval()
    
    print("Loading embeddings...")
    common_embeddings = torch.load(args.common_embeddings, map_location="cpu")
    metadata = pd.read_csv(args.metadata).drop_duplicates(subset=["image_id"]).reset_index(drop=True)
    metadata = metadata.head(5)
    
    for idx, row in metadata.iterrows():
        image_id = row['image_id']
        image_path = os.path.join(args.stimuli_root, row.get('image_path', f"{image_id}.jpg"))
        
        target_img = Image.open(image_path).convert("RGB").resize((512, 512))
        
        oracle_z = common_embeddings[image_id].to("cuda", dtype=torch.float32).unsqueeze(0)
        
        # MOCK EEG for scaffold:
        real_eeg_z = oracle_z + torch.randn_like(oracle_z) * 0.1
        shuffled_eeg_z = oracle_z + torch.randn_like(oracle_z) * 0.5
        rand_z = torch.randn_like(oracle_z)
        
        print(f"Generating for {image_id}...")
        
        with torch.no_grad():
            oracle_embeds = adapter(oracle_z)
            real_eeg_embeds = adapter(real_eeg_z)
            shuffled_eeg_embeds = adapter(shuffled_eeg_z)
            rand_embeds = adapter(rand_z)
            
            oracle_gen = backend.generate_from_embeds(oracle_embeds, height=512, width=512)[0]
            real_eeg_gen = backend.generate_from_embeds(real_eeg_embeds, height=512, width=512)[0]
            shuffled_eeg_gen = backend.generate_from_embeds(shuffled_eeg_embeds, height=512, width=512)[0]
            rand_gen = backend.generate_from_embeds(rand_embeds, height=512, width=512)[0]
            
        grid = build_grid(target_img, oracle_gen, real_eeg_gen, shuffled_eeg_gen, rand_gen)
        grid_path = os.path.join(args.output_dir, f"eeg_grid_{image_id}.jpg")
        grid.save(grid_path)
        print(f"Saved {grid_path}")

if __name__ == "__main__":
    main()

import torch
from pathlib import Path

def main():
    clip_dir = Path("data/processed/clip_embeddings")
    s1_path = clip_dir / "sub01_image_embeddings.pt"
    s2_path = clip_dir / "sub02_image_embeddings.pt"
    out_path = clip_dir / "combined_image_embeddings.pt"
    
    print(f"Loading sub-01 image embeddings from {s1_path}...")
    s1 = torch.load(s1_path, map_location="cpu")
    print(f"Loading sub-02 image embeddings from {s2_path}...")
    s2 = torch.load(s2_path, map_location="cpu")
    
    combined = {
        'model_name': s1['model_name'],
        'image_id': s1['image_id'] + s2['image_id'],
        'class_id': s1['class_id'] + s2['class_id'],
        'image_path': s1['image_path'] + s2['image_path'],
        'embedding': torch.cat([s1['embedding'], s2['embedding']], dim=0)
    }
    
    print(f"Saving combined image embeddings ({len(combined['image_id'])} images) to {out_path}...")
    torch.save(combined, out_path)
    print("Done!")

if __name__ == "__main__":
    main()

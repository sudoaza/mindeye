import torch
import pandas as pd
from pathlib import Path

def main():
    clip_dir = Path("data/processed/clip_embeddings")
    s1_sem_path = clip_dir / "image_semantic_text_embeddings.pt"
    s2_metadata_path = Path("data/processed/semantic_epochs/zuna_tight1s_sub02_runs01_32/all_runs_metadata.csv")
    out_path = clip_dir / "combined_image_semantic_text_embeddings.pt"
    
    print(f"Loading sub-01 semantic embeddings from {s1_sem_path}...")
    s1_sem = torch.load(s1_sem_path, map_location="cpu")
    
    # Extract the common semantic vector (which is the same for all sub-01 images)
    sample_key = next(iter(s1_sem["image_id_to_semantic"].keys()))
    common_vector = s1_sem["image_id_to_semantic"][sample_key]
    
    print(f"Loading sub-02 metadata from {s2_metadata_path}...")
    df_s2 = pd.read_csv(s2_metadata_path)
    unique_s2_images = df_s2["image_id"].unique()
    
    # Clone the dictionary
    combined_image_id_to_semantic = dict(s1_sem["image_id_to_semantic"])
    combined_image_id_to_caption_fields = dict(s1_sem["image_id_to_caption_fields"])
    
    # Add sub-02 images
    for img_id in unique_s2_images:
        img_id_str = str(img_id)
        combined_image_id_to_semantic[img_id_str] = common_vector
        combined_image_id_to_caption_fields[img_id_str] = {
            "image_path": f"data/raw/nod/stimuli/ImageNet/{img_id_str}.JPEG",
            "class_label": "",
            "vlm_model": s1_sem.get("vlm_model", "Qwen/Qwen2-VL-7B-Instruct"),
            "short_caption": "",
            "detailed_caption": "",
            "composition_caption": "",
            "attribute_caption": "",
            "objects": [],
            "scene": "",
            "setting": "",
            "spatial_layout": "",
            "dominant_colors": [],
            "materials_textures": [],
            "lighting": "",
            "viewpoint": "",
            "action_or_state": "",
            "mood": "",
            "uncertainties": [],
            "embedding_text": "  Composition:  Attributes: .",
            "quality_flags": {
                "mentions_uncertainty": False,
                "empty_or_failed": False,
                "too_generic": True
            }
        }
        
    combined = {
        "model": s1_sem.get("model", "openai/clip-vit-base-patch32"),
        "source": "image_semantics",
        "image_id_to_semantic": combined_image_id_to_semantic,
        "image_id_to_caption_fields": combined_image_id_to_caption_fields,
    }
    
    print(f"Saving combined semantic embeddings ({len(combined_image_id_to_semantic)} images) to {out_path}...")
    torch.save(combined, out_path)
    print("Done!")

if __name__ == "__main__":
    main()

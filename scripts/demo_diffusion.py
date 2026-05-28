#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

def parse_args():
    p = argparse.ArgumentParser(description="Phase 11 Frozen Diffusion Demo")
    p.add_argument("--checkpoint", required=True, help="Path to trained EEG-CLIP model checkpoint (best.pt)")
    p.add_argument("--common-probe", required=True, help="Path to pretrained common_probe.pt checkpoint")
    p.add_argument("--metadata", required=True, help="Path to validation metadata CSV")
    p.add_argument("--epochs-dir", required=True, help="Path to epochs directory")
    p.add_argument("--common-embeddings", required=True, help="Path to common_embeddings.pt containing target bank")
    p.add_argument("--stimuli-root", default="data/raw/nod/stimuli/ImageNet", help="Root folder of ImageNet stimulus images")
    p.add_argument("--output-dir", default="outputs/phase11_demo", help="Output directory for generated demo images")
    p.add_argument("--model-name", default="runwayml/stable-diffusion-v1-5", help="Stable Diffusion model checkpoint")
    p.add_argument("--device", default=None, help="cuda or cpu")
    p.add_argument("--num-samples", type=int, default=5, help="Number of demo samples to generate")
    return p.parse_args()

def add_watermark(image, text="Phase 11 demo, not validated reconstruction"):
    """Draw a prominent watermark banner on the image to label it clearly."""
    w, h = image.size
    draw = ImageDraw.Draw(image)
    
    # Draw dark semi-transparent banner at the bottom
    banner_h = 30
    draw.rectangle([0, h - banner_h, w, h], fill=(0, 0, 0))
    
    # Write text centered in the banner
    try:
        # Try loading a basic font
        font = ImageFont.load_default()
    except Exception:
        font = None
        
    draw.text((10, h - banner_h + 8), text, fill=(255, 60, 60))
    return image

def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Demo] Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load checkpoints and models
    print(f"[Demo] Loading EEG checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    setup = checkpoint["setup"]
    
    # Instantiate dataset
    from mindseye.datasets.semantic_pairs import SemanticPairConfig, ZunaClipPairDataset
    dataset_config = SemanticPairConfig(
        metadata_csv=args.metadata,
        epochs_dir=args.epochs_dir,
        epochs_dir_raw=setup.get("epochs_dir_raw"),
        epochs_dir_resample=setup.get("epochs_dir_resample"),
        common_embeddings_pt=args.common_embeddings,
        vlm_attributes_json=setup.get("vlm_attributes"),
        input_domain=setup.get("input_domain", "zuna"),
        target_mode="real",
        window_mode=setup.get("window_mode", "crop"),
        target_space=setup.get("target_space", "common"),
        add_event_marker=setup.get("add_event_marker", False),
        augment_eeg=False,
        subject_list=setup.get("subjects_loaded", None),
    )
    dataset = ZunaClipPairDataset(dataset_config)
    print(f"[Demo] Loaded validation dataset with {len(dataset)} samples.")
    
    # Build encoder
    n_channels, n_times = dataset.eeg_shape
    if setup.get("model") in ("spatial_temporal", "spatial_temporal_small"):
        from mindseye.models.spatial_temporal_encoder import build_spatial_temporal_encoder
        preset = "small" if setup["model"] == "spatial_temporal_small" else "medium"
        encoder = build_spatial_temporal_encoder(
            preset,
            n_channels=n_channels,
            embedding_dim=dataset.embedding_dim,
            ch_names=getattr(dataset, "ch_names", None),
            num_subjects=len(getattr(dataset, "unique_subjects", ["unknown"])),
        ).to(device)
    elif setup.get("model") in ("temporal_attn", "temporal_attn_small"):
        from mindseye.models.eeg_encoder import TemporalAttnEncoder
        num_subjects = len(setup.get("subjects_loaded", [1]))
        encoder = TemporalAttnEncoder(
            n_channels=n_channels,
            embedding_dim=dataset.embedding_dim,
            hidden_dim=setup.get("hidden_dim", 256),
            n_layers=setup.get("n_layers", 4),
            n_heads=setup.get("n_heads", 8),
            dropout=setup.get("dropout", 0.2),
            num_subjects=num_subjects,
        ).to(device)
    else:
        from mindseye.models.eeg_encoder import EEGClipEncoder
        encoder = EEGClipEncoder(
            n_channels=n_channels,
            n_times=n_times,
            embedding_dim=dataset.embedding_dim,
            hidden_dim=setup.get("hidden_dim", 256),
            dropout=setup.get("dropout", 0.2),
        ).to(device)
        
    encoder.load_state_dict(checkpoint["model_state"])
    encoder.eval()
    
    # Load Probe
    from mindseye.models.common_probe import CommonProbeModel, ATTRIBUTE_SCHEMAS
    probe_specs_path = Path(args.common_probe).parent / "task_specs.json"
    with open(probe_specs_path, "r") as f:
        task_specs = json.load(f)
    probe = CommonProbeModel(embedding_dim=dataset.embedding_dim, task_specs=task_specs).to(device)
    probe.load_state_dict(torch.load(args.common_probe, map_location=device))
    probe.eval()
    
    # Load stable diffusion pipelines
    print(f"[Demo] Loading Stable Diffusion {args.model_name}...")
    from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
    
    sd_text_pipe = StableDiffusionPipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None
    ).to(device)
    
    sd_img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None
    ).to(device)
    
    # Load candidate target embeddings for nearest-neighbor search
    common_embs = torch.load(args.common_embeddings, map_location="cpu")
    cand_ids = list(common_embs["image_id_to_common"].keys())
    cand_tensor = torch.stack([common_embs["image_id_to_common"][k] for k in cand_ids]).to(device)
    
    # 2. Run generation loop
    import random
    indices = list(range(len(dataset)))
    random.seed(42)
    random.shuffle(indices)
    selected_indices = indices[:args.num_samples]
    
    print(f"[Demo] Running inference and generation for {args.num_samples} samples...")
    for rank, idx in enumerate(selected_indices, 1):
        sample = dataset[idx]
        eeg = sample["eeg"].unsqueeze(0).to(device)
        image_id = sample["image_id"]
        
        # Predict z_pred_common
        with torch.inference_mode():
            subject_id = torch.tensor([sample["subject_id"]], device=device).long()
            kwargs = {"subject_id": subject_id} if getattr(encoder, "subject_embed", None) is not None else {}
            pred = encoder(eeg, **kwargs)
            pred_norm = F.normalize(pred, dim=-1)
            
            # Predict attributes
            logits_dict = probe(pred_norm)
            pred_attrs = {}
            for task, logits in logits_dict.items():
                if task in ATTRIBUTE_SCHEMAS:
                    pred_idx = logits[0].argmax().item()
                    pred_attrs[task] = ATTRIBUTE_SCHEMAS[task][pred_idx]
                    
            # Find nearest neighbor in target bank
            sims = torch.mm(pred_norm, cand_tensor.t())[0]
            top_idx = sims.argmax().item()
            prior_image_id = cand_ids[top_idx]
            
        print(f"\n[{rank}/{args.num_samples}] Target Image: {image_id}")
        print(f"  Predicted attributes: {json.dumps(pred_attrs)}")
        print(f"  Retrieved visual prior: {prior_image_id}")
        
        # Build prompt from predicted attributes
        desc = []
        if pred_attrs.get("is_animate") == "yes":
            if pred_attrs.get("face_visible") == "yes":
                if pred_attrs.get("animal_visible") == "yes":
                    desc.append("an animal face")
                else:
                    desc.append("a human face")
            elif pred_attrs.get("animal_visible") == "yes":
                desc.append("an animal")
            else:
                desc.append("a person")
        else:
            desc.append("an object")
            
        if pred_attrs.get("furry") == "yes":
            desc.append("furry texture")
        elif pred_attrs.get("soft_texture") == "yes":
            desc.append("soft texture")
            
        if pred_attrs.get("dominant_color") not in ("unclear", "gray"):
            desc.append(f"dominant {pred_attrs['dominant_color']} color")
            
        desc_str = ", ".join(desc) if desc else "a simple visual stimulus"
        prompt = f"A photo of {desc_str}, plain background, studio lighting, highly detailed"
        print(f"  Steered prompt: {prompt}")
        
        # Save Target Image for reference if available
        # NOD layout check
        from mindseye.embeddings.clip import resolve_image_path
        target_path = resolve_image_path(args.stimuli_root, "unknown", image_id)
        if target_path and target_path.exists():
            target_img = Image.open(target_path).resize((500, 500))
            target_img.save(output_dir / f"sample_{rank:02d}_target_{image_id}.jpg")
            
        # Resolve retrieved prior image
        prior_path = resolve_image_path(args.stimuli_root, "unknown", prior_image_id)
        prior_img = None
        if prior_path and prior_path.exists():
            prior_img = Image.open(prior_path).resize((500, 500))
            prior_img.save(output_dir / f"sample_{rank:02d}_prior_{prior_image_id}.jpg")
            
        # Mode A: Text prompt only
        print("  Generating Mode A (Text-only)...")
        with torch.inference_mode():
            img_a = sd_text_pipe(prompt, num_inference_steps=25).images[0]
        img_a = add_watermark(img_a)
        img_a.save(output_dir / f"sample_{rank:02d}_modeA_text.jpg")
        
        # Mode B: Image-prior conditioned
        if prior_img is not None:
            print("  Generating Mode B (Img2Img prior-guided)...")
            with torch.inference_mode():
                img_b = sd_img_pipe(
                    prompt=prompt,
                    image=prior_img,
                    strength=0.6,
                    num_inference_steps=25
                ).images[0]
            img_b = add_watermark(img_b)
            img_b.save(output_dir / f"sample_{rank:02d}_modeB_img2img.jpg")
            
    print(f"\n[Demo] Generation complete! All images saved to {output_dir}")

if __name__ == "__main__":
    main()

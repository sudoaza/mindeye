#!/usr/bin/env python3
import argparse
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindseye.generation.rae_backend import RaeDecoderBackend
from mindseye.generation.clip_native_backend import ClipNativeDecoderBackend
from mindseye.datasets.semantic_pairs import ZunaClipPairDataset, SemanticPairConfig, split_indices

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings-pt", default="data/processed/rae_embeddings/rae_dinov2_base_sub01_04_runs01_40.pt", help="Path to RAE latent bank")
    p.add_argument("--metadata", default="data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv", help="Path to metadata CSV")
    p.add_argument("--epochs-dir", default="data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40", help="Path to epochs directory")
    p.add_argument("--stimuli-dir", default="data/raw/nod/stimuli/ImageNet", help="Path to ImageNet stimulus images")
    p.add_argument("--num-samples", type=int, default=100, help="Number of validation samples to evaluate")
    p.add_argument("--output-dir", default="outputs/rae_oracle", help="Output directory")
    return p.parse_args()

def run_bootstrap_ci(values, num_iterations=1000, ci=95):
    boot_means = []
    n = len(values)
    values = np.array(values)
    for _ in range(num_iterations):
        boot_sample = np.random.choice(values, size=n, replace=True)
        boot_means.append(np.mean(boot_sample))
    boot_means = np.array(boot_means)
    lower_pct = (100 - ci) / 2.0
    upper_pct = 100 - lower_pct
    return {
        "mean": float(np.mean(values)),
        "ci_lower": float(np.percentile(boot_means, lower_pct)),
        "ci_upper": float(np.percentile(boot_means, upper_pct))
    }

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize RAE and CLIP backends
    rae_backend = RaeDecoderBackend(device=device, apply_patch=True)
    rae_backend.load()
    clip_backend = ClipNativeDecoderBackend(device=device)
    
    # Load dataset to get exact validation split indices
    print("Loading Validation Dataset...")
    dataset_config = SemanticPairConfig(
        common_embeddings_pt=args.embeddings_pt,
        metadata_csv=args.metadata,
        epochs_dir=args.epochs_dir,
        window_mode="tight1s",
        target_mode="real",
        input_domain="zuna",
        target_space="rae_unit"
    )
    dataset = ZunaClipPairDataset(dataset_config)
    
    val_fraction = 0.15
    seed = 13
    _, val_indices = split_indices(len(dataset), val_fraction=val_fraction, seed=seed)
    
    num_eval = min(args.num_samples, len(val_indices))
    eval_val_indices = val_indices[:num_eval]
    print(f"Evaluating {num_eval} validation samples...")
    
    # Load raw RAE tokens from latent bank table
    table = torch.load(args.embeddings_pt, map_location="cpu")
    image_id_to_rae_tokens = table["image_id_to_rae_tokens"]
    
    rae_native_cosines = []
    clip_cosines = []
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    grid_targets = []
    grid_recons = []
    
    stimuli_dir = Path(args.stimuli_dir)
    
    for count, idx in enumerate(tqdm(eval_val_indices, desc="Running Oracle Reconstruction")):
        item = dataset[idx]
        img_id = item["image_id"]
        
        # Load target image
        stim_path = stimuli_dir / f"{img_id}.JPEG"
        if not stim_path.exists():
            stim_path = stimuli_dir / f"{img_id}.png"
        if not stim_path.exists():
            continue
            
        target_img = Image.open(stim_path).convert("RGB")
        target_img_resized = target_img.resize((256, 256))
        
        # Retrieve true RAE spatial tokens
        if img_id not in image_id_to_rae_tokens:
            print(f"Warning: RAE tokens missing for {img_id}")
            continue
            
        tokens = image_id_to_rae_tokens[img_id].to(device).unsqueeze(0)  # [1, 768, 16, 16]
        
        # Decode back to image
        oracle_imgs = rae_backend.generate_from_embeds(tokens)
        oracle_img = oracle_imgs[0]
        
        # 1. RAE-native cosine similarity
        target_unit = dataset.image_id_to_target[img_id].to(device).float()  # [768]
        recon_latents = rae_backend.extract_rae_latent(oracle_img)
        recon_unit = recon_latents["unit"].squeeze(0)  # [768]
        
        rae_cos = F.cosine_similarity(target_unit, recon_unit, dim=0).item()
        rae_native_cosines.append(rae_cos)
        
        # 2. CLIP-space cosine similarity
        target_clip = clip_backend.extract_teacher_embeds(target_img_resized, normalize=True).squeeze(0)
        recon_clip = clip_backend.extract_teacher_embeds(oracle_img, normalize=True).squeeze(0)
        
        clip_cos = F.cosine_similarity(target_clip, recon_clip, dim=0).item()
        clip_cosines.append(clip_cos)
        
        # Save first 8 for visual grid
        if len(grid_targets) < 8:
            grid_targets.append(TF.to_tensor(target_img_resized))
            grid_recons.append(TF.to_tensor(oracle_img))
            
    # Calculate stats
    rae_stats = run_bootstrap_ci(rae_native_cosines)
    clip_stats = run_bootstrap_ci(clip_cosines)
    
    print("\n" + "="*80)
    print("RAE ORACLE EVALUATION RESULTS")
    print("="*80)
    print(f"RAE-Native Cosine: {rae_stats['mean']:.5f} ({rae_stats['ci_lower']:.5f} to {rae_stats['ci_upper']:.5f})")
    print(f"CLIP-Space Cosine: {clip_stats['mean']:.5f} ({clip_stats['ci_lower']:.5f} to {clip_stats['ci_upper']:.5f})")
    print("="*80)
    
    # Save metrics
    metrics = {
        "rae_native_oracle_cosine": rae_stats,
        "clip_space_oracle_cosine": clip_stats,
        "num_samples_evaluated": len(rae_native_cosines)
    }
    with open(output_dir / "oracle_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    # Save visual grid (Target | Oracle)
    all_tensors = []
    for t_img, r_img in zip(grid_targets, grid_recons):
        all_tensors.append(t_img)
        all_tensors.append(r_img)
        
    grid = make_grid(all_tensors, nrow=2, padding=4)
    grid_pil = TF.to_pil_image(grid)
    grid_pil.save(output_dir / "oracle_grid.png")
    
    print(f"Saved oracle grid to {output_dir / 'oracle_grid.png'}")
    print(f"Saved oracle metrics to {output_dir / 'oracle_metrics.json'}")

if __name__ == "__main__":
    main()

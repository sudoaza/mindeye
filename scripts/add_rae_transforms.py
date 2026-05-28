#!/usr/bin/env python3
import argparse
import sys
import torch
import pandas as pd
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindseye.datasets.semantic_pairs import split_indices

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", default="data/processed/rae_embeddings/rae_dinov2_base_sub01_04_runs01_40.pt", help="Path to RAE bank")
    p.add_argument("--metadata", default="data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv,data/processed/semantic_epochs/zuna_tight1s_sub02_runs01_40/all_runs_metadata.csv,data/processed/semantic_epochs/zuna_tight1s_sub03_runs01_40/all_runs_metadata.csv,data/processed/semantic_epochs/zuna_tight1s_sub04_runs01_40/all_runs_metadata.csv", help="Comma-separated metadata CSV paths")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--whitening-eps", type=float, default=1e-5)
    return p.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading RAE latent bank from {args.bank}...")
    bank = torch.load(args.bank, map_location="cpu")
    
    if "image_id_to_rae_global" not in bank:
        raise ValueError("Bank is missing 'image_id_to_rae_global'.")
        
    global_dict = bank["image_id_to_rae_global"]
    
    # Load metadata to find train images
    metadata_csv_list = [p.strip() for p in str(args.metadata).split(",")]
    dfs = []
    for csv_path in metadata_csv_list:
        if Path(csv_path).exists():
            dfs.append(pd.read_csv(csv_path))
        else:
            print(f"Warning: {csv_path} not found")
            
    if not dfs:
        raise FileNotFoundError("No valid metadata CSVs found.")
        
    metadata = pd.concat(dfs, ignore_index=True)
    metadata["image_id_str"] = metadata["image_id"].astype(str)
    
    # Filter metadata exactly as ZunaClipPairDataset does
    mask = metadata["image_id_str"].isin(global_dict.keys())
    metadata = metadata[mask].reset_index(drop=True)
    
    print(f"Total dataset items after filtering: {len(metadata)}")
    
    # Get train indices
    train_idx, val_idx = split_indices(len(metadata), val_fraction=args.val_fraction, seed=args.seed)
    
    # Get unique train image IDs
    train_image_ids = set(metadata.iloc[train_idx]["image_id_str"].tolist())
    val_image_ids = set(metadata.iloc[val_idx]["image_id_str"].tolist())
    
    print(f"Found {len(train_image_ids)} unique train image IDs and {len(val_image_ids)} unique validation image IDs.")
    
    # Build train set tensor
    train_vectors = torch.stack([global_dict[img_id].float() for img_id in train_image_ids])
    
    print(f"Train vectors shape: {train_vectors.shape}")
    
    # Compute centering (mean)
    rae_center_mean = train_vectors.mean(dim=0)
    
    # Compute whitening (PCA)
    train_centered = train_vectors - rae_center_mean
    U, S, V = torch.pca_lowrank(train_centered, q=min(train_centered.shape[0], train_centered.shape[1]), center=False)
    
    rae_pca_components = V
    rae_pca_eigenvalues = S ** 2 / (train_centered.shape[0] - 1)
    
    whitening_eps = args.whitening_eps
    
    print("Computing explicit variants for all bank images...")
    image_id_to_rae_centered_unit = {}
    image_id_to_rae_whitened_unit = {}
    
    for img_id, vec in global_dict.items():
        v = vec.float()
        
        # Centered + normalized
        v_centered = v - rae_center_mean
        v_centered_unit = torch.nn.functional.normalize(v_centered, dim=0)
        image_id_to_rae_centered_unit[img_id] = v_centered_unit
        
        # Whitened + normalized
        v_whitened = torch.matmul(v_centered, rae_pca_components)
        v_whitened = v_whitened / torch.sqrt(rae_pca_eigenvalues + whitening_eps)
        v_whitened_unit = torch.nn.functional.normalize(v_whitened, dim=0)
        image_id_to_rae_whitened_unit[img_id] = v_whitened_unit
        
    bank["image_id_to_rae_centered_unit"] = image_id_to_rae_centered_unit
    bank["image_id_to_rae_whitened_unit"] = image_id_to_rae_whitened_unit
    
    bank["rae_center_mean"] = rae_center_mean
    bank["rae_pca_components"] = rae_pca_components
    bank["rae_pca_eigenvalues"] = rae_pca_eigenvalues
    bank["whitening_eps"] = whitening_eps
    bank["fit_split"] = "train_only"
    
    print("Saving updated bank...")
    torch.save(bank, args.bank)
    print("Done!")

if __name__ == "__main__":
    main()

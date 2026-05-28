#!/usr/bin/env python3
import torch
import sys
from pathlib import Path

def audit(bank_path):
    print(f"Auditing target geometries in {bank_path}")
    bank = torch.load(bank_path, map_location="cpu")
    
    keys = ["image_id_to_rae_unit", "image_id_to_rae_centered_unit", "image_id_to_rae_whitened_unit"]
    
    for key in keys:
        if key not in bank:
            print(f"Missing {key}")
            continue
            
        print(f"--- {key} ---")
        global_dict = bank[key]
        image_ids = sorted(list(global_dict.keys()))
        target = torch.stack([global_dict[i].float() for i in image_ids])
        
        # ensure normalized
        target = torch.nn.functional.normalize(target, dim=-1)
        
        # Optimize memory usage using matrix multiplication
        cos = torch.mm(target, target.t())
        # Exclude diagonal
        cos_off = cos[~torch.eye(len(image_ids), dtype=torch.bool, device=cos.device)]
        
        print(f"  Target bank N: {len(image_ids)}")
        print(f"  Off-diag mean: {cos_off.mean().item():.5f}")
        print(f"  Off-diag std:  {cos_off.std().item():.5f}")
        
        # Compute nearest neighbor sanity check (top 10 closest for the first image)
        print(f"  Nearest neighbors for first image ({image_ids[0]}):")
        dists = cos[0]
        topk = torch.topk(dists, k=6)  # 1 is itself
        for rank, (idx, score) in enumerate(zip(topk.indices[1:], topk.values[1:])):
            print(f"    {rank+1}. {image_ids[idx]} (cos = {score.item():.4f})")
            
if __name__ == "__main__":
    audit("data/processed/rae_embeddings/rae_dinov2_base_sub01_04_runs01_40.pt")

#!/usr/bin/env python3
import os
import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import gc

# Ensure import paths work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


def compute_pairwise_cosine(features):
    # features: [N, D]
    norm_feat = F.normalize(features, dim=-1)
    sim = norm_feat @ norm_feat.T # [N, N]
    return sim


def main():
    parser = argparse.ArgumentParser(description="Run diagnostics on cached ZUNA latents (OOM-safe).")
    parser.add_argument("--cache-dir", type=str, required=True, help="Directory containing metadata.pt and latents_*.pt")
    args = parser.parse_args()
    
    metadata_json_path = os.path.join(args.cache_dir, "metadata.json")
    metadata_pt_path = os.path.join(args.cache_dir, "metadata.pt")
    
    if not os.path.exists(metadata_json_path):
        print(f"Error: {metadata_json_path} does not exist.")
        sys.exit(1)
    if not os.path.exists(metadata_pt_path):
        print(f"Error: {metadata_pt_path} does not exist.")
        sys.exit(1)
        
    with open(metadata_json_path, "r") as f:
        meta_info = json.load(f)
        
    all_layers = meta_info.get("cached_layers", [])
    print(f"Found cached layers in metadata: {all_layers}")
    
    print(f"Loading metadata records from {metadata_pt_path}...")
    records = torch.load(metadata_pt_path, map_location="cpu")
    n_records = len(records)
    print(f"Loaded {n_records} metadata records.")
    
    # 1. NaN and Inf check (Hard Health Check)
    print("\n--- Diagnostic 1: NaN / Inf check (Hard Gate) ---")
    any_nan_or_inf = False
    for layer in all_layers:
        layer_path = os.path.join(args.cache_dir, f"latents_{layer}.pt")
        if not os.path.exists(layer_path):
            print(f"  ❌ Layer {layer:10s} | Missing cache file: {layer_path}")
            any_nan_or_inf = True
            continue
            
        print(f"  Loading layer {layer}...")
        layer_dict = torch.load(layer_path, map_location="cpu")
        
        has_nan = False
        has_inf = False
        for r in records:
            s_id = r["sample_id"]
            if s_id in layer_dict:
                tensor = layer_dict[s_id]
                if torch.isnan(tensor).any():
                    has_nan = True
                if torch.isinf(tensor).any():
                    has_inf = True
            else:
                print(f"  ⚠️ Warning: sample_id '{s_id}' not found in layer '{layer}' dict.")
                
        # Clean memory
        del layer_dict
        gc.collect()
        
        if has_nan or has_inf:
            print(f"  ❌ Layer {layer:10s} | Has NaN: {has_nan} | Has Inf: {has_inf}")
            any_nan_or_inf = True
        else:
            print(f"  ✅ Layer {layer:10s} | Clean (no NaN/Inf)")
            
    if any_nan_or_inf:
        print("❌ FAILED: Found NaNs, Infs, or missing layers in ZUNA latents. Do not proceed!")
        sys.exit(1)
    else:
        print("✅ PASSED: No NaNs or Infs found.")
        
    # 2. Collapse check: standard deviation across trials (Hard Health Check)
    print("\n--- Diagnostic 2: Trial-to-trial standard deviation (Hard Gate) ---")
    all_passed_variance = True
    for layer in all_layers:
        layer_path = os.path.join(args.cache_dir, f"latents_{layer}.pt")
        print(f"  Loading layer {layer}...")
        layer_dict = torch.load(layer_path, map_location="cpu")
        
        # Stack tensors cleanly
        stacked_list = [layer_dict[r["sample_id"]].float() for r in records if r["sample_id"] in layer_dict]
        tensors = torch.stack(stacked_list) # [N, N_tokens, D]
        
        std_across_trials = tensors.std(dim=0).mean().item()
        mean_std = tensors.mean(dim=0).std().item()
        
        # Clean memory
        del layer_dict
        del stacked_list
        del tensors
        gc.collect()
        
        is_collapsed = std_across_trials < 0.001
        status = "❌ COLLAPSED" if is_collapsed else "✅ OK"
        print(f"  {status} | Layer {layer:10s} | Average trial std: {std_across_trials:.6f} | Feature mean std: {mean_std:.6f}")
        if is_collapsed:
            all_passed_variance = False
            
    if not all_passed_variance:
        print("❌ FAILED: Latents are collapsed (constant across trials). Check EEG normalization!")
        sys.exit(1)
    else:
        print("✅ PASSED: All layers show active variance across trials.")
        
    # 3. Same-class vs different-class clustering (Soft Warning Check)
    print("\n--- Diagnostic 3: Same-class vs Different-class clustering (Warning Gate) ---")
    class_groups = {}
    for r in records:
        cls = r["class_id"]
        if cls != "MISSING" and pd.notna(cls):
            if cls not in class_groups:
                class_groups[cls] = []
            class_groups[cls].append(r)
            
    valid_classes = {cls: group for cls, group in class_groups.items() if len(group) >= 2}
    print(f"Found {len(valid_classes)} classes with 2 or more trials.")
    
    if len(valid_classes) < 5:
        print("⚠️ WARNING: Too few classes with repeat trials to compute clustering metrics. Skipping.")
        return
        
    for layer in all_layers:
        layer_path = os.path.join(args.cache_dir, f"latents_{layer}.pt")
        layer_dict = torch.load(layer_path, map_location="cpu")
        
        trial_features = []
        trial_classes = []
        
        for r in records:
            cls = r["class_id"]
            s_id = r["sample_id"]
            if cls in valid_classes and s_id in layer_dict:
                feat = layer_dict[s_id].float().mean(dim=0) # [D]
                trial_features.append(feat)
                trial_classes.append(cls)
                
        # Clean memory of raw dictionary early
        del layer_dict
        gc.collect()
        
        trial_features = torch.stack(trial_features) # [N_subset, D]
        sim_matrix = compute_pairwise_cosine(trial_features).numpy() # [N_subset, N_subset]
        
        same_class_sims = []
        diff_class_sims = []
        
        n_subset = len(trial_classes)
        for i in range(n_subset):
            for j in range(i + 1, n_subset):
                sim = sim_matrix[i, j]
                if trial_classes[i] == trial_classes[j]:
                    same_class_sims.append(sim)
                else:
                    diff_class_sims.append(sim)
                    
        mean_same = np.mean(same_class_sims) if same_class_sims else 0.0
        mean_diff = np.mean(diff_class_sims) if diff_class_sims else 0.0
        gap = mean_same - mean_diff
        
        status = "✅ OK" if gap > 0.0 else "⚠️ WEAK"
        print(f"  {status} | Layer {layer:10s} | Same-class similarity: {mean_same:.4f} | Diff-class similarity: {mean_diff:.4f} | Gap: {gap:+.4f}")
        
        del trial_features
        gc.collect()
        
    print("\n✓ Diagnostics completed successfully.")


if __name__ == "__main__":
    main()

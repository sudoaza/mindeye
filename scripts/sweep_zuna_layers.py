import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

def ridge_regression_fit_predict(X_train, Y_train, X_test, alpha=1.0):
    # Closed-form Ridge Regression: W = (X_train^T X_train + alpha I)^(-1) X_train^T Y_train
    # X_train: [N, D_in], Y_train: [N, D_out]
    N, D_in = X_train.shape
    device = X_train.device
    
    # Add bias term by appending a column of ones
    ones_train = torch.ones(N, 1, device=device)
    X_train_bias = torch.cat([X_train, ones_train], dim=1)
    
    N_test = X_test.shape[0]
    ones_test = torch.ones(N_test, 1, device=device)
    X_test_bias = torch.cat([X_test, ones_test], dim=1)
    
    # Solve W
    XTX = X_train_bias.T @ X_train_bias
    I = torch.eye(D_in + 1, device=device)
    W = torch.linalg.solve(XTX + alpha * I, X_train_bias.T @ Y_train)
    
    # Predict
    Y_pred = X_test_bias @ W
    return Y_pred

def compute_cosine_sim(Y_pred, Y_test):
    # Y_pred: [N, D], Y_test: [N, D]
    pred_norm = Y_pred / (Y_pred.norm(dim=-1, keepdim=True) + 1e-8)
    test_norm = Y_test / (Y_test.norm(dim=-1, keepdim=True) + 1e-8)
    return (pred_norm * test_norm).sum(dim=-1).mean().item()

def compute_top10_retrieval(Y_pred, Y_test):
    # Predict over test fold using test fold targets as the candidate bank
    # Y_pred: [N, D], Y_test: [N, D]
    pred_norm = Y_pred / (Y_pred.norm(dim=-1, keepdim=True) + 1e-8)
    test_norm = Y_test / (Y_test.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Cosine matrix: [N, N]
    similarity_matrix = pred_norm @ test_norm.T
    
    # Top 10 indices
    top10_indices = similarity_matrix.topk(10, dim=-1).indices
    
    # Check if correct index (diagonal) is in top 10
    correct = 0
    N = Y_pred.shape[0]
    for i in range(N):
        if i in top10_indices[i]:
            correct += 1
    return correct / N

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Paths
    cache_path = "/workspace/mindeye/data/processed/zuna_latents/sub01_runs01_08_sweep/latents.pt"
    clip_path = "/workspace/mindeye/data/processed/clip_embeddings/common_embeddings.pt"
    rae_path = "/workspace/mindeye/data/processed/rae_embeddings/rae_dinov2_base_sub01_04_runs01_40.pt"
    
    print(f"Loading cached latents from {cache_path}...")
    records = torch.load(cache_path, map_location="cpu")
    print(f"Loaded {len(records)} records.")
    
    print(f"Loading CLIP embeddings from {clip_path}...")
    clip_data = torch.load(clip_path, map_location="cpu")
    clip_image = clip_data["image_id_to_image"]
    clip_common = clip_data["image_id_to_common"]
    
    print(f"Loading DINO/RAE embeddings from {rae_path}...")
    rae_data = torch.load(rae_path, map_location="cpu")
    rae_global = rae_data["image_id_to_rae_global"]
    rae_unit = rae_data["image_id_to_rae_unit"]
    
    target_spaces = {
        "CLIP-Image": clip_image,
        "CLIP-Common": clip_common,
        "DINO-Global": rae_global,
        "DINO-Unit": rae_unit
    }
    
    layers = ["layer_4", "layer_8", "layer_12", "layer_16", "pre_mmd", "post_mmd"]
    pooling_modes = ["all", "onset"]
    
    results = []
    
    # Build K-Fold splits
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Filter records to only those with valid targets
    valid_records = []
    for r in records:
        img_id = r["image_id"]
        # Ensure it has targets in all spaces
        if (img_id in clip_image and img_id in clip_common and 
            img_id in rae_global and img_id in rae_unit):
            valid_records.append(r)
            
    print(f"Found {len(valid_records)} valid records with target embeddings.")
    
    for layer in layers:
        for pool_mode in pooling_modes:
            # Prepare X inputs
            X_list = []
            for r in valid_records:
                # latent shape: [2480, dim]
                latent = r[layer].float()
                # reshape to [n_channels, tc, dim] = [62, 40, dim]
                # Since N = n_channels * tc = 2480
                dim = latent.shape[1]
                latent_reshaped = latent.view(62, 40, dim)
                
                if pool_mode == "all":
                    # Mean pool all tokens
                    x = latent_reshaped.mean(dim=(0, 1)) # [dim]
                else:
                    # Onset window: tc=24 to 32
                    x = latent_reshaped[:, 24:32, :].mean(dim=(0, 1)) # [dim]
                X_list.append(x)
                
            X = torch.stack(X_list).to(device) # [N, D_in]
            
            for target_name, target_dict in target_spaces.items():
                Y_list = [target_dict[r["image_id"]].float() for r in valid_records]
                Y = torch.stack(Y_list).to(device) # [N, D_out]
                
                fold_cos = []
                fold_top10 = []
                
                for train_idx, test_idx in kf.split(X):
                    X_train, X_test = X[train_idx], X[test_idx]
                    Y_train, Y_test = Y[train_idx], Y[test_idx]
                    
                    # Ridge fit and predict (alpha=10.0 default for ridge)
                    Y_pred = ridge_regression_fit_predict(X_train, Y_train, X_test, alpha=10.0)
                    
                    cos = compute_cosine_sim(Y_pred, Y_test)
                    top10 = compute_top10_retrieval(Y_pred, Y_test)
                    
                    fold_cos.append(cos)
                    fold_top10.append(top10)
                    
                mean_cos = np.mean(fold_cos)
                mean_top10 = np.mean(fold_top10)
                
                results.append({
                    "Layer": layer,
                    "Pooling": pool_mode,
                    "Target": target_name,
                    "Test Cosine": mean_cos,
                    "Top-10 Acc": mean_top10
                })
                print(f"Layer: {layer:8s} | Pooling: {pool_mode:5s} | Target: {target_name:12s} | Cosine: {mean_cos:.4f} | Top-10: {mean_top10:.4f}")
                
    # Sort and display results
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(by="Test Cosine", ascending=False)
    
    print("\n\n### ZUNA Layer Sweep Results (Linear Probe) ###")
    print(df_res.to_markdown(index=False))
    
    # Save results to a CSV in output directory
    df_res.to_csv("/workspace/mindeye/data/processed/zuna_latents/sub01_runs01_08_sweep/sweep_results.csv", index=False)

if __name__ == "__main__":
    main()

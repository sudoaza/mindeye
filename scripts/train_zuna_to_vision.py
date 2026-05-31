#!/usr/bin/env python3
import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from pathlib import Path
import json

# Ensure import paths work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from mindseye.adapters.qformer import ZunaToVisionQFormer
from mindseye.models.eeg_encoder import clip_contrastive_loss, retrieval_topk

def parse_runs_spec(spec: str) -> list[int]:
    runs = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-")
            runs.update(range(int(start), int(end) + 1))
        else:
            runs.add(int(part))
    return sorted(list(runs))

def variance_floor_loss(pred: torch.Tensor, target_std: torch.Tensor) -> torch.Tensor:
    pred_std = torch.sqrt(pred.var(dim=0) + 1e-6)
    return torch.mean(F.relu(target_std - pred_std))

class ZunaLatentTargetDataset(Dataset):
    """
    Dataset that loads cached ZUNA latents and maps them to target embeddings (CLIP/DINO).
    Supports baseline control target modes: real, shuffled, and random.
    """
    def __init__(
        self, 
        latents_pt_path: str, 
        targets_pt_path: str, 
        target_space: str, 
        layer_name: str, 
        target_mode: str = "real", 
        shuffle_seed: int = 42, 
        subject_list: list = None
    ):
        # Resolve cache dir and split paths
        if os.path.isdir(latents_pt_path):
            cache_dir = latents_pt_path
        else:
            cache_dir = os.path.dirname(latents_pt_path)
            
        metadata_path = os.path.join(cache_dir, "metadata.pt")
        layer_path = os.path.join(cache_dir, f"latents_{layer_name}.pt")
        
        # Fallback to combined latents.pt if metadata.pt doesn't exist
        if not os.path.exists(metadata_path):
            metadata_path = os.path.join(cache_dir, "latents.pt")
            
        print(f"Loading metadata from {metadata_path}...")
        self.records = torch.load(metadata_path, map_location="cpu")
        print(f"Loaded {len(self.records)} metadata records.")
        
        # Load layer dictionary
        if os.path.exists(layer_path):
            print(f"Loading layer '{layer_name}' latents from {layer_path}...")
            self.layer_dict = torch.load(layer_path, map_location="cpu")
            self.use_split_files = True
        else:
            print(f"Layer file {layer_path} not found. Assuming combined latents.pt structure.")
            self.use_split_files = False
        
        # Load targets dict
        print(f"Loading target embeddings from {targets_pt_path}...")
        targets_data = torch.load(targets_pt_path, map_location="cpu")
        
        # Map target spaces dynamically
        self.pca_dims = None
        target_space_key = target_space
        if "PCA" in target_space:
            self.pca_dims = int(target_space.split("-")[2])
            target_space_key = "rae_unit"
        elif target_space in ("DINO-Unit-768", "DINO-Unit"):
            target_space_key = "rae_unit"
        elif target_space in ("CLIP-Common-512", "CLIP-Common"):
            target_space_key = "common"

        # Resolve target key
        if target_space_key in targets_data:
            self.image_id_to_target = targets_data[target_space_key]
        elif f"image_id_to_{target_space_key}" in targets_data:
            self.image_id_to_target = targets_data[f"image_id_to_{target_space_key}"]
        else:
            possible_keys = [k for k in targets_data.keys() if target_space_key in k]
            if possible_keys:
                self.image_id_to_target = targets_data[possible_keys[0]]
            else:
                raise ValueError(f"Target space '{target_space_key}' not found in keys: {list(targets_data.keys())}")
                
        # Filter records that have valid target embeddings
        self.valid_records = []
        for r in self.records:
            if r["image_id"] in self.image_id_to_target:
                # If using split files, ensure the sample exists in the layer dict
                if self.use_split_files and r["sample_id"] not in self.layer_dict:
                    continue
                if subject_list is not None:
                    if f"sub-{r['subject_id']:02d}" in subject_list:
                        self.valid_records.append(r)
                else:
                    self.valid_records.append(r)
                    
        print(f"Found {len(self.valid_records)} valid records with target embeddings.")
        if len(self.valid_records) == 0:
            raise ValueError("No valid records found after filtering.")
            
        # Expose dimensions
        self.layer_name = layer_name
        if self.use_split_files:
            first_latent = self.layer_dict[self.valid_records[0]["sample_id"]]
        else:
            first_latent = self.valid_records[0][layer_name]
        self.latent_dim = first_latent.shape[-1]
        first_target = self.image_id_to_target[self.valid_records[0]["image_id"]]
        
        if self.pca_dims is not None:
            self.target_dim = self.pca_dims
        else:
            self.target_dim = first_target.shape[-1]
            
        print(f"Latent dim: {self.latent_dim} | Target dim: {self.target_dim}")
        
        self.target_mode = target_mode
        n = len(self.valid_records)
        rng = np.random.default_rng(shuffle_seed)
        
        if target_mode == "shuffled":
            self.target_perm = rng.permutation(n).tolist()
        elif target_mode == "random":
            # Generate unit-norm random gaussian targets
            vecs = rng.standard_normal((n, self.target_dim)).astype("float32")
            norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-8)
            self.random_targets = [
                torch.from_numpy(vecs[i] / norms[i]).float() for i in range(n)
            ]

    def __len__(self):
        return len(self.valid_records)

    def __getitem__(self, idx):
        record = self.valid_records[idx]
        s_id = record["sample_id"]
        
        if self.use_split_files:
            latent = self.layer_dict[s_id].float()
        else:
            latent = record[self.layer_name].float()
        
        if self.target_mode == "real":
            target = self.image_id_to_target[record["image_id"]].float()
        elif self.target_mode == "shuffled":
            perm_idx = self.target_perm[idx]
            perm_record = self.valid_records[perm_idx]
            target = self.image_id_to_target[perm_record["image_id"]].float()
        elif self.target_mode == "random":
            target = self.random_targets[idx]
        else:
            raise ValueError(f"Unknown target_mode: {self.target_mode}")
            
        return {
            "latent": latent,
            "target": target,
            "subject_id": torch.tensor(record["subject_id"] - 1, dtype=torch.long),
            "run_id": record["run_id"],
            "image_id": record["image_id"],
            "class_id": record["class_id"],
            "sample_id": record["sample_id"]
        }

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_epoch(model, loader, optimizer, temperature, target_std, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in loader:
        latents = batch["latent"].to(device)
        targets = batch["target"].to(device)
        subject_ids = batch["subject_id"].to(device)
        
        optimizer.zero_grad()
        preds = model(latents, subject_id=subject_ids)
        
        # InfoNCE + Cosine + Variance Floor Loss
        pred_u = F.normalize(preds, dim=-1)
        target_u = F.normalize(targets, dim=-1)
        
        loss_nce = clip_contrastive_loss(pred_u, target_u, temperature=temperature)
        loss_cos = 0.5 * (1.0 - F.cosine_similarity(pred_u, target_u, dim=-1).mean())
        loss_var = variance_floor_loss(preds, target_std.to(device))
        
        loss = loss_nce + loss_cos + 0.05 * loss_var
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
    return total_loss / max(num_batches, 1)

@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    for batch in loader:
        latents = batch["latent"].to(device)
        targets = batch["target"].to(device)
        subject_ids = batch["subject_id"].to(device)
        
        preds = model(latents, subject_id=subject_ids)
        
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
        
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute normalized evaluation metrics
    pred_eval = F.normalize(all_preds, dim=-1)
    target_eval = F.normalize(all_targets, dim=-1)
    
    metrics = retrieval_topk(pred_eval, target_eval)
    
    # Extract metric values
    mrr_norm = metrics["mrr"]
    top1_norm = metrics["top1"]
    top5_norm = metrics["top5"]
    top10_norm = metrics["top10"]
    cosine_norm = F.cosine_similarity(pred_eval, target_eval, dim=-1).mean().item()
    
    # Norm statistics
    pred_norms = all_preds.norm(dim=-1)
    val_pred_norm_mean = pred_norms.mean().item()
    val_pred_norm_std = pred_norms.std().item()
    
    # Pred std ratio (raw)
    pred_std = all_preds.std(dim=0).mean().item()
    target_std = all_targets.std(dim=0).mean().item()
    val_pred_std_ratio = pred_std / max(target_std, 1e-8)
    
    # collapse_pct
    pred_dims_std = all_preds.std(dim=0)
    target_dims_std = all_targets.std(dim=0)
    ratio = pred_dims_std / (target_dims_std + 1e-8)
    collapse_pct = float((ratio < 0.2).float().mean().item()) * 100.0
    
    eval_metrics = {
        "val_mrr_norm": mrr_norm,
        "val_top1_norm": top1_norm,
        "val_top5_norm": top5_norm,
        "val_top10_norm": top10_norm,
        "val_cosine_norm": cosine_norm,
        "val_pred_std_ratio": val_pred_std_ratio,
        "val_pred_norm_mean": val_pred_norm_mean,
        "val_pred_norm_std": val_pred_norm_std,
        "collapse_pct": collapse_pct,
        # backward compatibility keys for training loops
        "mrr": mrr_norm,
        "top1": top1_norm,
        "top10": top10_norm,
        "cosine": cosine_norm,
        "collapse_score": val_pred_std_ratio,
        "pred_std": pred_std,
        "target_std": target_std
    }
    return eval_metrics

@torch.no_grad()
def save_eval_metadata(model, loader, device, out_path):
    model.eval()
    all_preds = []
    all_targets = []
    sample_ids = []
    image_ids = []
    
    for batch in loader:
        latents = batch["latent"].to(device)
        targets = batch["target"].to(device)
        subject_ids = batch["subject_id"].to(device)
        
        preds = model(latents, subject_id=subject_ids)
        
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
        sample_ids.extend(batch["sample_id"])
        image_ids.extend(batch["image_id"])
        
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute rank and top10 hit against the entire set of targets
    pred_n = F.normalize(all_preds, dim=-1)
    tgt_n = F.normalize(all_targets, dim=-1)
    
    logits = pred_n @ tgt_n.T  # [N, N]
    n = pred_n.shape[0]
    truth = torch.arange(n)
    
    sorted_indices = logits.argsort(dim=-1, descending=True)
    rank_of_truth = (sorted_indices == truth[:, None]).nonzero(as_tuple=False)[:, 1].float()
    top10_hit = (rank_of_truth < 10).float()
    
    data = {
        "sample_id": sample_ids,
        "image_id": image_ids,
        "pred": all_preds,
        "target": all_targets,
        "rank": rank_of_truth,
        "top10_hit": top10_hit
    }
    torch.save(data, out_path)
    print(f"Saved evaluation metadata to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Train QFormer adapter mapping ZUNA latents to vision space.")
    # Dataset and paths
    parser.add_argument("--latents-pt", type=str, required=True, help="Path to cached latents.pt")
    parser.add_argument("--targets-pt", type=str, required=True, help="Path to targets.pt")
    parser.add_argument("--target-space", type=str, default="common", help="Target space name inside targets.pt")
    parser.add_argument("--target-mode", choices=("real", "shuffled", "random"), default="real", help="Mapping mode (controls)")
    parser.add_argument("--layer-name", type=str, default="post_mmd", help="Source ZUNA layer")
    
    # Explicit splits
    parser.add_argument("--train-runs", type=str, default=None, help="Train run list/range, e.g. 1-24")
    parser.add_argument("--val-runs", type=str, default=None, help="Val run list/range, e.g. 25-28")
    parser.add_argument("--test-runs", type=str, default=None, help="Test run list/range, e.g. 29-32")
    parser.add_argument("--subjects", type=str, default=None, help="Comma-separated subject IDs (e.g. 'sub-01') to filter")
    
    # QFormer architecture
    parser.add_argument("--num-query-tokens", type=int, default=32, help="Number of query tokens")
    parser.add_argument("--pooling-mode", choices=("cls", "attention", "mean"), default="cls", help="Query pooling mode")
    parser.add_argument("--hidden-dim", type=int, default=256, help="QFormer hidden dimension")
    parser.add_argument("--nhead", type=int, default=8, help="QFormer attention heads")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of QFormer layers")
    parser.add_argument("--dropout", type=float, default=0.15, help="QFormer dropout")
    parser.add_argument("--num-subjects", type=int, default=1, help="Number of subjects for FiLM embeddings")
    
    # Output stabilization head flags
    parser.add_argument("--output-layernorm", action="store_true", default=True, help="Enable LayerNorm in final head")
    parser.add_argument("--no-output-layernorm", action="store_false", dest="output_layernorm")
    parser.add_argument("--force-unit-output", action="store_true", default=True, help="Enable L2 normalization in final head")
    parser.add_argument("--no-force-unit-output", action="store_false", dest="force_unit_output")

    # Optimization
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--temperature", type=float, default=0.05, help="Contrastive InfoNCE temperature")
    
    # Output and execution
    parser.add_argument("--device", type=str, default="cuda", help="Torch device")
    parser.add_argument("--out-dir", type=str, default="outputs/qformer_aligned_grid", help="Directory to save checkpoints and logs")
    parser.add_argument("--slug", type=str, default=None, help="Optional experiment slug")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Parse subjects list
    subject_list = None
    if args.subjects:
        subject_list = [s.strip() for s in args.subjects.split(",") if s.strip()]
        
    # Map target spaces dynamically
    is_pca = "PCA" in args.target_space
    pca_dims = None
    target_space_key = args.target_space
    if is_pca:
        pca_dims = int(args.target_space.split("-")[2])
        target_space_key = "rae_unit"
    elif args.target_space in ("DINO-Unit-768", "DINO-Unit"):
        target_space_key = "rae_unit"
    elif args.target_space in ("CLIP-Common-512", "CLIP-Common"):
        target_space_key = "common"

    # Load dataset
    full_dataset = ZunaLatentTargetDataset(
        latents_pt_path=args.latents_pt,
        targets_pt_path=args.targets_pt,
        target_space=args.target_space,
        layer_name=args.layer_name,
        target_mode=args.target_mode,
        shuffle_seed=args.seed,
        subject_list=subject_list
    )
    
    # Explicit splits determination
    all_runs = sorted(list({int(r["run_id"]) for r in full_dataset.records}))
    max_run = max(all_runs) if all_runs else 32
    
    if args.train_runs:
        train_run_ids = parse_runs_spec(args.train_runs)
    else:
        train_run_ids = list(range(1, 25)) if max_run <= 32 else list(range(1, 33))
        
    if args.val_runs:
        val_run_ids = parse_runs_spec(args.val_runs)
    else:
        val_run_ids = list(range(25, 29)) if max_run <= 32 else list(range(33, 37))
        
    if args.test_runs:
        test_run_ids = parse_runs_spec(args.test_runs)
    else:
        test_run_ids = list(range(29, 33)) if max_run <= 32 else list(range(37, 41))

    train_indices = []
    val_indices = []
    test_indices = []
    
    for idx in range(len(full_dataset)):
        record = full_dataset.valid_records[idx]
        run_id = int(record["run_id"])
        if run_id in train_run_ids:
            train_indices.append(idx)
        elif run_id in val_run_ids:
            val_indices.append(idx)
        elif run_id in test_run_ids:
            test_indices.append(idx)
            
    print(f"Available dataset runs: {all_runs}")
    print(f"Split: Train={len(train_indices)} samples | Val={len(val_indices)} samples | Test={len(test_indices)} samples")
    print(f"Train runs: {train_run_ids}")
    print(f"Val runs: {val_run_ids}")
    print(f"Test runs: {test_run_ids}")
    
    if len(train_indices) == 0 or len(val_indices) == 0:
        raise ValueError(f"Invalid split: train={len(train_indices)} val={len(val_indices)}.")

    # Setup directories early to write PCA params
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{ts}_{args.target_mode}_{args.layer_name}_{args.target_space.replace('/', '_')}"
    if args.slug:
        experiment_name += f"_{args.slug}"
        
    run_dir = Path(args.out_dir) / experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoints and metrics to {run_dir}")

    # Fit PCA if target space is a PCA target
    if is_pca and args.target_mode != "random":
        train_image_ids = set()
        for idx in train_indices:
            rec = full_dataset.valid_records[idx]
            train_image_ids.add(rec["image_id"])
            
        train_image_ids = sorted(list(train_image_ids))
        train_targets = torch.stack([full_dataset.image_id_to_target[img_id] for img_id in train_image_ids])
        
        print(f"Fitting PCA with {full_dataset.pca_dims} components on {len(train_image_ids)} training images...")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=full_dataset.pca_dims)
        pca.fit(train_targets.numpy())
        
        # Transform all targets
        all_image_ids = list(full_dataset.image_id_to_target.keys())
        all_targets = torch.stack([full_dataset.image_id_to_target[img_id] for img_id in all_image_ids])
        all_transformed = pca.transform(all_targets.numpy())
        
        # Normalize transformed targets
        all_transformed_tensor = torch.from_numpy(all_transformed).float()
        all_transformed_tensor = F.normalize(all_transformed_tensor, dim=-1)
        
        # Update targets dict
        full_dataset.image_id_to_target = {
            img_id: all_transformed_tensor[i] for i, img_id in enumerate(all_image_ids)
        }
        
        # Save PCA params
        pca_params = {
            "mean": torch.from_numpy(pca.mean_).float(),
            "components": torch.from_numpy(pca.components_).float(),
            "explained_variance": torch.from_numpy(pca.explained_variance_).float(),
        }
        torch.save(pca_params, run_dir / "pca_params.pt")
        print(f"Saved PCA parameters to {run_dir / 'pca_params.pt'}")

    # Compute target_std from train split targets
    train_targets_list = []
    for idx in train_indices:
        train_targets_list.append(full_dataset[idx]["target"])
    train_targets_all = torch.stack(train_targets_list)
    target_std = train_targets_all.std(dim=0).clamp_min(1e-4)
    print(f"Calculated training target_std (mean={target_std.mean().item():.6f}, min={target_std.min().item():.6f}, max={target_std.max().item():.6f})")

    # Subset datasets and dataloaders
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    # Save training config with split info
    config_dict = vars(args)
    config_dict.update({
        "available_runs": all_runs,
        "train_runs": train_run_ids,
        "val_runs_list": val_run_ids,
        "test_runs_list": test_run_ids
    })
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
        
    # Instantiate stabilized QFormer adapter
    model = ZunaToVisionQFormer(
        d_in=full_dataset.latent_dim,
        d_out=full_dataset.target_dim,
        hidden_dim=args.hidden_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        num_query_tokens=args.num_query_tokens,
        pooling_mode=args.pooling_mode,
        dropout=args.dropout,
        num_subjects=args.num_subjects,
        output_layernorm=args.output_layernorm,
        force_unit_output=args.force_unit_output,
        normalize_output=False # disable old L2 normalizer
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized ZunaToVisionQFormer with {num_params:,} trainable parameters.")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Training loop
    best_mrr = -1.0
    history = []
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, args.temperature, target_std, device)
        scheduler.step()
        val_metrics = evaluate_model(model, val_loader, device)
        
        val_mrr = val_metrics["val_mrr_norm"]
        val_top10 = val_metrics["val_top10_norm"]
        val_cosine = val_metrics["val_cosine_norm"]
        std_ratio = val_metrics["val_pred_std_ratio"]
        collapse_pct = val_metrics["collapse_pct"]
        
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | "
              f"Val Cosine (Norm): {val_cosine:.4f} | Val MRR (Norm): {val_mrr:.4f} | "
              f"Val Top-10 (Norm): {val_top10:.4f} | StdRatio: {std_ratio:.3f} | Collapse: {collapse_pct:.1f}%")
              
        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics
        }
        history.append(history_row)
        
        # Save checkpoint if MRR improves
        is_best = val_mrr > best_mrr
        if is_best:
            best_mrr = val_mrr
            patience_counter = 0
            checkpoint_path = run_dir / "checkpoint_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mrr": val_mrr,
                "config": config_dict
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⏹ Early stopping at epoch {epoch} (patience={args.patience}, best MRR={best_mrr:.4f})")
                break
            
    # Save final model state
    torch.save(model.state_dict(), run_dir / "model_final.pt")
    
    # Save history to CSV
    df_history = pd.DataFrame(history)
    df_history.to_csv(run_dir / "history.csv", index=False)
    
    # Save final metrics summary
    final_metrics = history[-1]
    best_epoch_idx = df_history["val_mrr_norm"].idxmax()
    best_metrics = history[best_epoch_idx]
    
    summary = {
        "final": final_metrics,
        "best": best_metrics
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\n✓ Training complete! Best Validation MRR (Norm): {best_mrr:.4f} at epoch {best_metrics['epoch']}.")
    
    # Load best checkpoint and save predictions/targets with full sample metadata
    save_eval_metadata(model, val_loader, device, run_dir / "val_eval_preds.pt")
    if len(test_dataset) > 0:
        save_eval_metadata(model, test_loader, device, run_dir / "test_eval_preds.pt")
        
    print(f"Results saved in: {run_dir}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pretrain CommonProbeModel on visual common embeddings and VLM attributes."""

import argparse
import csv
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindseye.models.common_probe import CommonProbeModel, ATTRIBUTE_SCHEMAS, IGNORE_INDEX

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--metadata",
        default="data/processed/semantic_epochs/zuna_real_sub01_runs01_05/all_runs_metadata.csv",
        help="Crop metadata CSV or comma-separated CSV list",
    )
    p.add_argument(
        "--common-embeddings",
        required=True,
        help="Fused common embeddings .pt file",
    )
    p.add_argument(
        "--vlm-attributes",
        required=True,
        help="vlm_attributes.json file",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/common_probe",
        help="Output directory",
    )
    p.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    p.add_argument("--val-fraction", type=float, default=0.15, help="Fraction for validation split")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--device", default=None, help="cuda, cpu, or auto")
    p.add_argument(
        "--target-key",
        default="common",
        help="Key to read from the embeddings .pt file: 'common' (default) or 'decode_unit'",
    )
    return p.parse_args()

def split_paths(path_str) -> list[str]:
    if not path_str:
        return []
    return [p.strip() for p in str(path_str).split(",")]

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # Create output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load metadata CSV(s)
    metadata_paths = split_paths(args.metadata)
    dfs = []
    for path in metadata_paths:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Metadata file not found: {p}")
        dfs.append(pd.read_csv(p))
    metadata_df = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
    print(f"Loaded {len(metadata_df)} metadata rows.")
    
    # Get unique class names
    if "class" not in metadata_df.columns:
        raise ValueError("Metadata CSV missing 'class' column.")
    
    unique_classes = sorted(metadata_df["class"].dropna().unique().tolist())
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    idx_to_class = unique_classes
    print(f"Found {len(class_to_idx)} unique classes.")
    
    # Map image_id to class string
    df_unique = metadata_df.drop_duplicates(subset=["image_id"])
    img_to_class = dict(zip(df_unique["image_id"].astype(str), df_unique["class"].astype(str)))
    
    # 2. Load common embeddings
    embeddings_path = Path(args.common_embeddings)
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Common embeddings file not found: {embeddings_path}")
    embeddings_table = torch.load(embeddings_path, map_location="cpu")
    
    target_key = f"image_id_to_{args.target_key}"
    if target_key not in embeddings_table:
        available = [k for k in embeddings_table.keys() if k.startswith("image_id_to_")]
        raise KeyError(f"Key '{target_key}' not found in embeddings. Available: {available}")
    image_id_to_target = embeddings_table[target_key]
    embedding_dim = next(iter(image_id_to_target.values())).shape[-1]
    print(f"Loaded '{target_key}' embeddings for {len(image_id_to_target)} images (dimension={embedding_dim}).")
    # 3. Load VLM attributes
    vlm_path = Path(args.vlm_attributes)
    if not vlm_path.exists():
        raise FileNotFoundError(f"VLM attributes file not found: {vlm_path}")
    with open(vlm_path, "r") as f:
        vlm_attributes = json.load(f)
    print(f"Loaded VLM attributes for {len(vlm_attributes)} images.")
    
    # Filter image IDs to those that have embeddings
    image_ids = sorted(list(image_id_to_target.keys()))
    
    # Split train/val by image ID
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(image_ids), generator=g).tolist()
    n_val = max(1, int(round(len(image_ids) * args.val_fraction)))
    val_image_ids = set([image_ids[i] for i in perm[:n_val]])
    train_image_ids = set([image_ids[i] for i in perm[n_val:]])
    print(f"Split: {len(train_image_ids)} train images, {len(val_image_ids)} val images.")
    
    # Define tasks
    tasks = ["class_label"] + list(ATTRIBUTE_SCHEMAS.keys())
    task_specs = {"class_label": len(class_to_idx)}
    for attr, choices in ATTRIBUTE_SCHEMAS.items():
        task_specs[attr] = len(choices)
        
    # Prepare datasets
    def build_tensors(image_id_list):
        embeds = []
        targets = {task: [] for task in tasks}
        
        for img_id in image_id_list:
            embeds.append(F.normalize(image_id_to_target[img_id].float(), dim=-1))
            
            # class_label target
            cls_val = img_to_class.get(img_id, None)
            if cls_val is None or cls_val not in class_to_idx:
                targets["class_label"].append(IGNORE_INDEX)
            else:
                targets["class_label"].append(class_to_idx[cls_val])
                
            # attribute targets
            img_attrs = vlm_attributes.get(img_id, {})
            for attr in ATTRIBUTE_SCHEMAS.keys():
                val = img_attrs.get(attr, "unclear")
                targets[attr].append(CommonProbeModel.encode_label(attr, val))
                
        embeds_t = torch.stack(embeds)
        targets_t = {task: torch.tensor(targets[task], dtype=torch.long) for task in tasks}
        return embeds_t, targets_t

    train_embeds, train_targets = build_tensors(sorted(list(train_image_ids)))
    val_embeds, val_targets = build_tensors(sorted(list(val_image_ids)))
    
    # Build PyTorch DataLoaders
    # PyTorch's TensorDataset only accepts tensors, so we pass index and query targets dictionary in the loop
    train_ds = TensorDataset(train_embeds, torch.arange(len(train_embeds)))
    val_ds = TensorDataset(val_embeds, torch.arange(len(val_embeds)))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # Initialize Model
    model = CommonProbeModel(embedding_dim=embedding_dim, task_specs=task_specs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Train
    print("Starting pretraining...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for batch_embeds, batch_indices in train_loader:
            batch_embeds = batch_embeds.to(device)
            optimizer.zero_grad()
            
            logits_dict = model(batch_embeds)
            
            loss = 0.0
            n_active = 0
            for task in tasks:
                task_targets = train_targets[task][batch_indices].to(device)
                task_loss = F.cross_entropy(logits_dict[task], task_targets, ignore_index=IGNORE_INDEX)
                if not torch.isnan(task_loss):
                    loss = loss + task_loss
                    n_active += 1
            # Normalize by number of active tasks to prevent loss magnitude scaling with task count
            if n_active > 0:
                loss = loss / n_active
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(batch_embeds)
            
        epoch_loss /= len(train_embeds)
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch}/{args.epochs} - Loss: {epoch_loss:.4f}")
            
    # Evaluation
    model.eval()
    val_preds = {task: [] for task in tasks}
    
    with torch.no_grad():
        for batch_embeds, _ in val_loader:
            batch_embeds = batch_embeds.to(device)
            logits_dict = model(batch_embeds)
            for task in tasks:
                val_preds[task].append(logits_dict[task].cpu())
                
    for task in tasks:
        val_preds[task] = torch.cat(val_preds[task])
        
    # Compute Metrics per task
    metrics_per_task = {}
    active_tasks = []
    
    for task in tasks:
        y_pred_logits = val_preds[task]
        y_true = val_targets[task]
        num_classes = task_specs[task]
        
        # Filter out ignored indices
        valid_mask = y_true != IGNORE_INDEX
        n_valid = int(valid_mask.sum().item())
        total_samples = len(y_true)
        
        coverage = n_valid / total_samples if total_samples > 0 else 0.0
        
        if n_valid == 0:
            print(f"Task {task}: No valid samples in validation split. Gating out.")
            metrics_per_task[task] = {
                "coverage": coverage,
                "majority_baseline": 0.0,
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "macro_f1": 0.0,
                "confusion_matrix": []
            }
            continue
            
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred_logits[valid_mask].argmax(dim=-1)
        
        # Calculate majority baseline from training targets
        train_task_targets = train_targets[task]
        train_valid_mask = train_task_targets != IGNORE_INDEX
        if train_valid_mask.any():
            train_targets_valid = train_task_targets[train_valid_mask]
            # Find mode
            classes, counts = torch.unique(train_targets_valid, return_counts=True)
            majority_class = classes[counts.argmax()].item()
        else:
            majority_class = 0
            
        majority_baseline = (y_true_valid == majority_class).float().mean().item()
        accuracy = (y_pred_valid == y_true_valid).float().mean().item()
        
        # Balanced accuracy
        unique_classes_val = torch.unique(y_true_valid)
        recalls = []
        for c in range(num_classes):
            class_mask = (y_true_valid == c)
            if class_mask.sum() > 0:
                recalls.append((y_pred_valid[class_mask] == c).float().mean().item())
        balanced_accuracy = sum(recalls) / len(recalls) if recalls else 0.0
        
        # Macro F1
        f1s = []
        for c in range(num_classes):
            tp = ((y_pred_valid == c) & (y_true_valid == c)).sum().item()
            fp = ((y_pred_valid == c) & (y_true_valid != c)).sum().item()
            fn = ((y_pred_valid != c) & (y_true_valid == c)).sum().item()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            f1s.append(f1)
        macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
        
        # Confusion matrix
        cm = torch.zeros(num_classes, num_classes, dtype=torch.int32)
        for t, p in zip(y_true_valid, y_pred_valid):
            cm[t.item(), p.item()] += 1
            
        metrics_per_task[task] = {
            "coverage": coverage,
            "majority_baseline": majority_baseline,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": macro_f1,
            "confusion_matrix": cm.tolist()
        }
        
        print(f"Task {task:25s} | Coverage: {coverage:.3f} | Accuracy: {accuracy:.3f} (Baseline: {majority_baseline:.3f})")
        
        # Gating check
        if accuracy > majority_baseline:
            active_tasks.append(task)
            
    print(f"Active tasks (beating baseline): {active_tasks}")
    
    # Save files
    # 1. Save class mappings / label maps
    label_maps = {
        "class_label": class_to_idx,
    }
    for attr, choices in ATTRIBUTE_SCHEMAS.items():
        label_maps[attr] = {c: i for i, c in enumerate(choices)}
        
    with open(out_dir / "label_maps.json", "w") as f:
        json.dump(label_maps, f, indent=2)
        
    with open(out_dir / "class_mappings.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)
        
    # 2. Save task specs (active tasks only)
    active_task_specs = {task: task_specs[task] for task in active_tasks}
    with open(out_dir / "task_specs.json", "w") as f:
        json.dump(active_task_specs, f, indent=2)
        
    # 3. Save metrics.json
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_per_task, f, indent=2)
        
    # 4. Save per_task_metrics.csv
    csv_path = out_dir / "per_task_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "coverage", "majority_baseline", "accuracy", "balanced_accuracy", "macro_f1"])
        for task in tasks:
            m = metrics_per_task[task]
            w.writerow([
                task,
                f"{m['coverage']:.4f}",
                f"{m['majority_baseline']:.4f}",
                f"{m['accuracy']:.4f}",
                f"{m['balanced_accuracy']:.4f}",
                f"{m['macro_f1']:.4f}"
            ])
            
    # 5. Save model checkpoint (with only active heads & trunk)
    # We construct a new CommonProbeModel with active tasks only to filter weights cleanly
    active_model = CommonProbeModel(embedding_dim=embedding_dim, task_specs=active_task_specs)
    
    # Copy weights
    state_dict = model.state_dict()
    active_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("heads."):
            task_name = k.split(".")[1]
            if task_name in active_tasks:
                active_state_dict[k] = v
        else:
            active_state_dict[k] = v
            
    active_model.load_state_dict(active_state_dict)
    torch.save(active_model.state_dict(), out_dir / "common_probe.pt")
    
    print(f"Pretraining complete. Saved checkpoint and metrics to {out_dir}")

if __name__ == "__main__":
    main()

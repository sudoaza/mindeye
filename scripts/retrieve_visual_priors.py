#!/usr/bin/env python3
"""Query the FAISS index with predicted embeddings from a trained EEG encoder to retrieve visual priors.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindseye.embeddings.faiss_index import FAISSIndex
from mindseye.datasets.semantic_pairs import SemanticPairConfig, ZunaClipPairDataset, split_indices

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained model checkpoint (.pt)",
    )
    p.add_argument(
        "--index-prefix",
        required=True,
        help="Prefix path of the FAISS index and metadata",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output JSON file to save retrieved visual priors",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of nearest neighbors to retrieve (default: 10)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="cuda, cpu, or omitted for auto",
    )
    # Optional dataset overrides
    p.add_argument("--metadata", default=None, help="Override crop metadata CSV path")
    p.add_argument("--epochs-dir", default=None, help="Override crop NPZ epochs directory")
    p.add_argument("--common-embeddings", default=None, help="Override common embeddings .pt path")
    return p.parse_args()


def _parse_val_runs(value: str) -> set[int]:
    return {int(x.strip()) for x in value.split(",") if x.strip()}


def _split_by_run(dataset, val_runs: set[int]) -> tuple[list[int], list[int]]:
    if "run" not in dataset.metadata.columns:
        raise ValueError("run split requires a 'run' column in metadata")
    train_idx: list[int] = []
    val_idx: list[int] = []
    for idx, run in enumerate(dataset.metadata["run"].astype(int).tolist()):
        (val_idx if run in val_runs else train_idx).append(idx)
    return train_idx, val_idx


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    setup = checkpoint.get("setup", {})
    
    # Read training config if available in parent directory
    checkpoint_path = Path(args.checkpoint)
    config_json_path = checkpoint_path.parent / "config.json"
    train_config = {}
    if config_json_path.exists():
        with open(config_json_path, "r") as f:
            train_config = json.load(f)
            
    # Resolve dataset file paths
    metadata_csv = args.metadata or train_config.get("metadata") or setup.get("metadata")
    epochs_dir = args.epochs_dir or train_config.get("epochs_dir") or setup.get("epochs_dir")
    common_embeddings_pt = args.common_embeddings or train_config.get("common_embeddings") or setup.get("common_embeddings")
    
    if not metadata_csv or not epochs_dir or not common_embeddings_pt:
        raise ValueError(
            "Could not resolve dataset paths. Please supply --metadata, --epochs-dir, and --common-embeddings overrides."
        )
        
    print(f"Loading validation dataset using config metadata={metadata_csv}...")
    dataset = ZunaClipPairDataset(
        SemanticPairConfig(
            metadata_csv=metadata_csv,
            epochs_dir=epochs_dir,
            common_embeddings_pt=common_embeddings_pt,
            epochs_dir_raw=train_config.get("epochs_dir_raw"),
            epochs_dir_resample=train_config.get("epochs_dir_resample"),
            vlm_attributes_json=train_config.get("vlm_attributes"),
            input_domain=train_config.get("input_domain") or setup.get("input_domain", "zuna"),
            target_mode=train_config.get("target_mode") or setup.get("target_mode", "real"),
            window_mode=train_config.get("window_mode") or setup.get("window_mode", "crop"),
            target_space=train_config.get("target_space") or setup.get("target_space", "common"),
            add_event_marker=train_config.get("add_event_marker", setup.get("add_event_marker", False)),
            augment_eeg=False,
        )
    )
    
    # Resolve validation indices
    split_mode = train_config.get("split_mode") or setup.get("split_mode", "random")
    seed = train_config.get("seed") or setup.get("seed", 13)
    val_fraction = train_config.get("val_fraction") or setup.get("val_fraction", 0.15)
    val_runs_val = train_config.get("val_runs") or setup.get("val_runs")
    
    if split_mode == "run":
        if not val_runs_val:
            raise ValueError("run split requested but no validation runs were found in checkpoint or training config")
        val_runs = _parse_val_runs(val_runs_val) if isinstance(val_runs_val, str) else set(val_runs_val)
        _, val_idx = _split_by_run(dataset, val_runs)
    else:
        _, val_idx = split_indices(len(dataset), val_fraction=val_fraction, seed=seed)
        
    print(f"Recreated validation split: {len(val_idx)} validation trials.")
    
    # Reconstruct and load model
    model_type = setup.get("model") or train_config.get("model", "cnn")
    n_channels, n_times = dataset.eeg_shape
    
    print(f"Recreating encoder model architecture: '{model_type}'...")
    if model_type in {"spatial_temporal", "spatial_temporal_small"}:
        from mindseye.models.spatial_temporal_encoder import build_spatial_temporal_encoder
        preset = "small" if model_type == "spatial_temporal_small" else "medium"
        overrides = {}
        for key in ("hidden_dim", "n_layers", "n_heads", "dropout"):
            val = train_config.get(key)
            if val is not None:
                overrides[key] = val
        overrides["stem_dropout"] = train_config.get("stem_dropout1d", 0.15)
        overrides["spatial_mixing"] = not train_config.get("no_spatial_mixing", False)
        
        model = build_spatial_temporal_encoder(
            preset,
            n_channels=n_channels,
            embedding_dim=dataset.embedding_dim,
            ch_names=getattr(dataset, "ch_names", None),
            **overrides,
        ).to(device)
    elif model_type in {"temporal_attn", "temporal_attn_small"}:
        from mindseye.models.eeg_encoder import TemporalAttnEncoder
        hidden_dim = train_config.get("hidden_dim")
        n_layers = train_config.get("n_layers")
        n_heads = train_config.get("n_heads")
        dropout = train_config.get("dropout")
        if model_type == "temporal_attn_small":
            hidden_dim = hidden_dim or 128
            n_layers = n_layers or 2
            n_heads = n_heads or 4
            dropout = 0.35 if dropout is None else dropout
        else:
            hidden_dim = hidden_dim or 256
            n_layers = n_layers or 4
            n_heads = n_heads or 8
            dropout = 0.2 if dropout is None else dropout
        model = TemporalAttnEncoder(
            n_channels=n_channels,
            embedding_dim=dataset.embedding_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            stem_dropout1d=train_config.get("stem_dropout1d", 0.15),
        ).to(device)
    else:
        from mindseye.models.eeg_encoder import EEGClipEncoder
        hidden_dim = train_config.get("hidden_dim") or 256
        dropout = train_config.get("dropout")
        dropout = 0.2 if dropout is None else dropout
        model = EEGClipEncoder(
            n_channels=n_channels,
            n_times=n_times,
            embedding_dim=dataset.embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            stem_dropout1d=train_config.get("stem_dropout1d", 0.15),
        ).to(device)
        
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    # Load FAISS index
    print(f"Loading FAISS index from {args.index_prefix}...")
    index = FAISSIndex.load(args.index_prefix)
    
    # Run evaluation
    from torch.utils.data import DataLoader, Subset
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=64, shuffle=False)
    
    print("Computing predicted EEG embeddings...")
    preds = []
    ground_truth_ids = []
    
    with torch.no_grad():
        for batch in val_loader:
            eeg = batch["eeg"].to(device).float()
            pred = model(eeg)
            if isinstance(pred, tuple):
                pred = pred[0]
            preds.append(pred.cpu())
            ground_truth_ids.extend(batch["image_id"])
            
    preds = torch.cat(preds)
    
    print(f"Querying FAISS index for top-{args.top_k} visual priors...")
    distances, retrieved_ids = index.search(preds, k=args.top_k)
    
    # Write retrieved priors results
    results = []
    for i in range(len(val_idx)):
        results.append({
            "trial_index": int(val_idx[i]),
            "ground_truth_image_id": ground_truth_ids[i],
            "retrieved_results": [
                {
                    "rank": r + 1,
                    "image_id": retrieved_ids[i][r],
                    "score": float(distances[i][r])
                }
                for r in range(args.top_k)
            ]
        })
        
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Successfully saved retrieved visual priors to: {output_path}")
    
    # 1. Validation target bank check (using a temporary FAISS index built on-the-fly)
    print("\nComputing retrieval metrics against the validation-only target bank...")
    val_temp_index = FAISSIndex(dimension=dataset.embedding_dim, metric="cosine")
    val_targets = torch.stack([dataset._get_targets(idx) for idx in val_idx])
    val_temp_index.add(val_targets, [str(i) for i in range(len(val_idx))])
    
    val_distances, val_retrieved_ids = val_temp_index.search(preds, k=len(val_idx))
    val_ranks = []
    for i in range(len(preds)):
        rank = val_retrieved_ids[i].index(str(i))
        val_ranks.append(rank)
        
    val_ranks_t = torch.tensor(val_ranks, dtype=torch.float32)
    val_top1 = (val_ranks_t < 1).float().mean().item()
    val_top5 = (val_ranks_t < 5).float().mean().item()
    val_top10 = (val_ranks_t < 10).float().mean().item()
    val_mrr = (1.0 / (val_ranks_t + 1.0)).mean().item()
    val_median_rank = float(val_ranks_t.median().item() + 1)
    
    print("\n=== FAISS Validation Bank Metrics (Match Checkpoint) ===")
    print(f"Top-1:       {val_top1:.6f}")
    print(f"Top-5:       {val_top5:.6f}")
    print(f"Top-10:      {val_top10:.6f}")
    print(f"MRR:         {val_mrr:.6f}")
    print(f"Median Rank: {val_median_rank}")
    
    if "metrics" in checkpoint:
        ckpt_metrics = checkpoint["metrics"]
        print("\n=== Checkpoint Logged Metrics ===")
        print(f"Top-1:       {ckpt_metrics.get('top1', 'N/A')}")
        print(f"Top-5:       {ckpt_metrics.get('top5', 'N/A')}")
        print(f"Top-10:      {ckpt_metrics.get('top10', 'N/A')}")
        print(f"MRR:         {ckpt_metrics.get('mrr', 'N/A')}")
        print(f"Median Rank: {ckpt_metrics.get('median_rank', 'N/A')}")
        
    # 2. General FAISS index check
    print("\nRunning retrieval metrics sanity check against full FAISS index...")
    distances_full, retrieved_ids_full = index.search(preds, k=len(index.ids))
    
    ranks = []
    found_gt = 0
    for i, gt_id in enumerate(ground_truth_ids):
        try:
            rank = retrieved_ids_full[i].index(gt_id)
            ranks.append(rank)
            found_gt += 1
        except ValueError:
            pass
            
    if found_gt == len(ground_truth_ids):
        ranks_t = torch.tensor(ranks, dtype=torch.float32)
        top1 = (ranks_t < 1).float().mean().item()
        top5 = (ranks_t < 5).float().mean().item()
        top10 = (ranks_t < 10).float().mean().item()
        mrr = (1.0 / (ranks_t + 1.0)).mean().item()
        median_rank = float(ranks_t.median().item() + 1)
        
        print("\n=== FAISS Full Index Retrieval Metrics (All Candidates) ===")
        print(f"Top-1:       {top1:.6f}")
        print(f"Top-5:       {top5:.6f}")
        print(f"Top-10:      {top10:.6f}")
        print(f"MRR:         {mrr:.6f}")
        print(f"Median Rank: {median_rank}")
    else:
        print(f"\nWarning: Mapped only {found_gt}/{len(ground_truth_ids)} ground truth IDs in FAISS index. Skipping full metrics check.")


if __name__ == "__main__":
    main()

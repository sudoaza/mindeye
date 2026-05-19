#!/usr/bin/env python3
"""Train a baseline EEG→CLIP encoder on ZUNA semantic crops.

Supports all 6 conditions of the baseline matrix via --input-domain and
--target-mode flags.  Every run automatically creates a structured output
directory under outputs/runs/YYYYMMDD_HHMMSS_<slug>/ with full metrics,
history, checkpoint, and environment info.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--metadata",
        default="data/processed/semantic_epochs/zuna_real_sub01_runs01_05/all_runs_metadata.csv",
        help="Crop metadata CSV from run_cropper.py",
    )
    p.add_argument(
        "--epochs-dir",
        default="data/processed/semantic_epochs/zuna_real_sub01_runs01_05",
        help="Directory containing ZUNA-denoised per-run semantic NPZ files",
    )
    p.add_argument(
        "--epochs-dir-raw",
        default=None,
        help="Directory containing raw (un-denoised) crop NPZ files (for raw_* conditions)",
    )
    p.add_argument(
        "--epochs-dir-resample",
        default=None,
        help="Directory containing resample-only crop NPZ files",
    )
    p.add_argument(
        "--common-embeddings",
        required=True,
        help="Fused common embeddings .pt from build_common_embeddings.py",
    )
    p.add_argument(
        "--input-domain",
        choices=("zuna", "raw", "resample"),
        default="zuna",
        help="Which EEG input to use: zuna (default), raw, or resample-only",
    )
    p.add_argument(
        "--target-mode",
        choices=("real", "shuffled", "random", "sameclass"),
        default="real",
        help="Target mapping mode: real (default), shuffled, random, or sameclass distractors",
    )
    p.add_argument(
        "--target-space",
        choices=("common", "semantic", "image", "label"),
        default="common",
        help="Which embedding space to optimize the loss against",
    )
    p.add_argument(
        "--w-label",
        type=float,
        default=0.2,
        help="Weight for the auxiliary multiple-choice label classification loss",
    )
    p.add_argument("--output-dir", default=None,
                   help="Base output directory. Defaults to outputs/runs/")
    p.add_argument("--slug", default=None,
                   help="Optional slug appended to the run directory name")
    p.add_argument("--window-mode", choices=("crop", "full5s", "full5s_backaligned", "tight1s"), default="crop",
                   help="EEG window duration: crop (1.25s) or full5s (5s) or full5s_backaligned (5s) or tight1s (1.2s)")
    p.add_argument("--add-event-marker", action="store_true",
                   help="Add event marker bump as an extra channel to EEG inputs")
    p.add_argument("--model", choices=("cnn", "temporal_attn", "temporal_attn_small",
                                       "spatial_temporal", "spatial_temporal_small"), default="cnn",
                   help="Encoder architecture")
    p.add_argument("--hidden-dim", type=int, default=None,
                   help="Override encoder hidden width")
    p.add_argument("--n-layers", type=int, default=None,
                   help="Override TemporalAttn transformer layer count")
    p.add_argument("--n-heads", type=int, default=None,
                   help="Override TemporalAttn attention head count")
    p.add_argument("--dropout", type=float, default=None,
                   help="Override encoder/head dropout")
    p.add_argument("--stem-dropout1d", type=float, default=0.15,
                   help="Dropout1d probability in convolutional EEG stem")
    p.add_argument("--augment-eeg", action="store_true",
                   help="Apply train-time EEG augmentations (marker channel is never augmented)")
    p.add_argument("--aug-channel-dropout", type=float, default=0.10)
    p.add_argument("--aug-noise-std", type=float, default=0.03)
    p.add_argument("--aug-amp-scale", type=float, default=0.10)
    p.add_argument("--aug-time-mask", type=int, default=24)
    p.add_argument("--aug-time-jitter", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--loss", choices=("contrastive", "cosine_mse"), default="contrastive")
    p.add_argument("--temperature", type=float, default=0.07,
                   help="InfoNCE temperature for --loss contrastive")
    p.add_argument("--split-mode", choices=("random", "run"), default="random",
                   help="Random item split or hold-out full stimulus runs for validation")
    p.add_argument("--val-runs", default="5",
                   help="Comma-separated run numbers for --split-mode run")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--device", default=None, help="cuda, cpu, or omitted for auto")
    p.add_argument("--dry-run", action="store_true",
                   help="Load data/model and run one forward pass only")
    args = p.parse_args()
    if args.model in {"temporal_attn_small", "spatial_temporal_small", "spatial_temporal"} and args.weight_decay == 1e-4:
        args.weight_decay = 1e-2
    return args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_val_runs(value: str) -> set[int]:
    return {int(x.strip()) for x in value.split(",") if x.strip()}


def _split_by_run(dataset, val_runs: set[int]) -> tuple[list[int], list[int]]:
    if "run" not in dataset.metadata.columns:
        raise ValueError("--split-mode run requires a 'run' column in metadata")
    train_idx, val_idx = [], []
    for idx, run in enumerate(dataset.metadata["run"].astype(int).tolist()):
        (val_idx if run in val_runs else train_idx).append(idx)
    if not train_idx or not val_idx:
        raise ValueError(
            f"Invalid run split: train={len(train_idx)} val={len(val_idx)} "
            f"for val_runs={sorted(val_runs)}"
        )
    return train_idx, val_idx


def _batch_to_device(
    batch: dict,
    device,
) -> dict:
    ret = {
        "eeg": batch["eeg"].to(device).float(),
        "target": batch["target"].to(device).float()
    }
    if "true_label" in batch:
        ret["true_label"] = batch["true_label"].to(device).float()
        ret["distractor_labels"] = batch["distractor_labels"].to(device).float()
    return ret


def _loss_fn(pred, target, *, loss_name: str, temperature: float):
    if loss_name == "contrastive":
        from mindseye.models.eeg_encoder import clip_contrastive_loss
        return clip_contrastive_loss(pred, target, temperature=temperature)
    from mindseye.models.eeg_encoder import cosine_mse_loss
    return cosine_mse_loss(pred, target)


# ---------------------------------------------------------------------------
# Structured run directory
# ---------------------------------------------------------------------------

def _make_run_dir(args: argparse.Namespace) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug_parts = [args.input_domain, args.target_mode]
    if args.slug:
        slug_parts.append(args.slug)
    slug = "_".join(slug_parts)
    base = Path(args.output_dir) if args.output_dir else Path("outputs/runs")
    run_dir = base / f"{ts}_{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_env(run_dir: Path, args: argparse.Namespace, setup: dict) -> None:
    """Write environment.txt, git_commit.txt, and config.json."""
    (run_dir / "config.json").write_text(
        json.dumps({**vars(args), **{"setup": setup}}, indent=2, default=str)
    )
    env_lines = [
        f"Hostname: {os.uname().nodename}",
        f"Python: {sys.version}",
        f"Command: {' '.join(sys.argv)}",
    ]
    try:
        import torch as _torch
        env_lines.append(f"PyTorch: {_torch.__version__}")
        env_lines.append(f"CUDA Available: {_torch.cuda.is_available()}")
        if _torch.cuda.is_available():
            env_lines.append(f"CUDA Device: {_torch.cuda.get_device_name(0)}")
    except Exception:
        pass
    (run_dir / "environment.txt").write_text("\n".join(env_lines) + "\n")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        (run_dir / "git_commit.txt").write_text(commit + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, loader, device, *, loss_name: str, temperature: float, target_space: str) -> dict[str, float]:
    from mindseye.models.eeg_encoder import retrieval_topk
    import torch
    model.eval()
    preds, targets = [], []
    with torch.inference_mode():
        for batch in loader:
            batch_data = _batch_to_device(batch, device)
            pred = model(batch_data["eeg"])
            preds.append(pred.cpu())
            targets.append(batch_data["target"].cpu())

    pred_t = torch.cat(preds)
    target_t = torch.cat(targets)
    
    loss = float(_loss_fn(pred_t, target_t, loss_name=loss_name, temperature=temperature).item())
    
    metrics = {"loss": loss, "n": int(pred_t.shape[0])}
    
    # evaluate top-k
    m = retrieval_topk(pred_t, target_t)
    for k, v in m.items():
        metrics[k] = v
        
    metrics["mean_diag_cosine"] = float(
        (
            torch.nn.functional.normalize(pred_t, dim=-1)
            * torch.nn.functional.normalize(target_t, dim=-1)
        )
        .sum(dim=-1)
        .mean()
        .item()
    )

    # Expected random baselines
    n = metrics["n"]
    for k in (1, 5, 10):
        metrics[f"random_top{k}_expected"] = min(k, n) / n

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    global torch, DataLoader, Subset
    global SemanticPairConfig, ZunaClipPairDataset, split_indices
    global EEGClipEncoder
    import torch
    from torch.utils.data import DataLoader, Subset

    from mindseye.datasets.semantic_pairs import SemanticPairConfig, ZunaClipPairDataset, split_indices
    from mindseye.models.eeg_encoder import EEGClipEncoder

    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_config = SemanticPairConfig(
            metadata_csv=args.metadata,
            epochs_dir=args.epochs_dir,
            epochs_dir_raw=args.epochs_dir_raw,
            epochs_dir_resample=args.epochs_dir_resample,
            common_embeddings_pt=args.common_embeddings,
            input_domain=args.input_domain,
            target_mode=args.target_mode,
            window_mode=args.window_mode,
            target_space=args.target_space,
            add_event_marker=args.add_event_marker,
            augment_eeg=False,
        )
    dataset = ZunaClipPairDataset(dataset_config)
    target_bank_audit = dataset.audit_target_banks()
    print("[TargetBankAudit] " + json.dumps(target_bank_audit, sort_keys=True))

    train_dataset = dataset
    if args.augment_eeg:
        train_dataset = ZunaClipPairDataset(
            SemanticPairConfig(
                **{**dataset_config.__dict__,
                   "augment_eeg": True,
                   "aug_channel_dropout": args.aug_channel_dropout,
                   "aug_noise_std": args.aug_noise_std,
                   "aug_amp_scale": args.aug_amp_scale,
                   "aug_time_mask": args.aug_time_mask,
                   "aug_time_jitter": args.aug_time_jitter}
            )
        )

    if args.split_mode == "run":
        train_idx, val_idx = _split_by_run(dataset, _parse_val_runs(args.val_runs))
    else:
        train_idx, val_idx = split_indices(len(dataset), val_fraction=args.val_fraction, seed=args.seed)

    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=args.loss == "contrastive",
    )
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False)

    n_channels, n_times = dataset.eeg_shape
    print(f"\n[Dataset] window_mode: {args.window_mode}")
    print(f"[Dataset] target_space: {args.target_space}")
    print(f"[Dataset] add_event_marker: {args.add_event_marker}")
    print(f"[Dataset] EEG shape: [{n_channels}, {n_times}]")
    print(f"[Dataset] n_samples: {len(dataset)}")

    if args.model in {"spatial_temporal", "spatial_temporal_small"}:
        from mindseye.models.spatial_temporal_encoder import build_spatial_temporal_encoder
        preset = "small" if args.model == "spatial_temporal_small" else "medium"
        overrides = {}
        if args.hidden_dim is not None:
            overrides["hidden_dim"] = args.hidden_dim
        if args.n_layers is not None:
            overrides["n_layers"] = args.n_layers
        if args.n_heads is not None:
            overrides["n_heads"] = args.n_heads
        if args.dropout is not None:
            overrides["dropout"] = args.dropout
        overrides["stem_dropout"] = args.stem_dropout1d
        model = build_spatial_temporal_encoder(
            preset,
            n_channels=n_channels,
            embedding_dim=dataset.embedding_dim,
            ch_names=getattr(dataset, "ch_names", None),
            **overrides,
        ).to(device)
        hidden_dim = model.hidden_dim
        n_layers = len(model.spatial_transformer.layers)
        n_heads = model.spatial_transformer.layers[0].self_attn.num_heads
        dropout = args.dropout if args.dropout is not None else (0.35 if preset == "small" else 0.25)
        model.n_channels = n_channels
        print(f"[Model] model: {args.model} (preset={preset}) hidden_dim={hidden_dim} "
              f"n_layers={n_layers} n_heads={n_heads} dropout={dropout}")
    elif args.model in {"temporal_attn", "temporal_attn_small"}:
        from mindseye.models.eeg_encoder import TemporalAttnEncoder
        if args.model == "temporal_attn_small":
            hidden_dim = args.hidden_dim or 128
            n_layers = args.n_layers or 2
            n_heads = args.n_heads or 4
            dropout = args.dropout if args.dropout is not None else 0.35
        else:
            hidden_dim = args.hidden_dim or 256
            n_layers = args.n_layers or 4
            n_heads = args.n_heads or 8
            dropout = args.dropout if args.dropout is not None else 0.2
        model = TemporalAttnEncoder(
            n_channels=n_channels,
            embedding_dim=dataset.embedding_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            stem_dropout1d=args.stem_dropout1d,
        ).to(device)
        model.n_channels = n_channels
        print(f"[Model] model: {args.model} hidden_dim={hidden_dim} n_layers={n_layers} n_heads={n_heads} dropout={dropout}")
    else:
        hidden_dim = args.hidden_dim or 256
        dropout = args.dropout if args.dropout is not None else 0.2
        n_layers = None
        n_heads = None
        model = EEGClipEncoder(
            n_channels=n_channels,
            n_times=n_times,
            embedding_dim=dataset.embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            stem_dropout1d=args.stem_dropout1d,
        ).to(device)
        model.n_channels = n_channels
        print(f"[Model] model: cnn hidden_dim={hidden_dim} dropout={dropout}")
        
    if getattr(model, "n_channels", n_channels) != n_channels:
        raise ValueError(f"model.n_channels != dataset_eeg_channels")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Quick forward pass to verify shapes
    first_batch = next(iter(train_loader))
    batch_data = _batch_to_device(first_batch, device)
    with torch.inference_mode():
        pred = model(batch_data["eeg"])

    setup = {
        "items": len(dataset),
        "train_items": len(train_idx),
        "val_items": len(val_idx),
        "eeg_shape": list(dataset.eeg_shape),
        "embedding_dim": dataset.embedding_dim,
        "device": str(device),
        "first_forward_shape": list(pred.shape),
        "loss": args.loss,
        "temperature": args.temperature,
        "split_mode": args.split_mode,
        "val_runs": sorted(_parse_val_runs(args.val_runs)) if args.split_mode == "run" else None,
        "input_domain": args.input_domain,
        "target_mode": args.target_mode,
        "window_mode": args.window_mode,
        "target_space": args.target_space,
        "model": args.model,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "dropout": dropout,
        "stem_dropout1d": args.stem_dropout1d,
        "augment_eeg": args.augment_eeg,
        "augment_params": {
            "channel_dropout": args.aug_channel_dropout,
            "noise_std": args.aug_noise_std,
            "amp_scale": args.aug_amp_scale,
            "time_mask": args.aug_time_mask,
            "time_jitter": args.aug_time_jitter,
        },
        "target_bank_audit": target_bank_audit,
    }
    print(json.dumps({"setup": setup}, indent=2))

    if args.dry_run:
        return

    # Create structured run directory
    run_dir = _make_run_dir(args)
    print(f"Run directory: {run_dir}")
    _save_env(run_dir, args, setup)

    # Training loop
    best_score = float("-inf")
    best_epoch = None
    best_mrr = None
    best_top10 = None
    best_collapse_score = None
    history: list[dict] = []

    import csv
    log_path = run_dir / "train_log.csv"
    log_fields = ["epoch", "train_loss", "val_loss", "val_score",
                  "top1", "top5", "top10", "mrr", "median_rank", "mean_diag_cosine", "collapse_score"]

    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            batch_data = _batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_data["eeg"])
            
            # Primary contrastive loss
            loss = _loss_fn(pred, batch_data["target"], loss_name=args.loss, temperature=args.temperature)
            
            # Auxiliary multi-choice classification loss
            if "true_label" in batch_data and args.w_label > 0:
                import torch.nn.functional as F
                # pred: [B, D]
                # true_label: [B, D]
                # distractor_labels: [B, 15, D]
                
                # Normalize pred to get true cosine similarities
                pred_norm = F.normalize(pred, dim=-1)
                
                # Similarity to true label [B, 1]
                true_sim = (pred_norm * batch_data["true_label"]).sum(dim=-1, keepdim=True)
                # Similarity to distractors [B, 15]
                dist_sim = torch.bmm(batch_data["distractor_labels"], pred_norm.unsqueeze(2)).squeeze(2)
                
                # Combine [B, 16]
                logits = torch.cat([true_sim, dist_sim], dim=1)
                
                # True label is always at index 0
                label_idx = torch.zeros(pred.shape[0], dtype=torch.long, device=device)
                loss_ce = F.cross_entropy(logits / args.temperature, label_idx)
                
                loss = loss + args.w_label * loss_ce
                
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val = evaluate(model, val_loader, device, loss_name=args.loss,
                       temperature=args.temperature, target_space=args.target_space)
        score = float(val["mrr"] + 0.25 * val["top10"])
        if val["collapse_score"] < 0.1:
            score = -1.0
        row = {
            "epoch": epoch,
            "train_loss": float(sum(train_losses) / max(1, len(train_losses))),
            "val_loss": val["loss"],
            "val_score": score,
            "top1": val["top1"],
            "top5": val["top5"],
            "top10": val["top10"],
            "mrr": val["mrr"],
            "median_rank": val["median_rank"],
            "mean_diag_cosine": val["mean_diag_cosine"],
            "collapse_score": val["collapse_score"]
        }
        history.append(row)
        print(json.dumps(row))

        # Append to CSV log
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow(row)

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_mrr = float(val["mrr"])
            best_top10 = float(val["top10"])
            best_collapse_score = float(val["collapse_score"])
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "setup": setup,
                    "epoch": epoch,
                    "metrics": row,
                    "best_selection": {
                        "score": best_score,
                        "mrr": best_mrr,
                        "top10": best_top10,
                        "collapse_score": best_collapse_score,
                    },
                },
                run_dir / "best.pt",
            )

    # Save final artefacts
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))

    # Compute final val metrics on best checkpoint
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    final_metrics = evaluate(model, val_loader, device, loss_name=args.loss,
                             temperature=args.temperature, target_space=args.target_space)
    final_metrics["best_epoch"] = int(ckpt["epoch"])
    final_metrics["best_score"] = float(best_score)
    final_metrics["best_mrr"] = float(best_mrr) if best_mrr is not None else None
    final_metrics["best_top10"] = float(best_top10) if best_top10 is not None else None
    final_metrics["best_collapse_score"] = float(best_collapse_score) if best_collapse_score is not None else None
    final_metrics["target_bank_audit"] = target_bank_audit
    final_metrics.update(target_bank_audit)
    final_metrics["input_domain"] = args.input_domain
    final_metrics["target_mode"] = args.target_mode
    final_metrics["split_mode"] = args.split_mode
    final_metrics["condition"] = f"{args.input_domain}_{args.target_mode}"

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(final_metrics, indent=2))

    # Also write a single-row metrics.csv for aggregation
    import csv as _csv
    csv_path = run_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(final_metrics.keys()))
        w.writeheader()
        w.writerow(final_metrics)

    print(json.dumps({"best_score": best_score, "best_epoch": best_epoch, "run_dir": str(run_dir),
                      "metrics": final_metrics}, indent=2))


if __name__ == "__main__":
    import torch
    torch.backends.cudnn.enabled = False
    main()

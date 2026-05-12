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
        choices=("common", "semantic", "image"),
        default="common",
        help="Which embedding space to optimize the loss against",
    )
    p.add_argument("--output-dir", default=None,
                   help="Base output directory. Defaults to outputs/runs/")
    p.add_argument("--slug", default=None,
                   help="Optional slug appended to the run directory name")
    p.add_argument("--window-mode", choices=("crop", "full5s", "full5s_backaligned"), default="crop",
                   help="EEG window duration: crop (1.25s) or full5s (5s) or full5s_backaligned (5s)")
    p.add_argument("--add-event-marker", action="store_true",
                   help="Add event marker bump as an extra channel to EEG inputs")
    p.add_argument("--model", choices=("cnn", "temporal_attn"), default="cnn",
                   help="Encoder architecture: cnn (default) or temporal_attn")
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
    return p.parse_args()


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
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    eeg = batch["eeg"].to(device).float()
    tc = batch["target_common"].to(device).float()
    ti = batch["target_image"].to(device).float()
    ts = batch["target_semantic"].to(device).float()
    return eeg, tc, ti, ts


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
    preds, tc_list, ti_list, ts_list = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            eeg, tc, ti, ts = _batch_to_device(batch, device)
            pred = model(eeg)
            preds.append(pred.cpu())
            tc_list.append(tc.cpu())
            ti_list.append(ti.cpu())
            ts_list.append(ts.cpu())

    pred_t = torch.cat(preds)
    tc_t = torch.cat(tc_list)
    ti_t = torch.cat(ti_list)
    ts_t = torch.cat(ts_list)
    
    if target_space == "common":
        primary_target = tc_t
    elif target_space == "semantic":
        primary_target = ts_t
    else:
        primary_target = ti_t

    loss = float(_loss_fn(pred_t, primary_target, loss_name=loss_name, temperature=temperature).item())
    
    metrics = {"loss": loss, "n": int(pred_t.shape[0])}
    
    # evaluate for each space
    for name, t_t in [("common", tc_t), ("image", ti_t), ("semantic", ts_t)]:
        m = retrieval_topk(pred_t, t_t)
        for k, v in m.items():
            metrics[f"{name}_{k}"] = v
            
        metrics[f"{name}_mean_diag_cosine"] = float(
            (
                torch.nn.functional.normalize(pred_t, dim=-1)
                * torch.nn.functional.normalize(t_t, dim=-1)
            )
            .sum(dim=-1)
            .mean()
            .item()
        )
            
    # For backward compatibility with simple scripts parsing 'top10'
    m_primary = retrieval_topk(pred_t, primary_target)
    for k, v in m_primary.items():
        metrics[k] = v
    metrics["mean_diag_cosine"] = metrics[f"{target_space}_mean_diag_cosine"]

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

    dataset = ZunaClipPairDataset(
        SemanticPairConfig(
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
        )
    )

    if args.split_mode == "run":
        train_idx, val_idx = _split_by_run(dataset, _parse_val_runs(args.val_runs))
    else:
        train_idx, val_idx = split_indices(len(dataset), val_fraction=args.val_fraction, seed=args.seed)

    train_loader = DataLoader(
        Subset(dataset, train_idx),
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

    if args.model == "temporal_attn":
        from mindseye.models.eeg_encoder import TemporalAttnEncoder
        model = TemporalAttnEncoder(
            n_channels=n_channels, embedding_dim=dataset.embedding_dim
        ).to(device)
        model.n_channels = n_channels
        print(f"[Model] model: temporal_attn")
    else:
        model = EEGClipEncoder(
            n_channels=n_channels, n_times=n_times, embedding_dim=dataset.embedding_dim
        ).to(device)
        model.n_channels = n_channels
        print(f"[Model] model: cnn")
        
    if getattr(model, "n_channels", n_channels) != n_channels:
        raise ValueError(f"model.n_channels != dataset_eeg_channels")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Quick forward pass to verify shapes
    first_batch = next(iter(train_loader))
    eeg, tc, ti, ts = _batch_to_device(first_batch, device)
    with torch.inference_mode():
        pred = model(eeg)

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
    }
    print(json.dumps({"setup": setup}, indent=2))

    if args.dry_run:
        return

    # Create structured run directory
    run_dir = _make_run_dir(args)
    print(f"Run directory: {run_dir}")
    _save_env(run_dir, args, setup)

    # Training loop
    best_loss = float("inf")
    history: list[dict] = []

    import csv
    log_path = run_dir / "train_log.csv"
    log_fields = ["epoch", "train_loss", "val_loss", 
                  "val_common_top10", "val_common_mrr", "val_common_collapse_score",
                  "val_semantic_top10", "val_semantic_mrr", "val_semantic_collapse_score",
                  "val_image_top10", "val_image_mrr", "val_image_collapse_score",
                  "val_top10", "val_mrr"]

    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            eeg, tc, ti, ts = _batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(eeg)
            
            if args.target_space == "common":
                l_com = _loss_fn(pred, tc, loss_name=args.loss, temperature=args.temperature)
                l_sem = _loss_fn(pred, ts, loss_name=args.loss, temperature=args.temperature)
                l_img = _loss_fn(pred, ti, loss_name=args.loss, temperature=args.temperature)
                loss = l_com + 0.5 * l_sem + 0.1 * l_img
            elif args.target_space == "semantic":
                loss = _loss_fn(pred, ts, loss_name=args.loss, temperature=args.temperature)
            elif args.target_space == "image":
                loss = _loss_fn(pred, ti, loss_name=args.loss, temperature=args.temperature)
            else:
                loss = _loss_fn(pred, tc, loss_name=args.loss, temperature=args.temperature)
                
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val = evaluate(model, val_loader, device, loss_name=args.loss,
                       temperature=args.temperature, target_space=args.target_space)
        row = {
            "epoch": epoch,
            "train_loss": float(sum(train_losses) / max(1, len(train_losses))),
            "val_loss": val["loss"],
            
            "val_common_top10": val["common_top10"],
            "val_common_mrr": val["common_mrr"],
            "val_common_collapse_score": val["common_collapse_score"],
            
            "val_semantic_top10": val["semantic_top10"],
            "val_semantic_mrr": val["semantic_mrr"],
            "val_semantic_collapse_score": val["semantic_collapse_score"],
            
            "val_image_top10": val["image_top10"],
            "val_image_mrr": val["image_mrr"],
            "val_image_collapse_score": val["image_collapse_score"],
            
            "val_top10": val["top10"],
            "val_mrr": val["mrr"]
        }
        history.append(row)
        print(json.dumps(row))

        # Append to CSV log
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow(row)

        if val["loss"] < best_loss:
            best_loss = val["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "setup": setup,
                    "epoch": epoch,
                    "metrics": row,
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

    print(json.dumps({"best_loss": best_loss, "run_dir": str(run_dir),
                      "metrics": final_metrics}, indent=2))


if __name__ == "__main__":
    import torch
    torch.backends.cudnn.enabled = False
    main()

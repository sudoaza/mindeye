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
    # --target-space is kept as a hidden ablation flag; canonical path always uses 'common'.
    p.add_argument(
        "--target-space",
        choices=("common", "semantic", "image", "label"),
        default="common",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--common-probe",
        default=None,
        help="Path to pretrained common_probe.pt checkpoint",
    )
    p.add_argument(
        "--probe-weight",
        type=float,
        default=0.05,
        help="Weight for the auxiliary probe loss",
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
    p.add_argument("--no-spatial-mixing", action="store_true",
                   help="Disable early spatial mixing in spatial-temporal encoder")
    p.add_argument("--vlm-attributes", default=None,
                   help="Optional path to vlm_attributes.json for auxiliary multitask semantic training")
    p.add_argument("--patience", type=int, default=15,
                   help="Number of epochs without improvement before early stopping (autostop)")
    p.add_argument("--aux-start-epoch", type=int, default=1,
                   help="Epoch to start applying auxiliary multitask loss (for delayed starts)")
    p.add_argument("--aux-warmup-epochs", type=int, default=20,
                   help="Number of epochs to linearly warmup the auxiliary multitask weights (ramps from 0 to 1)")
    p.add_argument("--warmup-epochs", type=int, default=5,
                   help="Number of epochs to warmup learning rate")
    p.add_argument("--min-lr", type=float, default=1e-6,
                   help="Minimum learning rate for cosine annealing")
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
        "target": batch["target_common"].to(device).float() if "target_common" in batch else batch["target"].to(device).float()
    }
    if "probe_targets" in batch:
        ret["probe_targets"] = {k: v.to(device).long() for k, v in batch["probe_targets"].items()}
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

def evaluate(model, loader, device, *, loss_name: str, temperature: float, target_space: str = "common", probe_model=None, active_tasks=None, target_center=None) -> dict[str, float]:
    from mindseye.models.eeg_encoder import retrieval_topk
    import torch
    import torch.nn.functional as F
    model.eval()
    if probe_model:
        probe_model.eval()
    preds, targets = [], []
    probe_preds_dict = {task: [] for task in (active_tasks or [])}
    probe_targets_dict = {task: [] for task in (active_tasks or [])}
    
    with torch.inference_mode():
        for batch in loader:
            batch_data = _batch_to_device(batch, device)
            subject_id = batch_data.get("subject_id", None)
            kwargs = {"subject_id": subject_id} if "spatial_temporal" in type(model).__name__.lower() or "spatialtemporal" in type(model).__name__.lower() else {}
            pred = model(batch_data["eeg"], **kwargs)
            if probe_model and active_tasks and "probe_targets" in batch_data:
                # Probe was pretrained on F.normalize(z_common); normalize pred to match.
                a_preds = probe_model(F.normalize(pred, dim=-1))
                
            preds.append(pred.cpu())
            targets.append(batch_data["target"].cpu())
            
            if probe_model and active_tasks and "probe_targets" in batch_data:
                for task in active_tasks:
                    probe_preds_dict[task].append(a_preds[task].cpu())
                    probe_targets_dict[task].append(batch_data["probe_targets"][task].cpu())

    pred_t = torch.cat(preds)
    target_t = torch.cat(targets)
    
    if target_center is not None:
        tc = target_center.cpu()
        pred_t_eval = torch.nn.functional.normalize(pred_t - tc, dim=-1)
        target_t_eval = torch.nn.functional.normalize(target_t - tc, dim=-1)
    else:
        pred_t_eval = torch.nn.functional.normalize(pred_t, dim=-1)
        target_t_eval = torch.nn.functional.normalize(target_t, dim=-1)
        
    loss = float(_loss_fn(pred_t_eval, target_t_eval, loss_name=loss_name, temperature=temperature).item())
    
    metrics = {"loss": loss, "n": int(pred_t.shape[0])}
    
    # evaluate top-k
    m = retrieval_topk(pred_t_eval, target_t_eval)
    for k, v in m.items():
        metrics[k] = v
        
    metrics["mean_diag_cosine"] = float(
        (
            torch.nn.functional.normalize(pred_t_eval, dim=-1)
            * torch.nn.functional.normalize(target_t_eval, dim=-1)
        )
        .sum(dim=-1)
        .mean()
        .item()
    )

    # Expected random baselines
    n = metrics["n"]
    for k in (1, 5, 10):
        metrics[f"random_top{k}_expected"] = min(k, n) / n
        
    if probe_model and active_tasks and probe_preds_dict:
        from mindseye.models.common_probe import IGNORE_INDEX
        for task in active_tasks:
            ap = torch.cat(probe_preds_dict[task])
            at = torch.cat(probe_targets_dict[task])
            mask = at != IGNORE_INDEX
            if mask.any():
                acc = (ap.argmax(dim=-1)[mask] == at[mask]).float().mean().item()
                metrics[f"probe_{task}_acc"] = acc
            else:
                metrics[f"probe_{task}_acc"] = 0.0

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
    
    if args.vlm_attributes is None and args.common_probe:
        candidate = Path(args.common_probe).parent / "vlm_attributes_runs01_40.json"
        if candidate.exists():
            args.vlm_attributes = str(candidate)
            print(f"[Dataset] Auto-detected vlm_attributes at {args.vlm_attributes}")
            
    if args.target_space != "common":
        print("[WARN] Non-canonical target_space used for ablation only.")

    dataset_config = SemanticPairConfig(
            metadata_csv=args.metadata,
            epochs_dir=args.epochs_dir,
            epochs_dir_raw=args.epochs_dir_raw,
            epochs_dir_resample=args.epochs_dir_resample,
            common_embeddings_pt=args.common_embeddings,
            vlm_attributes_json=args.vlm_attributes,
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

    # Compute target center over the training split
    print("Computing mean target embedding vector over the training set...")
    with torch.inference_mode():
        all_train_targets = []
        for idx in train_idx:
            all_train_targets.append(dataset._get_targets(idx))
        target_center = torch.stack(all_train_targets).mean(dim=0).to(device)
        tc_norm = float(torch.linalg.norm(target_center).item())
        print(f"Target center vector computed. Norm: {tc_norm:.4f}")

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
        overrides["spatial_mixing"] = not args.no_spatial_mixing
        model = build_spatial_temporal_encoder(
            preset,
            n_channels=n_channels,
            embedding_dim=dataset.embedding_dim,
            ch_names=getattr(dataset, "ch_names", None),
            num_subjects=len(getattr(dataset, "unique_subjects", ["unknown"])),
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
    
    probe_model = None
    active_tasks = []
    if args.common_probe:
        from mindseye.models.common_probe import CommonProbeModel
        probe_specs_path = Path(args.common_probe).parent / "task_specs.json"
        if not probe_specs_path.exists():
            raise FileNotFoundError(f"Active tasks specification not found at {probe_specs_path}")
        with open(probe_specs_path, "r") as f:
            active_task_specs = json.load(f)
        
        probe_model = CommonProbeModel(
            embedding_dim=dataset.embedding_dim,
            task_specs=active_task_specs
        ).to(device)
        probe_model.load_state_dict(torch.load(args.common_probe, map_location=device))
        probe_model.eval()
        for p in probe_model.parameters():
            p.requires_grad = False
        active_tasks = list(active_task_specs.keys())
        print(f"[Model] Loaded frozen CommonProbeModel with {len(active_tasks)} active tasks from {args.common_probe}")
        
    opt_params = list(model.parameters())
        
    optimizer = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=args.weight_decay)

    # Quick forward pass to verify shapes
    first_batch = next(iter(train_loader))
    batch_data = _batch_to_device(first_batch, device)
    with torch.inference_mode():
        subject_id = batch_data.get("subject_id", None)
        kwargs = {"subject_id": subject_id} if "spatial_temporal" in type(model).__name__.lower() or "spatialtemporal" in type(model).__name__.lower() else {}
        pred = model(batch_data["eeg"], **kwargs)
        if probe_model is not None:
            _ = probe_model(pred)

    # Subject audit
    subjects_loaded = list(getattr(dataset, "unique_subjects", []))
    samples_per_subject = (
        dataset.metadata.groupby("subject").size().to_dict()
        if "subject" in dataset.metadata.columns else {}
    )
    # Infer requested subjects from comma-separated metadata paths
    subjects_requested = [
        Path(p.strip()).parent.name for p in str(args.metadata).split(",")
    ] if "," in str(args.metadata) else []
    subjects_skipped = [s for s in subjects_requested if not any(s in sl for sl in subjects_loaded)]

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
        "target_space": "common",
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
        "subjects_requested": subjects_requested,
        "subjects_loaded": subjects_loaded,
        "samples_per_subject": {str(k): int(v) for k, v in samples_per_subject.items()},
        "subjects_skipped": subjects_skipped,
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
    epochs_without_improvement = 0
    history: list[dict] = []

    import csv
    log_path = run_dir / "train_log.csv"
    log_fields = ["epoch", "train_loss", "val_loss", "val_score",
                  "top1", "top5", "top10", "mrr", "median_rank", "mean_diag_cosine", "collapse_score"]
                  
    if active_tasks:
        for task in active_tasks:
            log_fields.append(f"probe_{task}_acc")

    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    for epoch in range(1, args.epochs + 1):
        # Adjust learning rate with linear warmup + cosine annealing
        import math
        if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
            lr = args.lr * epoch / args.warmup_epochs
        else:
            total_decay_epochs = max(1, args.epochs - args.warmup_epochs)
            progress = (epoch - args.warmup_epochs - 1) / total_decay_epochs
            # Clamp progress between 0 and 1
            progress = max(0.0, min(1.0, progress))
            lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        print(f"\n--- Epoch {epoch}/{args.epochs} (Learning Rate: {lr:.6f}) ---")
        
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            batch_data = _batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            subject_id = batch_data.get("subject_id", None)
            kwargs = {"subject_id": subject_id} if "spatial_temporal" in type(model).__name__.lower() or "spatialtemporal" in type(model).__name__.lower() else {}
            pred = model(batch_data["eeg"], **kwargs)
            
            # Primary contrastive loss
            if target_center is not None:
                pred_for_loss = torch.nn.functional.normalize(pred - target_center, dim=-1)
                target_for_loss = torch.nn.functional.normalize(batch_data["target"] - target_center, dim=-1)
            else:
                pred_for_loss = pred
                target_for_loss = batch_data["target"]
                
            loss = _loss_fn(pred_for_loss, target_for_loss, loss_name=args.loss, temperature=args.temperature)
            
            # Auxiliary probe loss
            if probe_model is not None and "probe_targets" in batch_data:
                import torch.nn.functional as F
                from mindseye.models.common_probe import IGNORE_INDEX
                # Probe was pretrained on F.normalize(z_common); normalize pred to match.
                logits_dict = probe_model(F.normalize(pred, dim=-1))
                
                probe_loss = 0.0
                has_active = False
                for task in active_tasks:
                    task_targets = batch_data["probe_targets"][task]
                    if (task_targets != IGNORE_INDEX).any():
                        task_loss = F.cross_entropy(logits_dict[task], task_targets, ignore_index=IGNORE_INDEX)
                        probe_loss += task_loss
                        has_active = True
                
                if has_active:
                    loss = loss + args.probe_weight * probe_loss
                
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val = evaluate(model, val_loader, device, loss_name=args.loss,
                       temperature=args.temperature, target_space=args.target_space,
                       probe_model=probe_model, active_tasks=active_tasks,
                       target_center=target_center)
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
        for k, v in val.items():
            if k.startswith("probe_"):
                row[k] = v
                
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
            epochs_without_improvement = 0
            save_dict = {
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
                "target_center": target_center.cpu() if target_center is not None else None,
            }
            if probe_model is not None:
                save_dict["probe_model_state"] = probe_model.state_dict()
            torch.save(save_dict, run_dir / "best.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping triggered after {epoch} epochs (no improvement for {args.patience} epochs).")
                break

    # Save final artefacts
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))

    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if probe_model is not None and "probe_model_state" in ckpt:
        probe_model.load_state_dict(ckpt["probe_model_state"])
    target_center_eval = ckpt.get("target_center", None)
    if target_center_eval is not None:
        target_center_eval = target_center_eval.to(device)
    final_metrics = evaluate(model, val_loader, device, loss_name=args.loss,
                             temperature=args.temperature, target_space=args.target_space,
                             probe_model=probe_model, active_tasks=active_tasks,
                             target_center=target_center_eval)
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

#!/usr/bin/env python3
"""Train a baseline EEG→CLIP encoder on ZUNA semantic crops."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


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
        help="Directory containing per-run semantic NPZ files",
    )
    p.add_argument(
        "--clip-embeddings",
        default="data/processed/clip_embeddings/sub01_runs01_05_clip_vit_base_patch32.pt",
        help="CLIP embedding table from generate_clip_embeddings.py",
    )
    p.add_argument("--output-dir", default="outputs/eeg_clip_baseline")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--loss", choices=("contrastive", "cosine_mse"), default="contrastive")
    p.add_argument("--temperature", type=float, default=0.07, help="InfoNCE temperature for --loss contrastive")
    p.add_argument(
        "--center-clip",
        action="store_true",
        help="Subtract the train-set CLIP mean before target normalization/retrieval to reduce CLIP hubness",
    )
    p.add_argument(
        "--split-mode",
        choices=("random", "run"),
        default="random",
        help="Use a random item split or hold out full stimulus runs for validation",
    )
    p.add_argument("--val-runs", default="5", help="Comma-separated run numbers for --split-mode run")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--device", default=None, help="cuda, cpu, or omitted for auto")
    p.add_argument("--dry-run", action="store_true", help="Load data/model and run one forward pass only")
    return p.parse_args()


def _parse_val_runs(value: str) -> set[int]:
    return {int(x.strip()) for x in value.split(",") if x.strip()}


def _split_by_run(dataset: ZunaClipPairDataset, val_runs: set[int]) -> tuple[list[int], list[int]]:
    if "run" not in dataset.metadata.columns:
        raise ValueError("--split-mode run requires a 'run' column in metadata")
    train_idx: list[int] = []
    val_idx: list[int] = []
    for idx, run in enumerate(dataset.metadata["run"].astype(int).tolist()):
        (val_idx if run in val_runs else train_idx).append(idx)
    if not train_idx or not val_idx:
        raise ValueError(f"Invalid run split: train={len(train_idx)} val={len(val_idx)} for val_runs={sorted(val_runs)}")
    return train_idx, val_idx


def _target_center(dataset: ZunaClipPairDataset, indices: list[int], device: torch.device) -> torch.Tensor:
    clips = torch.stack([dataset[i]["clip"] for i in indices]).float().to(device)
    return clips.mean(dim=0, keepdim=True)


def _batch_to_device(
    batch: dict,
    device: torch.device,
    *,
    target_center: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    eeg = batch["eeg"].to(device).float()
    clip = batch["clip"].to(device).float()
    if target_center is not None:
        clip = clip - target_center
    return eeg, clip


def _loss_fn(pred: torch.Tensor, clip: torch.Tensor, *, loss_name: str, temperature: float) -> torch.Tensor:
    if loss_name == "contrastive":
        return clip_contrastive_loss(pred, clip, temperature=temperature)
    return cosine_mse_loss(pred, clip)


def evaluate(
    model: EEGClipEncoder,
    loader: DataLoader,
    device: torch.device,
    *,
    loss_name: str,
    temperature: float,
    target_center: torch.Tensor | None = None,
) -> dict[str, float]:
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in loader:
            eeg, clip = _batch_to_device(batch, device, target_center=target_center)
            pred = model(eeg)
            preds.append(pred.cpu())
            targets.append(clip.cpu())
    pred_t = torch.cat(preds)
    target_t = torch.cat(targets)
    metrics = retrieval_topk(pred_t, target_t, ks=(1, 5))
    metrics["loss"] = float(_loss_fn(pred_t, target_t, loss_name=loss_name, temperature=temperature).item())
    metrics["mean_diag_cosine"] = float((torch.nn.functional.normalize(pred_t, dim=-1) * torch.nn.functional.normalize(target_t, dim=-1)).sum(dim=-1).mean().item())
    metrics["pred_std"] = float(pred_t.std(dim=0).mean().item())
    metrics["n"] = int(pred_t.shape[0])
    return metrics


def main() -> None:
    args = parse_args()

    # Heavy ML/data imports are deferred so --help works in lightweight environments.
    global torch, DataLoader, Subset, SemanticPairConfig, ZunaClipPairDataset, split_indices
    global EEGClipEncoder, cosine_mse_loss, clip_contrastive_loss, retrieval_topk
    import torch
    from torch.utils.data import DataLoader, Subset

    from mindseye.datasets.semantic_pairs import SemanticPairConfig, ZunaClipPairDataset, split_indices
    from mindseye.models.eeg_encoder import EEGClipEncoder, clip_contrastive_loss, cosine_mse_loss, retrieval_topk

    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ZunaClipPairDataset(
        SemanticPairConfig(
            metadata_csv=args.metadata,
            epochs_dir=args.epochs_dir,
            clip_embeddings_pt=args.clip_embeddings,
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
    target_center = _target_center(dataset, train_idx, device) if args.center_clip else None

    n_channels, n_times = dataset.eeg_shape
    model = EEGClipEncoder(n_channels=n_channels, n_times=n_times, embedding_dim=dataset.embedding_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    first_batch = next(iter(train_loader))
    eeg, clip = _batch_to_device(first_batch, device, target_center=target_center)
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
        "center_clip": args.center_clip,
        "split_mode": args.split_mode,
        "val_runs": sorted(_parse_val_runs(args.val_runs)) if args.split_mode == "run" else None,
    }
    print(json.dumps({"setup": setup}, indent=2))
    if args.dry_run:
        return

    best_loss = float("inf")
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            eeg, clip = _batch_to_device(batch, device, target_center=target_center)
            optimizer.zero_grad(set_to_none=True)
            pred = model(eeg)
            loss = _loss_fn(pred, clip, loss_name=args.loss, temperature=args.temperature)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val = evaluate(
            model,
            val_loader,
            device,
            loss_name=args.loss,
            temperature=args.temperature,
            target_center=target_center,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(sum(train_losses) / max(1, len(train_losses))),
            "val_loss": val["loss"],
            "val_top1": val["top1"],
            "val_top5": val["top5"],
            "val_mean_diag_cosine": val["mean_diag_cosine"],
            "val_pred_std": val["pred_std"],
        }
        history.append(row)
        print(json.dumps(row))
        if val["loss"] < best_loss:
            best_loss = val["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "setup": setup,
                    "epoch": epoch,
                    "metrics": row,
                    "target_center": target_center.detach().cpu() if target_center is not None else None,
                },
                output_dir / "best.pt",
            )

    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(json.dumps({"best_loss": best_loss, "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()

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
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--device", default=None, help="cuda, cpu, or omitted for auto")
    p.add_argument("--dry-run", action="store_true", help="Load data/model and run one forward pass only")
    return p.parse_args()


def _batch_to_device(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return batch["eeg"].to(device).float(), batch["clip"].to(device).float()


def evaluate(model: EEGClipEncoder, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    losses: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            eeg, clip = _batch_to_device(batch, device)
            pred = model(eeg)
            loss = cosine_mse_loss(pred, clip)
            losses.append(float(loss.item()))
            preds.append(pred.cpu())
            targets.append(clip.cpu())
    pred_t = torch.cat(preds)
    target_t = torch.cat(targets)
    metrics = retrieval_topk(pred_t, target_t, ks=(1, 5))
    metrics["loss"] = float(sum(losses) / max(1, len(losses)))
    metrics["n"] = int(pred_t.shape[0])
    return metrics


def main() -> None:
    args = parse_args()

    # Heavy ML/data imports are deferred so --help works in lightweight environments.
    global torch, DataLoader, Subset, SemanticPairConfig, ZunaClipPairDataset, split_indices
    global EEGClipEncoder, cosine_mse_loss, retrieval_topk
    import torch
    from torch.utils.data import DataLoader, Subset

    from mindseye.datasets.semantic_pairs import SemanticPairConfig, ZunaClipPairDataset, split_indices
    from mindseye.models.eeg_encoder import EEGClipEncoder, cosine_mse_loss, retrieval_topk

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
    train_idx, val_idx = split_indices(len(dataset), val_fraction=args.val_fraction, seed=args.seed)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False)

    n_channels, n_times = dataset.eeg_shape
    model = EEGClipEncoder(n_channels=n_channels, n_times=n_times, embedding_dim=dataset.embedding_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    first_batch = next(iter(train_loader))
    eeg, clip = _batch_to_device(first_batch, device)
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
            eeg, clip = _batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(eeg)
            loss = cosine_mse_loss(pred, clip)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": float(sum(train_losses) / max(1, len(train_losses))),
            "val_loss": val["loss"],
            "val_top1": val["top1"],
            "val_top5": val["top5"],
        }
        history.append(row)
        print(json.dumps(row))
        if val["loss"] < best_loss:
            best_loss = val["loss"]
            torch.save({"model_state": model.state_dict(), "setup": setup, "epoch": epoch, "metrics": row}, output_dir / "best.pt")

    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(json.dumps({"best_loss": best_loss, "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()

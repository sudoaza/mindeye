#!/usr/bin/env python3
"""Train a RAE token bottleneck autoencoder.

Loads pre-extracted RAE tokens [768, 16, 16] from the RAE latent bank .pt file,
trains a chosen bottleneck architecture to compress→expand them, and optionally
runs an oracle image reconstruction check at the end.

Usage example (run on pod):

    python scripts/train_rae_token_bottleneck.py \\
        --rae-bank data/processed/rae_embeddings/rae_dinov2_base_sub01_04_runs01_40.pt \\
        --arch conv_256x4x4 \\
        --epochs 50 \\
        --batch-size 64 \\
        --lr 1e-3 \\
        --output-dir outputs/rae_bottleneck/conv_256x4x4 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindseye.models.rae_token_bottleneck import build_bottleneck, code_shape, CODE_SHAPES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rae-bank", required=True, help="Path to rae latent bank .pt file")
    p.add_argument(
        "--arch",
        default="conv_256x4x4",
        choices=list(CODE_SHAPES.keys()),
        help="Bottleneck architecture",
    )
    p.add_argument("--epochs", type=int, default=50, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    p.add_argument("--val-fraction", type=float, default=0.15, help="Validation split fraction")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--output-dir", default="outputs/rae_bottleneck/conv_256x4x4", help="Output directory")
    p.add_argument("--device", default=None, help="cuda, cpu, or auto")
    p.add_argument(
        "--oracle-check",
        action="store_true",
        default=True,
        help="Run oracle image reconstruction check after training (requires RAE backend)",
    )
    p.add_argument("--oracle-n", type=int, default=50, help="Number of val images to use in oracle check")
    p.add_argument(
        "--image-dir",
        default="data/raw/nod/stimuli/ImageNet",
        help="Image directory for oracle check (only needed with --oracle-check)",
    )
    p.add_argument("--no-oracle-check", dest="oracle_check", action="store_false")
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop if val_token_cosine has not improved for this many epochs (0 = disabled)",
    )
    return p.parse_args()


def load_rae_tokens(bank_path: Path, device: str = "cpu") -> tuple[list[str], torch.Tensor]:
    """Load image_id_to_rae_tokens from the RAE bank .pt file.

    Returns:
        image_ids: list of image IDs (sorted)
        tokens: FloatTensor [N, 768, 16, 16] on cpu
    """
    print(f"Loading RAE bank from {bank_path} ...")
    bank = torch.load(bank_path, map_location="cpu")
    tok_dict = bank.get("image_id_to_rae_tokens")
    if tok_dict is None:
        raise KeyError(f"'image_id_to_rae_tokens' not found in {bank_path}. Available: {list(bank.keys())}")
    image_ids = sorted(tok_dict.keys())
    tokens = torch.stack([tok_dict[img_id].float() for img_id in image_ids])  # [N, 768, 16, 16]
    print(f"Loaded {len(image_ids)} token maps, shape: {tokens.shape}")
    return image_ids, tokens


def build_loaders(tokens: torch.Tensor, val_fraction: float, batch_size: int, seed: int):
    """Split tokens into train/val and build DataLoaders.

    Returns:
        train_loader, val_loader, train_idx, val_idx
    """
    N = len(tokens)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g).tolist()
    n_val = max(1, int(round(N * val_fraction)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_ds = TensorDataset(tokens[train_idx])
    val_ds = TensorDataset(tokens[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)
    print(f"Split: {len(train_idx)} train, {len(val_idx)} val tokens")
    return train_loader, val_loader, train_idx, val_idx


# ---------------------------------------------------------------------------
# Oracle reconstruction check
# ---------------------------------------------------------------------------

def oracle_reconstruction_check(
    model: nn.Module,
    val_tokens: torch.Tensor,
    n: int,
    device: torch.device,
) -> dict:
    """Compress → expand → compute token-level reconstruction quality vs original.

    Measures:
    - Channel-vector cosine similarity at each spatial site (F.cosine_similarity dim=1)
    - Token-level MSE
    - Relative std ratio: per-channel code_std / tok_std (collapse detection)

    For the image oracle (compress→expand→decode image→re-encode), use a separate
    evaluate_rae_bottleneck.py script that loads the RAE decoder backend.

    Args:
        model: trained bottleneck model
        val_tokens: [M, 768, 16, 16] validation tokens
        n: number of samples to use
        device: torch device

    Returns:
        dict with mean_token_cosine, mean_token_mse, mean_std_ratio, pct_collapsed_channels
    """
    model.eval()
    indices = torch.randperm(len(val_tokens))[:n].tolist()
    sample = val_tokens[indices].to(device).float()

    with torch.no_grad():
        out = model(sample)

    expanded = out["expanded"]  # [n, 768, 16, 16]
    original = sample

    # Channel-vector cosine at each spatial site: F.cosine_similarity(dim=1) → [B, H, W]
    cos_spatial = F.cosine_similarity(expanded, original, dim=1)  # [n, 16, 16]
    mean_token_cosine = cos_spatial.mean().item()

    mean_token_mse = F.mse_loss(expanded, original).item()

    # Relative std collapse detection
    code = out["code"]
    B, Cc, Hc, Wc = code.shape
    code_flat = code.reshape(B, Cc, -1)                  # [B, Cc, Hc*Wc]
    code_std = code_flat.std(dim=[0, 2])                  # [Cc]

    tok_flat = original.reshape(B, original.shape[1], -1) # [B, 768, 256]
    tok_std = tok_flat.std(dim=[0, 2]).mean()              # scalar reference

    eps = 1e-6
    std_ratio = code_std / (tok_std + eps)                 # [Cc]  >1 ok, <0.2 = collapsed
    mean_std_ratio = std_ratio.mean().item()
    pct_collapsed = (std_ratio < 0.2).float().mean().item() * 100.0

    return {
        "mean_token_cosine": mean_token_cosine,
        "mean_token_mse": mean_token_mse,
        "mean_std_ratio": mean_std_ratio,
        "pct_collapsed_channels": pct_collapsed,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}  |  Arch: {args.arch}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    bank_path = Path(args.rae_bank)
    image_ids, all_tokens = load_rae_tokens(bank_path)

    train_loader, val_loader, train_idx, val_idx = build_loaders(
        all_tokens, args.val_fraction, args.batch_size, args.seed
    )

    # 2. Build model
    model = build_bottleneck(args.arch).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    has_params = n_params > 0
    if has_params:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)
    else:
        # Should not happen with updated architectures (spatial_768x4x4 has a learned expander now)
        print("[INFO] No learnable params found — skipping optimizer (eval only).")

    # 3. CSV log
    log_fields = ["epoch", "train_loss", "train_mse", "train_cos", "train_std", "val_loss", "val_mse", "val_cos", "val_std"]
    log_path = out_dir / "train_log.csv"
    log_file = open(log_path, "w", newline="")
    writer = csv.DictWriter(log_file, fieldnames=log_fields)
    writer.writeheader()

    best_val_cos = -1.0
    best_epoch = 0
    patience_counter = 0

    # 4. Train
    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        tr_loss = tr_mse = tr_cos = tr_std = 0.0
        n_tr = 0

        for (batch_tokens,) in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False):
            batch_tokens = batch_tokens.to(device)
            if has_params:
                optimizer.zero_grad()

            out = model(batch_tokens)
            loss = out["loss"]

            if has_params:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            B = batch_tokens.size(0)
            tr_loss += out["loss"].item() * B
            tr_mse += out["mse"].item() * B
            tr_cos += out["cos"].item() * B
            tr_std += out["std"].item() * B
            n_tr += B

        if has_params:
            scheduler.step()

        tr_loss /= n_tr
        tr_mse /= n_tr
        tr_cos /= n_tr
        tr_std /= n_tr

        # --- Val ---
        model.eval()
        vl_loss = vl_mse = vl_cos = vl_std = 0.0
        n_vl = 0

        with torch.no_grad():
            for (batch_tokens,) in val_loader:
                batch_tokens = batch_tokens.to(device)
                out = model(batch_tokens)
                B = batch_tokens.size(0)
                vl_loss += out["loss"].item() * B
                vl_mse += out["mse"].item() * B
                vl_cos += out["cos"].item() * B
                vl_std += out["std"].item() * B
                n_vl += B

        vl_loss /= n_vl
        vl_mse /= n_vl
        vl_cos /= n_vl
        vl_std /= n_vl

        # val_cos here is (1 - cosine) loss; token_cosine = 1 - val_cos
        val_token_cosine = 1.0 - vl_cos

        row = {
            "epoch": epoch,
            "train_loss": f"{tr_loss:.6f}",
            "train_mse": f"{tr_mse:.6f}",
            "train_cos": f"{tr_cos:.6f}",
            "train_std": f"{tr_std:.6f}",
            "val_loss": f"{vl_loss:.6f}",
            "val_mse": f"{vl_mse:.6f}",
            "val_cos": f"{vl_cos:.6f}",
            "val_std": f"{vl_std:.6f}",
        }
        writer.writerow(row)
        log_file.flush()

        if epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"train_loss={tr_loss:.4f} mse={tr_mse:.4f} cos={tr_cos:.4f} | "
                f"val_loss={vl_loss:.4f} mse={vl_mse:.4f} token_cosine={val_token_cosine:.4f}"
            )

        # Save best checkpoint (best val token cosine = lowest cos loss)
        improved = val_token_cosine > best_val_cos
        if improved and has_params:
            best_val_cos = val_token_cosine
            best_epoch = epoch
            patience_counter = 0
            torch.save({"arch": args.arch, "state_dict": model.state_dict()}, out_dir / "best.pt")
        elif args.early_stop_patience > 0:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.early_stop_patience} epochs)")
                break

    log_file.close()

    # Also save final checkpoint
    if has_params:
        torch.save({"arch": args.arch, "state_dict": model.state_dict()}, out_dir / "final.pt")
        print(f"Best val token cosine: {best_val_cos:.4f} at epoch {best_epoch}")
        # Reload best for oracle check
        ckpt = torch.load(out_dir / "best.pt", map_location=device)
        model.load_state_dict(ckpt["state_dict"])
    else:
        print("[INFO] No learnable params — spatial baseline oracle check only.")

    # 5. Oracle reconstruction check (token-level)
    print("\n--- Oracle Token Reconstruction Check ---")
    val_tokens_all = all_tokens[val_idx]
    oracle = oracle_reconstruction_check(model, val_tokens_all, n=args.oracle_n, device=device)
    print(f"  mean_token_cosine:      {oracle['mean_token_cosine']:.4f}  (aspirational target: > 0.90)")
    print(f"  mean_token_mse:         {oracle['mean_token_mse']:.6f}")
    print(f"  mean_std_ratio:         {oracle['mean_std_ratio']:.4f}  (code_std / tok_std; > 0.2 per channel = healthy)")
    print(f"  pct_collapsed_channels: {oracle['pct_collapsed_channels']:.2f}%  (aspirational target: < 5%)")

    # Aspirational gate evaluation (informational — not hard fail)
    gate_token_cos = oracle["mean_token_cosine"] > 0.90
    gate_collapse = oracle["pct_collapsed_channels"] < 5.0
    gate_pass = gate_token_cos and gate_collapse
    print(f"\n  [Aspirational] token_cosine > 0.90:   {'MET' if gate_token_cos else 'not yet met'}")
    print(f"  [Aspirational] collapsed < 5%:        {'MET' if gate_collapse else 'not yet met'}")
    print(f"  Both targets met:                     {'YES ✓' if gate_pass else 'NO — review metrics and decide'}")
    print(f"  NOTE: Thresholds are aspirational. Compare across architectures before deciding.")

    # 6. Save metrics
    code_sz = code_shape(args.arch)
    metrics = {
        "arch": args.arch,
        "code_shape": list(code_sz),
        "code_values": int(code_sz[0] * code_sz[1] * code_sz[2]),
        "n_params": n_params,
        "epochs": args.epochs,
        "best_epoch": best_epoch if has_params else 0,
        "best_val_token_cosine": best_val_cos if has_params else None,
        "oracle_token_cosine": oracle["mean_token_cosine"],
        "oracle_token_mse": oracle["mean_token_mse"],
        "oracle_mean_std_ratio": oracle["mean_std_ratio"],
        "oracle_pct_collapsed_channels": oracle["pct_collapsed_channels"],
        # Aspirational targets — compare across archs before deciding
        "aspirational_token_cosine_met": gate_token_cos,
        "aspirational_collapse_met": gate_collapse,
        "aspirational_both_met": gate_pass,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save arch config alongside checkpoint for loading without CLI flags
    config = {
        "arch": args.arch,
        "code_shape": list(code_sz),
        "rae_bank": str(bank_path),
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    train(parse_args())

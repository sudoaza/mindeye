#!/usr/bin/env python3
"""Compute per-site mean/std of RAE bottleneck codes on the training split.

Used by Phase 18E z-scored code prediction:
    pred_code = z_pred * code_std + code_mean
    target_z = (target_code - code_mean) / code_std

Usage:
    python scripts/build_rae_code_stats.py \\
        --codes-bank data/processed/rae_embeddings/rae_bottleneck_codes_4x4.pt \\
        --metadata data/processed/semantic_epochs/.../all_runs_metadata.csv \\
        --output data/processed/rae_embeddings/rae_code_stats_4x4.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindseye.datasets.semantic_pairs import SemanticPairConfig, ZunaClipPairDataset, split_indices


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--codes-bank", required=True, help="Bottleneck codes .pt (image_id_to_rae_code)")
    p.add_argument("--metadata", required=True, help="Metadata CSV (comma-separated for multi-subject)")
    p.add_argument("--epochs-dir", default=None, help="Epochs dir(s), comma-separated (for split alignment)")
    p.add_argument("--output", required=True, help="Output .pt path for code_mean / code_std")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--split-mode", choices=("random", "run"), default="random")
    p.add_argument("--val-runs", default="5", help="Held-out runs when --split-mode run")
    return p.parse_args()


def _parse_val_runs(value: str) -> set[int]:
    return {int(x.strip()) for x in value.split(",") if x.strip()}


def _split_by_run(dataset, val_runs: set[int]) -> tuple[list[int], list[int]]:
    train_idx, val_idx = [], []
    for idx, run in enumerate(dataset.metadata["run"].astype(int).tolist()):
        (val_idx if run in val_runs else train_idx).append(idx)
    return train_idx, val_idx


def main() -> None:
    args = parse_args()
    codes_data = torch.load(args.codes_bank, map_location="cpu")
    image_id_to_code = codes_data["image_id_to_rae_code"]
    code_shape = tuple(codes_data.get("code_shape", [768, 4, 4]))

    # Minimal dataset to reproduce train/val split indices
    epochs_dir = args.epochs_dir
    if epochs_dir is None:
        meta_paths = [p.strip() for p in args.metadata.split(",")]
        epochs_dir = ",".join(str(Path(p).parent) for p in meta_paths)

    ds_config = SemanticPairConfig(
        metadata_csv=args.metadata,
        epochs_dir=epochs_dir,
        common_embeddings_pt=args.codes_bank,
        target_space="rae_code",
        target_key="image_id_to_rae_code",
        window_mode="tight1s",
        augment_eeg=False,
    )
    dataset = ZunaClipPairDataset(ds_config)

    if args.split_mode == "run":
        train_idx, val_idx = _split_by_run(dataset, _parse_val_runs(args.val_runs))
    else:
        train_idx, _ = split_indices(len(dataset), val_fraction=args.val_fraction, seed=args.seed)

    train_codes = []
    for idx in train_idx:
        img_id = str(dataset.metadata.iloc[idx]["image_id"])
        if img_id not in image_id_to_code:
            continue
        code = image_id_to_code[img_id].float()
        if code.ndim == 1:
            code = code.reshape(*code_shape)
        train_codes.append(code)

    if not train_codes:
        raise RuntimeError("No training codes found — check metadata / codes-bank alignment")

    stacked = torch.stack(train_codes, dim=0)  # [N, C, H, W]
    code_mean = stacked.mean(dim=0)
    code_std = stacked.std(dim=0).clamp(min=1e-6)

    out = {
        "code_shape": list(code_shape),
        "code_mean": code_mean,
        "code_std": code_std,
        "n_train_images": int(stacked.shape[0]),
        "n_train_indices": len(train_idx),
        "codes_bank": str(args.codes_bank),
        "metadata": args.metadata,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "split_mode": args.split_mode,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(json.dumps({k: v for k, v in out.items() if k not in ("code_mean", "code_std")}, indent=2))
    print(f"code_mean: {code_mean.mean().item():.6f}  code_std (mean): {code_std.mean().item():.6f}")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()

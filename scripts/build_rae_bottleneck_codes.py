#!/usr/bin/env python3
"""Extract and save compressed RAE bottleneck codes for all images in the bank.

After training a bottleneck autoencoder with train_rae_token_bottleneck.py,
use this script to compress all image tokens and save the codes. The resulting
.pt file contains `image_id_to_rae_code` which can be used as EEG training targets.

Usage:

    python scripts/build_rae_bottleneck_codes.py \\
        --rae-bank data/processed/rae_embeddings/rae_dinov2_base_sub01_04_runs01_40.pt \\
        --checkpoint outputs/rae_bottleneck/conv_256x4x4/best.pt \\
        --output data/processed/rae_embeddings/rae_bottleneck_codes_conv256.pt \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindseye.models.rae_token_bottleneck import build_bottleneck


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rae-bank", required=True, help="Path to RAE latent bank .pt file")
    p.add_argument("--checkpoint", required=True, help="Path to trained bottleneck checkpoint (.pt)")
    p.add_argument(
        "--output",
        default="data/processed/rae_embeddings/rae_bottleneck_codes.pt",
        help="Output path for the code bank",
    )
    p.add_argument("--batch-size", type=int, default=128, help="Batch size for extraction")
    p.add_argument("--device", default=None, help="cuda, cpu, or auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    arch = ckpt["arch"]
    print(f"Loaded bottleneck checkpoint: arch={arch}")

    model = build_bottleneck(arch)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()

    # Load RAE bank
    bank_path = Path(args.rae_bank)
    print(f"Loading RAE bank from {bank_path} ...")
    bank = torch.load(bank_path, map_location="cpu")
    tok_dict = bank["image_id_to_rae_tokens"]
    image_ids = sorted(tok_dict.keys())
    print(f"Extracting codes for {len(image_ids)} images ...")

    image_id_to_rae_code = {}

    # Process in batches
    for i in tqdm(range(0, len(image_ids), args.batch_size), desc="Compressing"):
        batch_ids = image_ids[i : i + args.batch_size]
        batch_tokens = torch.stack([tok_dict[img_id].float() for img_id in batch_ids]).to(device)

        with torch.no_grad():
            codes = model.compress(batch_tokens)  # [B, C, 4, 4]

        for img_id, code in zip(batch_ids, codes):
            image_id_to_rae_code[img_id] = code.cpu().half()  # save as fp16 for storage efficiency

    # Determine code shape
    sample_shape = list(next(iter(image_id_to_rae_code.values())).shape)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_dict = {
        "arch": arch,
        "code_shape": sample_shape,
        "source_bank": str(bank_path),
        "checkpoint": str(args.checkpoint),
        "image_id_to_rae_code": image_id_to_rae_code,
    }
    torch.save(out_dict, out_path)

    print(f"Saved {len(image_id_to_rae_code)} codes (shape {sample_shape}) to {out_path}")


if __name__ == "__main__":
    main()

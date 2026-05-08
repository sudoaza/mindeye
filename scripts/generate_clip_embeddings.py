#!/usr/bin/env python3
"""Generate CLIP image embeddings for cropped NOD/ZUNA semantic epochs.

If stimulus images are not downloaded yet, first create an include list:
  PYTHONPATH=src python scripts/generate_clip_embeddings.py \
    --metadata data/processed/semantic_epochs/zuna_real_sub01_runs01_05/all_runs_metadata.csv \
    --write-openneuro-include-list data/processed/clip_embeddings/openneuro_image_includes.txt

Then download those OpenNeuro paths, and rerun without the include-list flag.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindseye.embeddings.clip import (
    ClipEmbeddingConfig,
    build_clip_embedding_table,
    missing_image_includes,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--metadata",
        default="data/processed/semantic_epochs/zuna_real_sub01_runs01_05/all_runs_metadata.csv",
        help="Crop metadata CSV containing class_id and image_id columns",
    )
    p.add_argument(
        "--stimuli-root",
        default="data/raw/nod/stimuli/ImageNet",
        help="Root containing NOD ImageNet stimulus images",
    )
    p.add_argument(
        "--output",
        default="data/processed/clip_embeddings/sub01_runs01_05_clip_vit_base_patch32.pt",
        help="Output .pt embedding table",
    )
    p.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None, help="cuda, cpu, or omitted for auto")
    p.add_argument(
        "--write-openneuro-include-list",
        default=None,
        help="Write targeted OpenNeuro image include paths here and exit",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(args.metadata)

    if args.write_openneuro_include_list:
        includes = missing_image_includes(metadata)
        out = Path(args.write_openneuro_include_list)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(includes) + "\n")
        print(json.dumps({"include_list": str(out), "paths": len(includes)}, indent=2))
        return

    result = build_clip_embedding_table(
        metadata_csv=args.metadata,
        stimuli_root=args.stimuli_root,
        output_pt=args.output,
        config=ClipEmbeddingConfig(
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=args.device,
        ),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

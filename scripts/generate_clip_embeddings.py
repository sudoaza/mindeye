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
import csv
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _read_metadata_rows(metadata_csv: str | Path) -> list[dict[str, str]]:
    """Read crop metadata with the standard library for lightweight include-list generation."""
    with Path(metadata_csv).open(newline="") as f:
        rows = list(csv.DictReader(f))
    missing = {"class_id", "image_id"} - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"Metadata missing required columns: {sorted(missing)}")
    return rows


def _normalize_image_id(image_id: str) -> str:
    stem = str(image_id)
    for suffix in (".JPEG", ".jpeg", ".JPG", ".jpg", ".PNG", ".png"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _missing_image_includes_from_rows(
    rows: list[dict[str, str]],
    stimuli_prefix: str = "stimuli/ImageNet",
    *,
    layout: str = "flat",
) -> list[str]:
    if layout not in {"flat", "synset", "both"}:
        raise ValueError(f"Unsupported layout {layout!r}; expected flat, synset, or both")
    includes: list[str] = []
    seen_pairs: set[tuple[str, str]] = set()
    seen_includes: set[str] = set()
    for row in rows:
        class_id = str(row["class_id"])
        stem = _normalize_image_id(str(row["image_id"]))
        pair = (class_id, stem)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        candidates: list[str] = []
        if layout in {"flat", "both"}:
            candidates.append(f"{stimuli_prefix}/{stem}.JPEG")
        if layout in {"synset", "both"}:
            candidates.append(f"{stimuli_prefix}/{class_id}/{stem}.JPEG")
        for inc in candidates:
            if inc not in seen_includes:
                seen_includes.add(inc)
                includes.append(inc)
    return includes


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
    p.add_argument(
        "--openneuro-layout",
        choices=("flat", "synset", "both"),
        default="flat",
        help="Stimulus layout to emit for include-list mode (default: flat for NOD ds005811)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.write_openneuro_include_list:
        rows = _read_metadata_rows(args.metadata)
        includes = _missing_image_includes_from_rows(rows, layout=args.openneuro_layout)
        out = Path(args.write_openneuro_include_list)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(includes) + "\n")
        print(json.dumps({"include_list": str(out), "paths": len(includes)}, indent=2))
        return

    # Heavy ML/data dependencies are imported only for actual embedding generation,
    # so --help and include-list prep work in lightweight environments.
    from mindseye.embeddings.clip import ClipEmbeddingConfig, build_clip_embedding_table

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

#!/usr/bin/env python3
"""Crop event-aligned semantic windows from real ZUNA outputs.

Example:
  PYTHONPATH=src python scripts/run_cropper.py \
    --zuna-dir data/processed/zuna_real_sub01_runs01_05/4_fif_output \
    --runs 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindseye.zuna.cropper import CropConfig, crop_zuna_runs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data/raw/nod", help="OpenNeuro NOD root")
    p.add_argument("--subject", default="sub-01")
    p.add_argument("--session", default="ImageNet01")
    p.add_argument("--runs", nargs="+", type=int, default=[1], help="Run numbers to crop")
    p.add_argument(
        "--zuna-dir",
        default="data/processed/zuna_real_sub01_runs01_05/4_fif_output",
        help="Directory containing real ZUNA output FIFs",
    )
    p.add_argument(
        "--output-dir",
        default="data/processed/semantic_epochs/zuna_real_sub01_runs01_05",
        help="Output directory for semantic epoch FIF/NPZ/metadata files",
    )
    p.add_argument("--tmin", type=float, default=-0.25)
    p.add_argument("--tmax", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    summary = crop_zuna_runs(
        raw_dir=data_root / "derivatives/preprocessed/raw",
        zuna_dir=Path(args.zuna_dir),
        events_csv=data_root / "derivatives/detailed_events" / f"{args.subject}_events.csv",
        output_dir=Path(args.output_dir),
        subject=args.subject,
        session=args.session,
        runs=args.runs,
        config=CropConfig(tmin=args.tmin, tmax=args.tmax),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

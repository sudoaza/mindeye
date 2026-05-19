#!/usr/bin/env python3
"""Crop event-aligned semantic windows from raw, resample-only, or ZUNA output FIFs.

Examples:
  # ZUNA crops (default)
  PYTHONPATH=src python scripts/run_cropper.py --mode zuna \
    --zuna-dir data/processed/zuna_real_sub01_runs01_05/4_fif_output \
    --runs 1 2 3 4 5

  # Raw crops (no ZUNA, crop straight from preprocessed FIF)
  PYTHONPATH=src python scripts/run_cropper.py --mode raw --runs 1 2 3 4 5

  # Resample-only crops (250Hz → 256Hz, no ZUNA denoising)
  PYTHONPATH=src python scripts/run_cropper.py --mode resample --runs 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindseye.zuna.cropper import CropConfig, crop_runs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="data/raw/nod", help="OpenNeuro NOD root")
    p.add_argument("--subject", default="sub-01")
    p.add_argument("--session", default="ImageNet01")
    p.add_argument("--runs", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                   help="Run numbers to crop")
    p.add_argument(
        "--mode", choices=("zuna", "raw", "resample"), default="zuna",
        help=(
            "zuna: crop from ZUNA-denoised FIF (needs --zuna-dir); "
            "raw: crop directly from preprocessed FIF at its native sfreq; "
            "resample: resample raw to --resample-sfreq then crop"
        ),
    )
    p.add_argument(
        "--zuna-dir",
        default="data/processed/zuna_output/4_fif_output",
        help="Directory of ZUNA output FIFs (only used when --mode zuna)",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Output directory for NPZ/FIF/metadata",
    )
    p.add_argument("--raw-dir", default=None, help="Override default raw directory (e.g. for simulated data)")
    p.add_argument("--tmin", type=float, default=-0.25)
    p.add_argument("--tmax", type=float, default=1.0)
    p.add_argument("--full5s", action="store_true",
                   help="Shortcut for --tmin -1.0 --tmax 4.0 (5s window) - Deprecated/Noisy")
    p.add_argument("--full5s-backaligned", action="store_true",
                   help="Shortcut for --tmin -3.0 --tmax 2.0 (5s window)")
    p.add_argument("--add-event-marker", action="store_true",
                   help="If set, records has_event_marker=True in metadata")
    p.add_argument("--expected-sfreq", type=float, default=256.0)
    p.add_argument("--resample-sfreq", type=float, default=256.0)
    p.add_argument("--event-name", default="stim_on")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)

    if args.full5s_backaligned:
        tmin, tmax = -3.0, 2.0
        window_mode = "full5s_backaligned"
        window_tag = "full5s_backaligned"
    elif args.full5s:
        tmin, tmax = -1.0, 4.0
        window_mode = "full5s"
        window_tag = "full5s"
    else:
        tmin, tmax = args.tmin, args.tmax
        window_mode = "crop"
        window_tag = f"crop_{tmin}_{tmax}"

    config = CropConfig(
        tmin=tmin,
        tmax=tmax,
        expected_sfreq=args.expected_sfreq,
        resample_sfreq=args.resample_sfreq,
        event_name=args.event_name,
        mode=args.mode,
        window_mode=window_mode,
        has_event_marker=args.add_event_marker,
    )

    run_str = "".join(f"{r:02d}" for r in sorted(args.runs))
    output_dir = Path(args.output_dir or f"data/processed/semantic_epochs/{args.mode}_{window_tag}_{args.subject}_runs{run_str}")

    source_dir = Path(args.zuna_dir) if args.mode == "zuna" else None

    print(f"Mode        : {args.mode}")
    print(f"Runs        : {args.runs}")
    print(f"Output dir  : {output_dir}")
    if source_dir:
        print(f"Source dir  : {source_dir}")

    summary = crop_runs(
        raw_dir=Path(args.raw_dir) if args.raw_dir else data_root / "derivatives/preprocessed/raw",
        source_dir=source_dir,
        events_csv=data_root / "derivatives/detailed_events" / f"{args.subject}_events.csv",
        output_dir=output_dir,
        subject=args.subject,
        session=args.session,
        runs=args.runs,
        config=config,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

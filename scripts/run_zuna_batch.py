#!/usr/bin/env python3
"""Run ZUNA on downloaded NOD-EEG continuous FIF runs.

Real ZUNA is the default. Use `--resample-only-baseline` only for local
plumbing checks when GPU/RAM are unavailable; baseline outputs are clearly
suffixed and should not be used for modeling results.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
import sys

import mne

mne.set_log_level("ERROR")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindseye.zuna.offline_pipeline import run_zuna_offline


def resample_only_baseline(input_fif_dir: str | Path, working_dir: str | Path) -> str:
    """
    Lightweight baseline for local pipeline checks.

    This does not run ZUNA. It only resamples inputs to 256Hz and writes them to
    `4_fif_output` with a `_resample_only.fif` suffix.
    """
    print("\n=== [BASELINE] Resample-only pipeline; not ZUNA denoising ===")
    input_fif_dir = Path(input_fif_dir)
    fif_output_dir = Path(working_dir) / "4_fif_output"
    fif_output_dir.mkdir(parents=True, exist_ok=True)

    fif_files = sorted(input_fif_dir.glob("*.fif"))
    if not fif_files:
        raise FileNotFoundError(f"No .fif files in {input_fif_dir}")

    for fif_path in fif_files:
        print(f"Resampling baseline for {fif_path.name}")
        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
        if raw.info["sfreq"] != 256:
            raw.resample(256.0)
        out_path = fif_output_dir / fif_path.name.replace(".fif", "_resample_only.fif")
        raw.save(out_path, overwrite=True, verbose=False)

    print(f"\n[BASELINE] Finished. Output in: {fif_output_dir}")
    return str(fif_output_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default="data/raw/nod/derivatives/preprocessed/raw")
    p.add_argument("--output-dir", default="data/processed/zuna_real")
    p.add_argument("--gpu-device", default="0", help="GPU id for ZUNA, or empty string for CPU")
    p.add_argument("--diffusion-steps", type=int, default=15)
    p.add_argument("--data-norm", type=float, default=10.0)
    p.add_argument(
        "--resample-only-baseline",
        action="store_true",
        help="Do not run ZUNA; only resample to 256Hz for plumbing checks",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.input_dir) or not glob.glob(os.path.join(args.input_dir, "*.fif")):
        print(f"Error: No continuous runs found in {args.input_dir}")
        sys.exit(1)

    print(f"Starting batch pipeline on {args.input_dir}")
    if args.resample_only_baseline:
        resample_only_baseline(args.input_dir, args.output_dir)
    else:
        run_zuna_offline(
            input_fif_dir=args.input_dir,
            working_dir=args.output_dir,
            target_channels=None,
            gpu_device=args.gpu_device,
            diffusion_steps=args.diffusion_steps,
            data_norm=args.data_norm,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run ZUNA on all unprocessed downloaded NOD-EEG continuous FIF runs in a single batch.

This avoids the torch.compile (inductor compile worker) startup overhead of running
each run in a separate subprocess.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import tempfile
from pathlib import Path
import sys
import mne

mne.set_log_level("ERROR")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default="data/raw/nod/derivatives/preprocessed/raw")
    p.add_argument("--output-dir", default="data/processed/zuna_real")
    p.add_argument("--gpu-device", default="0", help="GPU id for ZUNA, or empty string for CPU")
    p.add_argument("--diffusion-steps", type=int, default=15)
    p.add_argument("--data-norm", type=float, default=10.0)
    p.add_argument("--subject", default="sub-02", help="Only process files matching this subject prefix")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    
    fif_output_dir = os.path.join(args.output_dir, "4_fif_output")
    os.makedirs(fif_output_dir, exist_ok=True)

    # Find all source files for the subject
    all_source_files = sorted(glob.glob(os.path.join(args.input_dir, f"{args.subject}_*.fif")))
    if not all_source_files:
        print(f"Error: No continuous runs found in {args.input_dir} matching {args.subject}_*.fif")
        sys.exit(1)

    # Filter to only unprocessed files
    to_process = []
    for src_path in all_source_files:
        base_name = os.path.basename(src_path)
        dest_path = os.path.join(fif_output_dir, base_name)
        if os.path.exists(dest_path):
            print(f"Skipping {base_name} (already processed)")
        else:
            to_process.append(src_path)

    if not to_process:
        print("All files are already processed!")
        return

    print(f"\nFound {len(to_process)} unprocessed files to process in a single batch:")
    for p in to_process:
        print(f" - {os.path.basename(p)}")

    from zuna import preprocessing, inference, pt_to_fif

    with tempfile.TemporaryDirectory(dir=args.output_dir) as tmpdir:
        tmp_in = os.path.join(tmpdir, "in")
        tmp_1 = os.path.join(tmpdir, "1")
        tmp_2 = os.path.join(tmpdir, "2")
        tmp_3 = os.path.join(tmpdir, "3")
        tmp_4 = os.path.join(tmpdir, "4")
        for d in [tmp_in, tmp_1, tmp_2, tmp_3, tmp_4]:
            os.makedirs(d)

        # Copy all files to input directory
        print("\n=== Copying files to temp input directory ===")
        for p in to_process:
            shutil.copy(p, tmp_in)

        # Preprocessing
        preprocess_kwargs = dict(
            input_dir=tmp_in,
            output_dir=tmp_2,
            apply_notch_filter=False,
            apply_highpass_filter=True,
            apply_average_reference=True,
            preprocessed_fif_dir=tmp_1,
        )
        print("\n=== Preprocessing ===")
        preprocessing(**preprocess_kwargs)

        # Inference
        print("\n=== Inference (ZUNA) ===")
        inference(
            input_dir=tmp_2,
            output_dir=tmp_3,
            gpu_device=args.gpu_device,
            data_norm=args.data_norm,
            diffusion_sample_steps=args.diffusion_steps,
        )

        # Reconstruction
        print("\n=== Reconstruction ===")
        pt_to_fif(
            input_dir=tmp_3,
            output_dir=tmp_4,
        )
        
        # Copy result to final output dir
        print("\n=== Copying reconstructed files to output directory ===")
        out_files = glob.glob(os.path.join(tmp_4, "*.fif"))
        for out_f in out_files:
            shutil.copy(out_f, fif_output_dir)
            print(f"Saved: {os.path.basename(out_f)}")

    print("\nBatch process successfully finished!")


if __name__ == "__main__":
    main()

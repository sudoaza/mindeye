#!/usr/bin/env python3
"""Run the full 6-condition EEG→CLIP baseline matrix.

Each condition trains a fresh model via train_eeg_clip.py and saves its
metrics.json into a structured run directory.  At the end a summary CSV
and a console table are produced so you can gate Sprint 3.

Gate: zuna_real must beat zuna_shuffled, zuna_random, and zuna_sameclass on
top10 and MRR.  pred_std must be non-collapsed (collapse_score > 0.1).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Condition definitions
# ---------------------------------------------------------------------------

CONDITIONS = [
    # name               input_domain   target_mode  split
    ("raw_runheldout",         "raw",     "real",      "run"),
    ("resample_runheldout",    "resample","real",      "run"),
    ("zuna_runheldout",        "zuna",    "real",      "run"),
    ("zuna_shuffled_labels",   "zuna",    "shuffled",  "run"),
    ("zuna_random_targets",    "zuna",    "random",    "run"),
    ("zuna_sameclass",         "zuna",    "sameclass", "run"),
]

# ---------------------------------------------------------------------------
# Default paths — override via CLI
# ---------------------------------------------------------------------------

DEFAULTS = {
    "metadata": "data/processed/semantic_epochs/zuna_real_sub01_runs01_05/all_runs_metadata.csv",
    "epochs_dir": "data/processed/semantic_epochs/zuna_real_sub01_runs01_05",
    "clip_embeddings": "data/processed/clip_embeddings/sub01_runs01_05_clip_vit_base_patch32.pt",
    "val_runs": "5",
    "epochs": "30",
    "batch_size": "64",
    "device": "cuda",
}


def run_condition(
    name: str,
    input_domain: str,
    target_mode: str,
    split: str,
    matrix_dir: Path,
    args: argparse.Namespace,
) -> dict:
    """Launch train_eeg_clip.py for one condition and return its metrics dict."""
    print(f"\n{'='*60}")
    print(f"  Condition: {name}  ({input_domain} → {target_mode}, split={split})")
    print(f"{'='*60}")

    # Select metadata based on domain
    metadata_path = args.metadata
    if input_domain == "raw" and args.epochs_dir_raw:
        metadata_path = str(Path(args.epochs_dir_raw) / "all_runs_metadata.csv")
    elif input_domain == "resample" and args.epochs_dir_resample:
        metadata_path = str(Path(args.epochs_dir_resample) / "all_runs_metadata.csv")

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "train_eeg_clip.py"),
        "--metadata",       metadata_path,
        "--epochs-dir",     args.epochs_dir,
        "--clip-embeddings",args.clip_embeddings,
        "--input-domain",   input_domain,
        "--target-mode",    target_mode,
        "--split-mode",     split,
        "--val-runs",       args.val_runs,
        "--epochs",         str(args.epochs),
        "--batch-size",     str(args.batch_size),
        "--loss",           "contrastive",
        "--center-clip",
        "--output-dir",     str(matrix_dir),
        "--slug",           f"{name}_{args.slug}" if args.slug else name,
        "--window-mode",     args.window_mode,
        "--semantic-target", args.semantic_target,
        "--model",           args.model,
    ]
    if args.text_embeddings:
        cmd += ["--text-embeddings", args.text_embeddings]
    # Forward optional raw/resample dirs if provided
    if args.epochs_dir_raw:
        cmd += ["--epochs-dir-raw", args.epochs_dir_raw]
    if args.epochs_dir_resample:
        cmd += ["--epochs-dir-resample", args.epochs_dir_resample]

    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    result = subprocess.run(cmd, env=env, check=False, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout)

    if result.returncode != 0:
        print(f"[WARN] Condition {name} exited with code {result.returncode}")
        return {"condition": name, "status": "failed", "error": result.stdout[-500:]}

    # Locate the newest metrics.json written to matrix_dir
    candidates = sorted(matrix_dir.glob(f"*{name}*/metrics.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        print(f"[WARN] No metrics.json found for {name}")
        return {"condition": name, "status": "no_metrics"}

    metrics = json.loads(candidates[-1].read_text())
    metrics["condition"] = name
    metrics["status"] = "ok"
    return metrics


def gate_check(df: pd.DataFrame) -> None:
    """Print a pass/fail gate report for Sprint 2."""
    print("\n" + "=" * 60)
    print("  SPRINT 2 GATE CHECK")
    print("=" * 60)

    zuna = df[df["condition"] == "zuna_runheldout"]
    if zuna.empty or "top10" not in zuna.columns:
        print("[SKIP] zuna_runheldout metrics not found in results.")
        return

    zuna_top10 = float(zuna["top10"].iloc[0])
    zuna_mrr   = float(zuna["mrr"].iloc[0])
    zuna_cs    = float(zuna["collapse_score"].iloc[0])

    controls = [c for c in df["condition"].tolist() if c != "zuna_runheldout"]
    all_passed = True
    for ctrl in controls:
        row = df[df["condition"] == ctrl]
        if row.empty or pd.isna(row.get("top10", pd.Series([np.nan])).iloc[0]):
            print(f"  ❌  zuna_runheldout vs {ctrl}: {ctrl} missing metrics")
            all_passed = False
            continue
            
        ctrl_top10 = float(row["top10"].iloc[0])
        ctrl_mrr   = float(row["mrr"].iloc[0])
        passes = zuna_top10 > ctrl_top10 and zuna_mrr > ctrl_mrr
        symbol = "✅" if passes else "❌"
        all_passed = all_passed and passes
        print(f"  {symbol}  zuna_runheldout vs {ctrl}: "
              f"top10 {zuna_top10:.3f} vs {ctrl_top10:.3f}, "
              f"MRR {zuna_mrr:.3f} vs {ctrl_mrr:.3f}")

    cs_ok = zuna_cs > 0.1
    print(f"  {'✅' if cs_ok else '❌'}  collapse_score = {zuna_cs:.3f} (need > 0.1)")
    all_passed = all_passed and cs_ok

    print()
    print(f"  GATE: {'PASS — proceed to Sprint 3 ✅' if all_passed else 'FAIL — do not proceed ❌'}")
    print("=" * 60)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata",       default=DEFAULTS["metadata"])
    p.add_argument("--epochs-dir",     default=DEFAULTS["epochs_dir"])
    p.add_argument("--epochs-dir-raw", default=None,
                   help="NPZ dir for raw (un-denoised) crops.  Required for raw_runheldout.")
    p.add_argument("--epochs-dir-resample", default=None,
                   help="NPZ dir for resample-only crops.  Required for resample_runheldout.")
    p.add_argument("--clip-embeddings",default=DEFAULTS["clip_embeddings"])
    p.add_argument("--val-runs",       default=DEFAULTS["val_runs"])
    p.add_argument("--epochs",         type=int, default=int(DEFAULTS["epochs"]))
    p.add_argument("--batch-size",     type=int, default=int(DEFAULTS["batch_size"]))
    p.add_argument("--out-dir",        default="outputs/baseline_matrix",
                   help="Parent directory for all matrix runs")
    p.add_argument("--slug",           default=None)
    p.add_argument("--device",         default=DEFAULTS["device"])
    p.add_argument("--window-mode",     choices=("crop", "full5s"), default="crop")
    p.add_argument("--semantic-target", choices=("image", "text", "image_text"), default="image")
    p.add_argument("--text-embeddings", default=None)
    p.add_argument("--model",           choices=("cnn", "temporal_attn"), default="cnn")
    p.add_argument("--conditions",     nargs="*", default=None,
                   help="Subset of condition names to run (default: all 6)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned conditions and exit without training")
    args = p.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    matrix_dir = Path(args.out_dir) / f"{ts}_matrix"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    selected = set(args.conditions) if args.conditions else None
    active = [c for c in CONDITIONS if selected is None or c[0] in selected]

    if args.dry_run:
        print("Dry run — would execute:")
        for name, inp, tgt, split in active:
            print(f"  {name}: input={inp}, target={tgt}, split={split}")
        return

    all_metrics: list[dict] = []
    for name, input_domain, target_mode, split in active:
        # Skip raw/resample conditions if their dirs are not supplied
        if input_domain == "raw" and not args.epochs_dir_raw:
            print(f"[SKIP] {name}: --epochs-dir-raw not provided")
            all_metrics.append({"condition": name, "status": "skipped_no_raw_dir"})
            continue
        if input_domain == "resample" and not args.epochs_dir_resample:
            print(f"[SKIP] {name}: --epochs-dir-resample not provided")
            all_metrics.append({"condition": name, "status": "skipped_no_resample_dir"})
            continue

        m = run_condition(name, input_domain, target_mode, split, matrix_dir, args)
        all_metrics.append(m)

    # Aggregate
    df = pd.DataFrame(all_metrics)
    summary_csv = matrix_dir / "matrix_summary.csv"
    df.to_csv(summary_csv, index=False)
    print(f"\nMatrix summary written to: {summary_csv}")

    # Print a readable table of key metrics
    key_cols = [c for c in
                ["condition", "top1", "top5", "top10", "mrr", "median_rank",
                 "collapse_score", "off_diag_cosine", "pred_std", "status"]
                if c in df.columns]
    print(df[key_cols].to_markdown(index=False))

    gate_check(df)


if __name__ == "__main__":
    main()

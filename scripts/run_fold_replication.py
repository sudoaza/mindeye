#!/usr/bin/env python3
"""Run cross-fold replication sweep over val_runs and probe_weights.

Folds: val_runs = [8, 16, 24, 32, 40]
Probe weights: [0.00, 0.03, 0.05]

Aggregates metrics and prints a summary table to compare probe weights across folds.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--metadata",
        default="data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv",
        help="Metadata CSV path",
    )
    p.add_argument(
        "--epochs-dir",
        default="data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40",
        help="Epochs directory",
    )
    p.add_argument(
        "--common-embeddings",
        default="data/processed/clip_embeddings/common_embeddings.pt",
        help="CLIP common embeddings pt file",
    )
    p.add_argument(
        "--common-probe",
        default="outputs/common_probe/common_probe.pt",
        help="Path to pretrained common_probe.pt",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model", default="temporal_attn_small")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--out-dir", default="outputs/fold_replication", help="Base output directory")
    p.add_argument("--dry-run", action="store_true", help="Print planned runs and exit")
    return p.parse_args()

def main() -> None:
    args = parse_args()

    val_runs = [8, 16, 24, 32, 40]
    probe_weights = [0.00, 0.03, 0.05]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    replication_dir = Path(args.out_dir) / f"{ts}_replication"
    runs_dir = replication_dir / "runs"

    if not args.dry_run:
        runs_dir.mkdir(parents=True, exist_ok=True)

    planned_runs = []
    for val_run in val_runs:
        for pw in probe_weights:
            pw_str = f"{pw:.2f}".replace(".", "")
            planned_runs.append({
                "val_run": val_run,
                "probe_weight": pw,
                "pw_str": pw_str,
                "slug": f"val{val_run}_w{pw_str}"
            })

    print(f"Plan: {len(planned_runs)} runs to execute.")
    if args.dry_run:
        for r in planned_runs:
            print(f"  val_run={r['val_run']}, probe_weight={r['probe_weight']} (slug: {r['slug']})")
        return

    results = []

    for idx, r in enumerate(planned_runs, 1):
        print(f"\n[{idx}/{len(planned_runs)}] Starting: val_run={r['val_run']}, probe_weight={r['probe_weight']}")
        
        # Build command
        cmd = [
            "python", "-u", "scripts/train_eeg_clip.py",
            "--metadata", args.metadata,
            "--epochs-dir", args.epochs_dir,
            "--common-embeddings", args.common_embeddings,
            "--val-runs", str(r["val_run"]),
            "--window-mode", "tight1s",
            "--model", args.model,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--device", args.device,
            "--seed", str(args.seed),
            "--split-mode", "run",
            "--augment-eeg",
            "--add-event-marker",
            "--output-dir", str(runs_dir),
            "--slug", r["slug"],
        ]
        
        if r["probe_weight"] > 0:
            cmd.extend([
                "--common-probe", args.common_probe,
                "--probe-weight", str(r["probe_weight"])
            ])
            
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
        log_file = replication_dir / f"{r['slug']}.log"
        
        print(f"  Running: {' '.join(cmd)}")
        print(f"  Logging stdout/stderr to: {log_file}")
        
        with open(log_file, "w") as f:
            process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                # Write to log file
                f.write(line)
                f.flush()
                # Print status updates or errors to console
                if "Epoch" in line or "Error" in line or "Exception" in line or "early stopping" in line.lower() or "Auto-detected" in line:
                    sys.stdout.write("    " + line)
                    sys.stdout.flush()
            process.wait()
            
        if process.returncode != 0:
            print(f"❌ Run failed with exit code {process.returncode}. Check logs: {log_file}")
            results.append({
                "val_run": r["val_run"],
                "probe_weight": r["probe_weight"],
                "status": "failed",
                "top10": None,
                "mrr": None,
                "collapse_score": None
            })
            continue
            
        # Locate the run directory to load metrics
        matching_dirs = list(runs_dir.glob(f"*_{r['slug']}"))
        if not matching_dirs:
            print(f"⚠️ Could not find run directory matching *_{r['slug']}")
            results.append({
                "val_run": r["val_run"],
                "probe_weight": r["probe_weight"],
                "status": "missing_dir",
                "top10": None,
                "mrr": None,
                "collapse_score": None
            })
            continue
            
        # Sort by creation time to get the latest one just in case
        matching_dirs.sort(key=lambda d: d.stat().st_mtime)
        run_dir = matching_dirs[-1]
        metrics_file = run_dir / "metrics.json"
        
        if not metrics_file.exists():
            print(f"⚠️ Could not find metrics.json in {run_dir}")
            results.append({
                "val_run": r["val_run"],
                "probe_weight": r["probe_weight"],
                "status": "missing_metrics",
                "top10": None,
                "mrr": None,
                "collapse_score": None
            })
            continue
            
        with open(metrics_file, "r") as mf:
            metrics = json.load( mf)
            
        results.append({
            "val_run": r["val_run"],
            "probe_weight": r["probe_weight"],
            "status": "ok",
            "top10": metrics.get("top10"),
            "mrr": metrics.get("mrr"),
            "collapse_score": metrics.get("collapse_score"),
            "best_epoch": metrics.get("best_epoch"),
            "probe_is_animate_acc": metrics.get("probe_is_animate_acc")
        })
        
        print(f"✅ Success: Top-10: {metrics.get('top10'):.1%}, MRR: {metrics.get('mrr'):.1%}, collapse: {metrics.get('collapse_score'):.3f}")

    # Write summary
    df = pd.DataFrame(results)
    df.to_csv(replication_dir / "summary.csv", index=False)
    print(f"\nReplication summary written to: {replication_dir / 'summary.csv'}")

    # Aggregate by probe weight to show mean and std
    print("\n" + "=" * 60)
    print("  Aggregated Fold Replication Results (Mean ± Std)")
    print("=" * 60)
    
    ok_df = df[df["status"] == "ok"]
    if not ok_df.empty:
        summary_agg = ok_df.groupby("probe_weight").agg({
            "top10": ["mean", "std"],
            "mrr": ["mean", "std"],
            "collapse_score": ["mean", "std"]
        })
        print(summary_agg.to_markdown())
    else:
        print("No successful runs to aggregate.")

    print("\n" + "=" * 60)
    print("  Detailed Results Table")
    print("=" * 60)
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()

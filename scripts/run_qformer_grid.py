#!/usr/bin/env python3
import os
import sys
import subprocess
import json
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Ensure import paths work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

def compute_paired_bootstrap(metric_real, metric_control, n_bootstrap=10000, seed=42):
    diffs = metric_real - metric_control
    n = len(diffs)
    rng = np.random.default_rng(seed)
    # Vectorized bootstrap resamples
    boot_indices = rng.choice(n, size=(n_bootstrap, n), replace=True)
    boot_means = diffs[boot_indices].mean(axis=1)
    
    ci_lower = np.percentile(boot_means, 2.5)
    ci_upper = np.percentile(boot_means, 97.5)
    sign_rate = (boot_means > 0).mean()
    mean_diff = diffs.mean()
    return float(mean_diff), (float(ci_lower), float(ci_upper)), float(sign_rate)

def align_and_compute_deltas(real_data, control_data):
    # Align trials by sample_id to ensure exact pairing
    real_df = pd.DataFrame({
        "sample_id": real_data["sample_id"],
        "rank_real": real_data["rank"].numpy(),
        "top10_real": real_data["top10_hit"].numpy()
    })
    real_df["mrr_real"] = 1.0 / (real_df["rank_real"] + 1.0)
    
    control_df = pd.DataFrame({
        "sample_id": control_data["sample_id"],
        "rank_ctrl": control_data["rank"].numpy(),
        "top10_ctrl": control_data["top10_hit"].numpy()
    })
    control_df["mrr_ctrl"] = 1.0 / (control_df["rank_ctrl"] + 1.0)
    
    merged = pd.merge(real_df, control_df, on="sample_id", how="inner")
    if len(merged) == 0:
        raise ValueError("Could not align sample IDs between real and control runs!")
        
    diff_mrr = (merged["mrr_real"] - merged["mrr_ctrl"]).values
    diff_top10 = (merged["top10_real"] - merged["top10_ctrl"]).values
    return diff_mrr, diff_top10

def main():
    parser = argparse.ArgumentParser(description="Run minimal QFormer training grid with bootstrap evaluation.")
    parser.add_argument("--latents-pt", type=str, default="/workspace/mindeye/data/processed/zuna_latents/sub01_runs01_32")
    parser.add_argument("--clip-pt", type=str, default="/workspace/mindeye/data/processed/clip_embeddings/common_embeddings.pt")
    parser.add_argument("--rae-pt", type=str, default="/workspace/mindeye/data/processed/rae_embeddings/rae_dinov2_base_sub01_04_runs01_40.pt")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs per run")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device")
    parser.add_argument("--out-dir", type=str, default="outputs/qformer_aligned_grid", help="Parent outputs directory")
    args = parser.parse_args()
    
    # 12-run minimal grid configuration
    layers = ["post_mmd"]
    targets = [
        ("CLIP-Common-512", "CLIP-Common-512", args.clip_pt),
        ("DINO-Unit-768", "DINO-Unit-768", args.rae_pt),
        ("DINO-PCA-256-Unit", "DINO-PCA-256-Unit", args.rae_pt),
        ("DINO-PCA-128-Unit", "DINO-PCA-128-Unit", args.rae_pt)
    ]
    modes = ["real", "shuffled", "random"]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_dir = Path(args.out_dir) / f"grid_{timestamp}"
    grid_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting stabilized QFormer minimal grid training...")
    print(f"Grid Output Root: {grid_dir}\n")
    
    # Check if input latents exist
    if not os.path.exists(args.latents_pt):
        print(f"Error: Latents path {args.latents_pt} not found. Please wait for caching to complete.")
        sys.exit(1)
        
    runs_paths = {}
    runs_metrics = []
    
    for layer in layers:
        for target_name, target_space, target_path in targets:
            for mode in modes:
                if not os.path.exists(target_path):
                    print(f"Warning: Target path {target_path} not found. Skipping {target_name} ({mode}).")
                    continue
                    
                slug = f"grid_{layer}_{target_name}_{mode}"
                print(f"\n==================================================")
                print(f" Running: Layer={layer} | Target={target_name} | Mode={mode}")
                print(f"==================================================")
                
                cmd = [
                    sys.executable,
                    "scripts/train_zuna_to_vision.py",
                    "--latents-pt", args.latents_pt,
                    "--targets-pt", target_path,
                    "--target-space", target_space,
                    "--target-mode", mode,
                    "--layer-name", layer,
                    "--epochs", str(args.epochs),
                    "--patience", str(args.patience),
                    "--batch-size", str(args.batch_size),
                    "--lr", str(args.lr),
                    "--device", args.device,
                    "--out-dir", str(grid_dir),
                    "--slug", slug
                ]
                
                env = {**os.environ, "PYTHONPATH": "src"}
                
                process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                process.wait()
                
                if process.returncode != 0:
                    print(f"❌ Run failed with code {process.returncode}")
                    continue
                    
                # Find output directory
                run_dirs = list(grid_dir.glob(f"*{slug}*"))
                if not run_dirs:
                    print(f"Warning: Could not find output directory for {slug}")
                    continue
                newest_run = max(run_dirs, key=lambda d: d.stat().st_mtime)
                
                runs_paths[(target_name, mode)] = newest_run
                
                # Load metrics.json
                metrics_path = newest_run / "metrics.json"
                if metrics_path.exists():
                    with open(metrics_path, "r") as f:
                        metrics_data = json.load(f)
                    best_m = metrics_data["best"]
                    runs_metrics.append({
                        "Target": target_name,
                        "Mode": mode,
                        "MRR_Norm": best_m.get("val_mrr_norm", 0.0),
                        "Top1_Norm": best_m.get("val_top1_norm", 0.0),
                        "Top10_Norm": best_m.get("val_top10_norm", 0.0),
                        "Cosine_Norm": best_m.get("val_cosine_norm", 0.0),
                        "StdRatio": best_m.get("val_pred_std_ratio", 0.0),
                        "NormMean": best_m.get("val_pred_norm_mean", 0.0),
                        "NormStd": best_m.get("val_pred_norm_std", 0.0),
                        "CollapsePct": best_m.get("collapse_pct", 0.0),
                        "BestEpoch": best_m.get("epoch", 0)
                    })

    # 2. Paired Bootstrap Analysis
    bootstrap_results = []
    
    print("\n\n" + "=" * 60)
    print("  RUNNING PAIRED BOOTSTRAP ANALYSIS")
    print("=" * 60)
    
    for target_name, _, _ in targets:
        real_key = (target_name, "real")
        shuf_key = (target_name, "shuffled")
        rand_key = (target_name, "random")
        
        if real_key not in runs_paths:
            print(f"Skipping bootstrap for {target_name}: real run missing.")
            continue
            
        real_dir = runs_paths[real_key]
        real_data = torch.load(real_dir / "val_eval_preds.pt", map_location="cpu")
        
        # Shuffled comparison
        shuf_mrr_delta = shuf_mrr_ci = shuf_mrr_sig = "N/A"
        shuf_t10_delta = shuf_t10_ci = shuf_t10_sig = "N/A"
        if shuf_key in runs_paths:
            shuf_dir = runs_paths[shuf_key]
            shuf_data = torch.load(shuf_dir / "val_eval_preds.pt", map_location="cpu")
            try:
                diff_mrr, diff_t10 = align_and_compute_deltas(real_data, shuf_data)
                
                shuf_mrr_delta, mrr_ci, shuf_mrr_sig = compute_paired_bootstrap(diff_mrr, 0.0)
                shuf_mrr_ci = f"[{mrr_ci[0]:+.4f}, {mrr_ci[1]:+.4f}]"
                
                shuf_t10_delta, t10_ci, shuf_t10_sig = compute_paired_bootstrap(diff_t10, 0.0)
                shuf_t10_ci = f"[{t10_ci[0]:+.4f}, {t10_ci[1]:+.4f}]"
            except Exception as e:
                print(f"Error during shuffled bootstrap: {e}")
                
        # Random comparison
        rand_mrr_delta = rand_mrr_ci = rand_mrr_sig = "N/A"
        rand_t10_delta = rand_t10_ci = rand_t10_sig = "N/A"
        if rand_key in runs_paths:
            rand_dir = runs_paths[rand_key]
            rand_data = torch.load(rand_dir / "val_eval_preds.pt", map_location="cpu")
            try:
                diff_mrr, diff_t10 = align_and_compute_deltas(real_data, rand_data)
                
                rand_mrr_delta, mrr_ci, rand_mrr_sig = compute_paired_bootstrap(diff_mrr, 0.0)
                rand_mrr_ci = f"[{mrr_ci[0]:+.4f}, {mrr_ci[1]:+.4f}]"
                
                rand_t10_delta, t10_ci, rand_t10_sig = compute_paired_bootstrap(diff_t10, 0.0)
                rand_t10_ci = f"[{t10_ci[0]:+.4f}, {t10_ci[1]:+.4f}]"
            except Exception as e:
                print(f"Error during random bootstrap: {e}")
                
        # Evaluate Gate Criteria
        # Load best metrics for real run
        real_metrics = next((r for r in runs_metrics if r["Target"] == target_name and r["Mode"] == "real"), None)
        shuf_metrics = next((r for r in runs_metrics if r["Target"] == target_name and r["Mode"] == "shuffled"), None)
        rand_metrics = next((r for r in runs_metrics if r["Target"] == target_name and r["Mode"] == "random"), None)
        
        gate_passed = False
        gate_reasons = []
        
        if real_metrics and shuf_metrics and rand_metrics:
            # 1. Delta MRR > +0.005 for both controls
            shuf_mrr_val = float(shuf_mrr_delta) if isinstance(shuf_mrr_delta, float) else 0.0
            rand_mrr_val = float(rand_mrr_delta) if isinstance(rand_mrr_delta, float) else 0.0
            
            cond_mrr_delta = (shuf_mrr_val > 0.005) and (rand_mrr_val > 0.005)
            if not cond_mrr_delta:
                gate_reasons.append(f"Paired delta MRR below +0.005 (shuffled={shuf_mrr_val:+.4f}, random={rand_mrr_val:+.4f})")
                
            # 2. CI excludes 0 (meaning lower bound > 0 for both)
            try:
                shuf_mrr_lower = float(shuf_mrr_ci.strip("[]").split(",")[0])
                rand_mrr_lower = float(rand_mrr_ci.strip("[]").split(",")[0])
                cond_ci = (shuf_mrr_lower > 0.0) and (rand_mrr_lower > 0.0)
            except Exception:
                cond_ci = False
            if not cond_ci:
                gate_reasons.append(f"Confidence interval includes 0 (shuffled={shuf_mrr_ci}, random={rand_mrr_ci})")
                
            # 3. Real > Shuffled and Random on MRR and Top10
            cond_better = (real_metrics["MRR_Norm"] > shuf_metrics["MRR_Norm"]) and \
                          (real_metrics["MRR_Norm"] > rand_metrics["MRR_Norm"]) and \
                          (real_metrics["Top10_Norm"] > shuf_metrics["Top10_Norm"]) and \
                          (real_metrics["Top10_Norm"] > rand_metrics["Top10_Norm"])
            if not cond_better:
                gate_reasons.append("Real performance not superior to both controls on MRR and Top10")
                
            # 4. StdRatio between 0.3 and 2.0
            cond_ratio = 0.3 <= real_metrics["StdRatio"] <= 2.0
            if not cond_ratio:
                gate_reasons.append(f"StdRatio out of bounds [0.3, 2.0] (value={real_metrics['StdRatio']:.3f})")
                
            # 5. collapse_pct < 20.0
            cond_collapse = real_metrics["CollapsePct"] < 20.0
            if not cond_collapse:
                gate_reasons.append(f"Collapse percentage too high (value={real_metrics['CollapsePct']:.1f}%)")
                
            gate_passed = cond_mrr_delta and cond_ci and cond_better and cond_ratio and cond_collapse
            
        bootstrap_results.append({
            "Target": target_name,
            "Shuf MRR Δ": f"{shuf_mrr_delta:.4f}" if isinstance(shuf_mrr_delta, float) else "N/A",
            "Shuf MRR 95% CI": shuf_mrr_ci,
            "Shuf Sign Rate": f"{shuf_mrr_sig:.1%}" if isinstance(shuf_mrr_sig, float) else "N/A",
            "Rand MRR Δ": f"{rand_mrr_delta:.4f}" if isinstance(rand_mrr_delta, float) else "N/A",
            "Rand MRR 95% CI": rand_mrr_ci,
            "Rand Sign Rate": f"{rand_mrr_sig:.1%}" if isinstance(rand_mrr_sig, float) else "N/A",
            "Gate Status": "PASS ✅" if gate_passed else "FAIL ❌",
            "Fail Reasons": "; ".join(gate_reasons) if not gate_passed else "None"
        })
        
    # Write summary reports
    if runs_metrics:
        df_runs = pd.DataFrame(runs_metrics)
        df_boot = pd.DataFrame(bootstrap_results)
        
        df_runs.to_csv(grid_dir / "runs_summary.csv", index=False)
        df_boot.to_csv(grid_dir / "bootstrap_summary.csv", index=False)
        
        report_md = f"""# ZUNA-to-Vision Stabilized Grid Rerun Report
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Experiment Root: `{grid_dir}`

## 1. Grid Performance Metrics (Normalized)
{df_runs.to_markdown(index=False)}

## 2. Paired Bootstrap Analysis & Gates
{df_boot.to_markdown(index=False)}

## Gate Criteria Reminder
- **Normalized paired Δ MRR**: > +0.005
- **Confidence Interval**: Excludes 0
- **Comparisons**: Real > Shuffled and Random on MRR and Top10
- **StdRatio**: Between 0.3 and 2.0
- **Dimension Collapse**: < 20% collapse percentage
"""
        with open(grid_dir / "README.md", "w") as f:
            f.write(report_md)
            
        print("\n\n" + "=" * 60)
        print("  GRID RUNS SUMMARY")
        print("=" * 60)
        print(df_runs.to_markdown(index=False))
        print("\n\n" + "=" * 60)
        print("  BOOTSTRAP ANALYSIS & GATES")
        print("=" * 60)
        print(df_boot.to_markdown(index=False))
        print("=" * 60)
        print(f"Summary report written to {grid_dir / 'README.md'}")
    else:
        print("No runs completed successfully.")

if __name__ == "__main__":
    main()

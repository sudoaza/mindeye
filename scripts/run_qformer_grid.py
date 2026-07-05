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
    # Prefer the honest full-bank rank/hit when available (finding H1); fall back
    # to within-val rank for older eval files.
    def _cols(data):
        if "rank_full" in data and "top10_hit_full" in data:
            return data["rank_full"].numpy(), data["top10_hit_full"].numpy(), "full"
        return data["rank"].numpy(), data["top10_hit"].numpy(), "within_val"

    real_rank, real_top10, real_scope = _cols(real_data)
    ctrl_rank, ctrl_top10, ctrl_scope = _cols(control_data)
    scope = real_scope if real_scope == ctrl_scope else "mixed"

    # Align trials by sample_id to ensure exact pairing
    real_df = pd.DataFrame({
        "sample_id": real_data["sample_id"],
        "rank_real": real_rank,
        "top10_real": real_top10,
    })
    real_df["mrr_real"] = 1.0 / (real_df["rank_real"] + 1.0)

    control_df = pd.DataFrame({
        "sample_id": control_data["sample_id"],
        "rank_ctrl": ctrl_rank,
        "top10_ctrl": ctrl_top10,
    })
    control_df["mrr_ctrl"] = 1.0 / (control_df["rank_ctrl"] + 1.0)

    merged = pd.merge(real_df, control_df, on="sample_id", how="inner")
    if len(merged) == 0:
        raise ValueError("Could not align sample IDs between real and control runs!")

    diff_mrr = (merged["mrr_real"] - merged["mrr_ctrl"]).values
    diff_top10 = (merged["top10_real"] - merged["top10_ctrl"]).values
    # Paired rank delta: positive means the real model ranks the true image BETTER
    # (lower rank) than the control. This is the metric that surfaces a diffuse
    # median-rank win that the MRR/Top10 gate (dominated by the very top ranks)
    # cannot see for an EEG signal that shifts the whole distribution left.
    diff_rank = (merged["rank_ctrl"] - merged["rank_real"]).values
    median_rank_real = float(np.median(merged["rank_real"].values)) + 1.0
    median_rank_ctrl = float(np.median(merged["rank_ctrl"].values)) + 1.0
    return {
        "diff_mrr": diff_mrr,
        "diff_top10": diff_top10,
        "diff_rank": diff_rank,
        "median_rank_real": median_rank_real,
        "median_rank_ctrl": median_rank_ctrl,
        "scope": scope,
    }

def main():
    parser = argparse.ArgumentParser(description="Run minimal QFormer training grid with bootstrap evaluation.")
    parser.add_argument("--latents-pt", type=str, default="data/processed/zuna_latents/cohort")
    parser.add_argument("--rae-pt", type=str, default="data/processed/rae_embeddings/rae_dinov2_base_bank.pt")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs per run")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device")
    parser.add_argument("--out-dir", type=str, default="outputs/qformer_aligned_grid", help="Parent outputs directory")
    # Explicit run splits (overrides smoke-test defaults)
    parser.add_argument("--train-runs", type=str, default=None, help="Train run range, e.g. '1-24'")
    parser.add_argument("--val-runs", type=str, default=None, help="Val run range, e.g. '25-28'")
    parser.add_argument("--test-runs", type=str, default=None, help="Test run range, e.g. '29-32'")
    # Smoke-test mode: RAE-only (DINO-Unit-768), 8 runs, fixed splits, fast epochs
    parser.add_argument("--smoke-test", action="store_true", default=False,
                        help="Smoke test: RAE DINO-Unit-768 only, 8 runs (1-6 train, 7-8 val), 25 epochs, batch 32")
    parser.add_argument("--target-spaces", type=str, default=None,
                        help="Comma-separated target spaces to run (e.g. 'DINO-Unit-768,DINO-CLS-768'). "
                             "Overrides the default 3-target list; each is loaded from --rae-pt.")
    # Temporal windowing
    parser.add_argument("--temporal-window", action="store_true", default=True,
                        help="Enable temporal windowing in train script (default: on)")
    parser.add_argument("--no-temporal-window", action="store_false", dest="temporal_window")
    parser.add_argument("--latent-tc-start", type=int, default=20, help="Latent time slice start index")
    parser.add_argument("--latent-tc-end", type=int, default=36, help="Latent time slice end index (exclusive)")
    parser.add_argument("--num-subjects", type=int, default=1, help="Number of subjects in the cohort (enables subject FiLM in the QFormer when > 1)")
    # Reconstruction / luminance grounding (opt-in). When > 0, each run additionally
    # predicts an RAE token grid, decodes it through the frozen RAE, and adds a
    # stimulus-vs-generated luminance-grid loss. Overrides HANDOVER non-negotiable #3
    # by explicit decision — see docs/HANDOVER.md.
    parser.add_argument("--recon-luma-weight", type=float, default=0.0,
                        help="Weight of stimulus-vs-generated luminance-grid loss (0 = pure retrieval, unchanged behaviour)")
    parser.add_argument("--stimuli-dir", type=str, default="data/raw/nod/stimuli/ImageNet",
                        help="Stimulus image dir for the luminance loss (used only when --recon-luma-weight > 0)")
    # Recover a completed-but-crashed grid: skip all training and rebuild the
    # summary + paired-bootstrap gate from an existing grid dir's run outputs.
    parser.add_argument("--analyze-only", type=str, default=None,
                        help="Path to an existing grid_<ts> dir; skip training and (re)run analysis over its run dirs.")
    args = parser.parse_args()
    
    # --- Smoke-test overrides ---
    if args.smoke_test:
        print("=== SMOKE TEST MODE: RAE DINO-Unit-768 only, runs 1-6 train / 7-8 val, 25 epochs ===")
        epochs = args.epochs if args.epochs != 40 else 25
        batch_size = args.batch_size if args.batch_size != 64 else 32
        train_runs = args.train_runs or "1-6"
        val_runs   = args.val_runs   or "7-8"
        test_runs  = args.test_runs  or ""
        targets = [("DINO-Unit-768", "DINO-Unit-768", args.rae_pt)]
    else:
        epochs = args.epochs
        batch_size = args.batch_size
        train_runs = args.train_runs
        val_runs   = args.val_runs
        test_runs  = args.test_runs
        if args.target_spaces:
            names = [s.strip() for s in args.target_spaces.split(",") if s.strip()]
            targets = [(n, n, args.rae_pt) for n in names]
            print(f"[targets] using custom target spaces: {names}")
        else:
            targets = [
                ("DINO-Unit-768", "DINO-Unit-768", args.rae_pt),
                ("DINO-PCA-256-Unit", "DINO-PCA-256-Unit", args.rae_pt),
                ("DINO-PCA-128-Unit", "DINO-PCA-128-Unit", args.rae_pt)
            ]

    # 12-run minimal grid configuration
    layers = ["post_mmd"]
    modes = ["real", "shuffled", "random"]

    runs_paths = {}
    runs_metrics = []

    def _load_run_metrics(target_name, mode, run_dir):
        """Populate runs_metrics from a completed run dir's metrics.json (DRY: same
        shape as the training-loop path below)."""
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            print(f"Warning: {metrics_path} missing; skipping {target_name} ({mode}).")
            return
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)
        best_m = metrics_data["best"]
        runs_metrics.append({
            "Target": target_name,
            "Mode": mode,
            "MRR_Norm": best_m.get("val_mrr_norm", 0.0),
            "Top1_Norm": best_m.get("val_top1_norm", 0.0),
            "Top10_Norm": best_m.get("val_top10_norm", 0.0),
            "MRR_Full": best_m.get("val_mrr_full"),
            "Top10_Full": best_m.get("val_top10_full"),
            "BankSize": best_m.get("bank_size"),
            "Cosine_Norm": best_m.get("val_cosine_norm", 0.0),
            "StdRatio": best_m.get("val_pred_std_ratio", 0.0),
            "NormMean": best_m.get("val_pred_norm_mean", 0.0),
            "NormStd": best_m.get("val_pred_norm_std", 0.0),
            "CollapsePct": best_m.get("collapse_pct", 0.0),
            "BestEpoch": best_m.get("epoch", 0),
        })

    if args.analyze_only:
        # Recovery path: rebuild runs_paths/runs_metrics from an existing grid dir,
        # then fall through to the shared bootstrap-analysis + reporting block.
        grid_dir = Path(args.analyze_only)
        if not grid_dir.is_dir():
            print(f"Error: --analyze-only dir {grid_dir} not found.")
            sys.exit(1)
        print(f"=== ANALYZE-ONLY: rebuilding gate from existing runs in {grid_dir} ===")
        for layer in layers:
            for target_name, _, _ in targets:
                for mode in modes:
                    slug = f"grid_{layer}_{target_name}_{mode}"
                    run_dirs = [d for d in grid_dir.glob(f"*{slug}*") if d.is_dir()]
                    if not run_dirs:
                        print(f"Warning: no run dir for {slug}.")
                        continue
                    newest_run = max(run_dirs, key=lambda d: d.stat().st_mtime)
                    runs_paths[(target_name, mode)] = newest_run
                    _load_run_metrics(target_name, mode, newest_run)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        grid_dir = Path(args.out_dir) / f"grid_{timestamp}"
        grid_dir.mkdir(parents=True, exist_ok=True)

        print(f"Starting stabilized QFormer minimal grid training...")
        print(f"Grid Output Root: {grid_dir}\n")

        # Check if input latents exist
        if not os.path.exists(args.latents_pt):
            print(f"Error: Latents path {args.latents_pt} not found. Please wait for caching to complete.")
            sys.exit(1)

        for layer in layers:
            for target_name, target_space, target_path in targets:
                processes = []
                for mode in modes:
                    if not os.path.exists(target_path):
                        print(f"Warning: Target path {target_path} not found. Skipping {target_name} ({mode}).")
                        continue

                    slug = f"grid_{layer}_{target_name}_{mode}"
                    print(f"\n==================================================")
                    print(f" Preparing: Layer={layer} | Target={target_name} | Mode={mode}")
                    print(f"==================================================")

                    # Control runs (shuffled/random) only establish the chance
                    # baseline and plateau within a few epochs, so cap them at <=8
                    # epochs to avoid wasting compute. Real runs the full schedule.
                    mode_epochs = epochs if mode == "real" else min(epochs, 8)
                    if mode_epochs != epochs:
                        print(f"[epochs] {mode} control capped at {mode_epochs} epochs (real uses {epochs})")

                    cmd = [
                        sys.executable,
                        "scripts/train_zuna_to_vision.py",
                        "--latents-pt", args.latents_pt,
                        "--targets-pt", target_path,
                        "--target-space", target_space,
                        "--target-mode", mode,
                        "--layer-name", layer,
                        "--epochs", str(mode_epochs),
                        "--patience", str(args.patience),
                        "--batch-size", str(batch_size),
                        "--lr", str(args.lr),
                        "--device", args.device,
                        "--out-dir", str(grid_dir),
                        "--slug", slug,
                        "--num-subjects", str(args.num_subjects),
                    ]
                    if args.recon_luma_weight > 0.0:
                        cmd += [
                            "--recon-luma-weight", str(args.recon_luma_weight),
                            "--stimuli-dir", args.stimuli_dir,
                        ]
                    # Pass temporal window flag
                    if args.temporal_window:
                        cmd += [
                            "--temporal-window",
                            "--latent-tc-start", str(args.latent_tc_start),
                            "--latent-tc-end", str(args.latent_tc_end),
                        ]
                    else:
                        cmd.append("--no-temporal-window")
                    # Pass run splits if specified
                    if train_runs:
                        cmd += ["--train-runs", train_runs]
                    if val_runs:
                        cmd += ["--val-runs", val_runs]
                    if test_runs:
                        cmd += ["--test-runs", test_runs]

                    env = {**os.environ, "PYTHONPATH": "src"}

                    print(f"Launching subprocess: Layer={layer} | Target={target_name} | Mode={mode}")
                    # Finding L1: stream child output to a log file rather than a PIPE, so a
                    # child that fills the 64KB stdout pipe buffer can't deadlock before we
                    # call communicate() on the others.
                    log_path = grid_dir / f"{slug}.log"
                    log_fh = open(log_path, "w")
                    process = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT, text=True)
                    processes.append((mode, slug, process, log_fh, log_path))

                # Wait for all modes in this target group to complete
                for mode, slug, process, log_fh, log_path in processes:
                    print(f"\nWaiting for {target_name} ({mode}) to finish...")
                    process.wait()
                    log_fh.close()
                    try:
                        stdout = log_path.read_text()
                    except OSError:
                        stdout = ""
                    print(f"=== Subprocess Output for {target_name} ({mode}) (log: {log_path}) ===")
                    print(stdout)

                    if process.returncode != 0:
                        print(f"❌ {target_name} ({mode}) failed with code {process.returncode}")
                        continue

                    # Find output directory. Restrict to directories so the per-mode
                    # "<slug>.log" file (also matched by the glob) can never win the
                    # most-recent-mtime selection and get treated as a run dir.
                    run_dirs = [d for d in grid_dir.glob(f"*{slug}*") if d.is_dir()]
                    if not run_dirs:
                        print(f"Warning: Could not find output directory for {slug}")
                        continue
                    newest_run = max(run_dirs, key=lambda d: d.stat().st_mtime)

                    runs_paths[(target_name, mode)] = newest_run
                    _load_run_metrics(target_name, mode, newest_run)

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

        # Numeric results per control (finding M5: keep numbers, not formatted strings).
        # A missing control is recorded explicitly rather than silently coerced to 0.0.
        def _compare_control(control_key, control_name):
            if control_key not in runs_paths:
                return {"present": False, "reason": f"{control_name} run missing"}
            control_data = torch.load(runs_paths[control_key] / "val_eval_preds.pt", map_location="cpu")
            deltas = align_and_compute_deltas(real_data, control_data)
            mrr_delta, mrr_ci, mrr_sig = compute_paired_bootstrap(deltas["diff_mrr"], 0.0)
            t10_delta, t10_ci, t10_sig = compute_paired_bootstrap(deltas["diff_top10"], 0.0)
            rank_delta, rank_ci, rank_sig = compute_paired_bootstrap(deltas["diff_rank"], 0.0)
            return {
                "present": True,
                "scope": deltas["scope"],
                "mrr_delta": mrr_delta, "mrr_ci": mrr_ci, "mrr_sig": mrr_sig,
                "t10_delta": t10_delta, "t10_ci": t10_ci, "t10_sig": t10_sig,
                # Rank delta (positive = real ranks the true image better than control).
                "rank_delta": rank_delta, "rank_ci": rank_ci, "rank_sig": rank_sig,
                "median_rank_real": deltas["median_rank_real"],
                "median_rank_ctrl": deltas["median_rank_ctrl"],
            }

        shuf = _compare_control(shuf_key, "shuffled")
        rand = _compare_control(rand_key, "random")

        # Evaluate Gate Criteria
        real_metrics = next((r for r in runs_metrics if r["Target"] == target_name and r["Mode"] == "real"), None)
        shuf_metrics = next((r for r in runs_metrics if r["Target"] == target_name and r["Mode"] == "shuffled"), None)
        rand_metrics = next((r for r in runs_metrics if r["Target"] == target_name and r["Mode"] == "random"), None)

        gate_passed = False
        gate_reasons = []

        # Prefer the honest full-bank metrics for the "real > controls" comparison,
        # falling back to within-val when full-bank is unavailable (finding H1).
        def _pick(m, full_key, norm_key):
            if m is None:
                return None
            v = m.get(full_key)
            return v if v is not None else m.get(norm_key, 0.0)

        if not (real_metrics and shuf_metrics and rand_metrics):
            gate_reasons.append("Missing one or more of real/shuffled/random runs (cannot gate)")
        elif not (shuf["present"] and rand["present"]):
            gate_reasons.append(
                "; ".join(c["reason"] for c in (shuf, rand) if not c["present"])
            )
        else:
            # 1. Paired Δ MRR > +0.005 for both controls (numeric, full-bank when present)
            cond_mrr_delta = (shuf["mrr_delta"] > 0.005) and (rand["mrr_delta"] > 0.005)
            if not cond_mrr_delta:
                gate_reasons.append(
                    f"Paired delta MRR below +0.005 (shuffled={shuf['mrr_delta']:+.4f}, random={rand['mrr_delta']:+.4f})"
                )

            # 2. CI lower bound > 0 for both controls — using the numeric CI directly
            cond_ci = (shuf["mrr_ci"][0] > 0.0) and (rand["mrr_ci"][0] > 0.0)
            if not cond_ci:
                gate_reasons.append(
                    f"Confidence interval includes 0 (shuffled=[{shuf['mrr_ci'][0]:+.4f}, {shuf['mrr_ci'][1]:+.4f}], "
                    f"random=[{rand['mrr_ci'][0]:+.4f}, {rand['mrr_ci'][1]:+.4f}])"
                )

            # 3. Paired Δ Top10 > 0 for both controls (finding M5: actually gate on it)
            cond_t10_delta = (shuf["t10_delta"] > 0.0) and (rand["t10_delta"] > 0.0)
            if not cond_t10_delta:
                gate_reasons.append(
                    f"Paired delta Top10 not > 0 (shuffled={shuf['t10_delta']:+.4f}, random={rand['t10_delta']:+.4f})"
                )

            # 4. Real > Shuffled and Random on MRR and Top10 (full-bank when available)
            real_mrr = _pick(real_metrics, "MRR_Full", "MRR_Norm")
            shuf_mrr = _pick(shuf_metrics, "MRR_Full", "MRR_Norm")
            rand_mrr = _pick(rand_metrics, "MRR_Full", "MRR_Norm")
            real_t10 = _pick(real_metrics, "Top10_Full", "Top10_Norm")
            shuf_t10 = _pick(shuf_metrics, "Top10_Full", "Top10_Norm")
            rand_t10 = _pick(rand_metrics, "Top10_Full", "Top10_Norm")
            cond_better = (real_mrr > shuf_mrr) and (real_mrr > rand_mrr) and \
                          (real_t10 > shuf_t10) and (real_t10 > rand_t10)
            if not cond_better:
                gate_reasons.append("Real performance not superior to both controls on MRR and Top10")

            # 5. StdRatio between 0.3 and 2.0
            cond_ratio = 0.3 <= real_metrics["StdRatio"] <= 2.0
            if not cond_ratio:
                gate_reasons.append(f"StdRatio out of bounds [0.3, 2.0] (value={real_metrics['StdRatio']:.3f})")

            # 6. collapse_pct < 20.0
            cond_collapse = real_metrics["CollapsePct"] < 20.0
            if not cond_collapse:
                gate_reasons.append(f"Collapse percentage too high (value={real_metrics['CollapsePct']:.1f}%)")

            # 7. Paired rank-delta CI excludes 0 for both controls: the real model
            # ranks the true image strictly better than each control. This catches a
            # diffuse median-rank win invisible to the MRR/Top10 conditions above.
            cond_rank = (shuf["rank_ci"][0] > 0.0) and (rand["rank_ci"][0] > 0.0)
            if not cond_rank:
                gate_reasons.append(
                    f"Paired rank-delta CI includes 0 "
                    f"(shuffled=[{shuf['rank_ci'][0]:+.1f}, {shuf['rank_ci'][1]:+.1f}], "
                    f"random=[{rand['rank_ci'][0]:+.1f}, {rand['rank_ci'][1]:+.1f}])"
                )

            gate_passed = (cond_mrr_delta and cond_ci and cond_t10_delta
                           and cond_better and cond_ratio and cond_collapse and cond_rank)

        def _fmt_delta(c, key):
            return f"{c[key]:+.4f}" if c.get("present") else "N/A"

        def _fmt_ci(c, key):
            return f"[{c[key][0]:+.4f}, {c[key][1]:+.4f}]" if c.get("present") else "N/A"

        def _fmt_sig(c, key):
            return f"{c[key]:.1%}" if c.get("present") else "N/A"

        scope_tag = shuf.get("scope") if shuf.get("present") else (rand.get("scope") if rand.get("present") else "n/a")

        def _fmt_rank(c, key):
            return f"{c[key]:.0f}" if c.get("present") else "N/A"

        # Real median rank is identical across controls (same real run); take it from
        # whichever control is present for reporting.
        real_median_rank = (
            shuf.get("median_rank_real") if shuf.get("present")
            else (rand.get("median_rank_real") if rand.get("present") else None)
        )
        bootstrap_results.append({
            "Target": target_name,
            "Rank Scope": scope_tag,
            "Real MedianRank": f"{real_median_rank:.0f}" if real_median_rank is not None else "N/A",
            "Shuf MedianRank": _fmt_rank(shuf, "median_rank_ctrl"),
            "Rand MedianRank": _fmt_rank(rand, "median_rank_ctrl"),
            "Shuf RankΔ": _fmt_delta(shuf, "rank_delta"),
            "Shuf RankΔ 95% CI": _fmt_ci(shuf, "rank_ci"),
            "Rand RankΔ": _fmt_delta(rand, "rank_delta"),
            "Rand RankΔ 95% CI": _fmt_ci(rand, "rank_ci"),
            "Shuf MRR Δ": _fmt_delta(shuf, "mrr_delta"),
            "Shuf MRR 95% CI": _fmt_ci(shuf, "mrr_ci"),
            "Shuf Sign Rate": _fmt_sig(shuf, "mrr_sig"),
            "Shuf Top10 Δ": _fmt_delta(shuf, "t10_delta"),
            "Rand MRR Δ": _fmt_delta(rand, "mrr_delta"),
            "Rand MRR 95% CI": _fmt_ci(rand, "mrr_ci"),
            "Rand Sign Rate": _fmt_sig(rand, "mrr_sig"),
            "Rand Top10 Δ": _fmt_delta(rand, "t10_delta"),
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
- **Rank scope**: full-bank retrieval when available (honest), else within-val (inflated, diagnostic)
- **Paired Δ MRR**: > +0.005 for both controls (numeric CI, no string parsing)
- **Confidence Interval**: lower bound > 0 for both controls
- **Paired Δ Top10**: > 0 for both controls
- **Paired rank-delta**: 95% CI lower bound > 0 for both controls (real ranks true image better; catches a diffuse median-rank win invisible to MRR/Top10)
- **Comparisons**: Real > Shuffled and Random on MRR and Top10
- **StdRatio**: Between 0.3 and 2.0
- **Dimension Collapse**: < 20% collapse percentage
- A missing control run is an explicit gate failure (never silently coerced to 0).
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

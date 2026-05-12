#!/usr/bin/env python3
import subprocess
import sys
import json
import numpy as np
from pathlib import Path

def get_latest_metrics(base_dir: Path, target_slug: str):
    # Finds the latest run output for a given slug
    dirs = list(base_dir.glob(f"*{target_slug}*/"))
    if not dirs:
        return None
    dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    metrics_file = dirs[0] / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            return json.load(f)
    return None

def main():
    seeds = [13, 42, 123]
    val_runs = 8
    
    results = []
    
    for seed in seeds:
        print(f"=====================================")
        print(f" Running Matrix for seed={seed}")
        print(f"=====================================")
        
        cmd = [
            "python", "scripts/run_baseline_matrix.py",
            "--window-mode", "full5s_backaligned",
            "--add-event-marker",
            "--semantic-target", "text",
            "--text-embeddings", "data/processed/clip_embeddings/imagenet_text_embeddings.pt",
            "--model", "temporal_attn",
            "--val-runs", str(val_runs),
            "--epochs", "50",
            "--batch-size", "64",
            "--device", "cuda",
            "--slug", f"replicate_text_seed{seed}",
            "--metadata", "data/processed/semantic_epochs/zuna_full5s_backaligned_sub-01_runs0102030405060708/all_runs_metadata.csv",
            "--epochs-dir", "data/processed/semantic_epochs/zuna_full5s_backaligned_sub-01_runs0102030405060708",
            "--clip-embeddings", "data/processed/clip_embeddings/sub01_image_embeddings.pt",
            "--conditions", "zuna_real", "zuna_shuffled", "zuna_random",
            "--seed", str(seed)
        ]
        
        # Run it
        subprocess.run(cmd, check=True)
        
        # Now parse the results
        # run_baseline_matrix.py saves in outputs/baseline_matrix/...
        base_dir = Path("outputs/baseline_matrix")
        
        # We need the most recently created matrix directory
        matrix_dirs = list(base_dir.glob("*_matrix"))
        matrix_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_matrix = matrix_dirs[0]
        
        real_metrics = get_latest_metrics(latest_matrix, "zuna_real")
        shuffled_metrics = get_latest_metrics(latest_matrix, "zuna_shuffled")
        random_metrics = get_latest_metrics(latest_matrix, "zuna_random")
        
        res = {
            "seed": seed,
            "real_top10": real_metrics["top10"],
            "shuffled_top10": shuffled_metrics["top10"],
            "random_top10": random_metrics["top10"],
            "real_mrr": real_metrics["mrr"],
            "shuffled_mrr": shuffled_metrics["mrr"],
            "random_mrr": random_metrics["mrr"],
            "collapse_score": real_metrics["collapse_score"]
        }
        
        res["delta_top10_shuffled"] = res["real_top10"] - res["shuffled_top10"]
        res["delta_top10_random"] = res["real_top10"] - res["random_top10"]
        res["delta_mrr_shuffled"] = res["real_mrr"] - res["shuffled_mrr"]
        res["delta_mrr_random"] = res["real_mrr"] - res["random_mrr"]
        
        results.append(res)
    
    # Calculate stats
    delta_top10_shuffled = [r["delta_top10_shuffled"] for r in results]
    delta_top10_random = [r["delta_top10_random"] for r in results]
    delta_mrr_shuffled = [r["delta_mrr_shuffled"] for r in results]
    delta_mrr_random = [r["delta_mrr_random"] for r in results]
    collapse_scores = [r["collapse_score"] for r in results]
    
    wins_shuffled = sum(1 for x in delta_top10_shuffled if x > 0)
    wins_random = sum(1 for x in delta_top10_random if x > 0)
    
    print("\n\n=====================================")
    print(" REPLICATION RESULTS")
    print("=====================================")
    for r in results:
        print(f"Seed {r['seed']}:")
        print(f"  Top10: Real={r['real_top10']:.3f}, Shuffled={r['shuffled_top10']:.3f}, Random={r['random_top10']:.3f}")
        print(f"  MRR: Real={r['real_mrr']:.3f}, Shuffled={r['shuffled_mrr']:.3f}, Random={r['random_mrr']:.3f}")
        print(f"  Collapse Score: {r['collapse_score']:.3f}")
        
    print("\nSTATISTICS ACROSS SEEDS:")
    print(f"Delta Top10 vs Shuffled: mean={np.mean(delta_top10_shuffled):.3f}, std={np.std(delta_top10_shuffled):.3f}")
    print(f"Delta Top10 vs Random  : mean={np.mean(delta_top10_random):.3f}, std={np.std(delta_top10_random):.3f}")
    print(f"Delta MRR vs Shuffled  : mean={np.mean(delta_mrr_shuffled):.3f}, std={np.std(delta_mrr_shuffled):.3f}")
    print(f"Delta MRR vs Random    : mean={np.mean(delta_mrr_random):.3f}, std={np.std(delta_mrr_random):.3f}")
    print(f"Collapse Score         : mean={np.mean(collapse_scores):.3f}, std={np.std(collapse_scores):.3f}")
    
    print(f"\nWINS (Real Top10 > Control):")
    print(f"  vs Shuffled: {wins_shuffled}/{len(seeds)}")
    print(f"  vs Random  : {wins_random}/{len(seeds)}")
    
    if wins_shuffled > len(seeds)/2 and wins_random > len(seeds)/2:
        print("\nCONCLUSION: WEAK POSITIVE REPLICATES! Signal is likely real.")
    else:
        print("\nCONCLUSION: SIGNAL DOES NOT REPLICATE. It was likely noise.")

if __name__ == "__main__":
    main()

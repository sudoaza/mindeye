#!/usr/bin/env python3
import os
import subprocess
import json
import numpy as np
from pathlib import Path

# Config
MODES = ["cosine", "attr", "cosine_attr", "cosine_attr_div"]
FOLDS = [8, 16, 24, 32]
CONDITIONS = ["real", "no-probe", "shuffled", "random"]

# Checkpoints
FOLD_RUNS_DIR = "/workspace/mindeye/outputs/fold_replication/20260522_214524_replication/runs"
BASELINE_MATRIX_DIR = "/workspace/mindeye/outputs/baseline_matrix/20260522_200958_matrix"

CHECKPOINTS = {
    "real": {
        8: f"{FOLD_RUNS_DIR}/20260522_215227_zuna_real_val8_w005/best.pt",
        16: f"{FOLD_RUNS_DIR}/20260522_220445_zuna_real_val16_w005/best.pt",
        24: f"{FOLD_RUNS_DIR}/20260522_221634_zuna_real_val24_w005/best.pt",
        32: f"{FOLD_RUNS_DIR}/20260522_222941_zuna_real_val32_w005/best.pt",
    },
    "no-probe": {
        8: f"{FOLD_RUNS_DIR}/20260522_214540_zuna_real_val8_w000/best.pt",
        16: f"{FOLD_RUNS_DIR}/20260522_215714_zuna_real_val16_w000/best.pt",
        24: f"{FOLD_RUNS_DIR}/20260522_220820_zuna_real_val24_w000/best.pt",
        32: f"{FOLD_RUNS_DIR}/20260522_222134_zuna_real_val32_w000/best.pt",
    },
    # Shuffled and random didn't get fold replication, so we test the baseline on the split
    "shuffled": {
        f: f"{BASELINE_MATRIX_DIR}/20260522_201405_zuna_shuffled_zuna_shuffled_ablation_probe/best.pt" for f in FOLDS
    },
    "random": {
        f: f"{BASELINE_MATRIX_DIR}/20260522_201745_zuna_random_zuna_random_ablation_probe/best.pt" for f in FOLDS
    }
}

EVAL_SCRIPT = "scripts/evaluate_retrieved_priors.py"
INDEX_PREFIX = "data/processed/clip_embeddings/common_index"
COMMON_EMBS = "data/processed/clip_embeddings/common_embeddings.pt"
VLM_ATTRS = "/workspace/mindeye/outputs/common_probe/vlm_attributes_runs01_40.json"
COMMON_PROBE = "/workspace/mindeye/outputs/common_probe/common_probe.pt"
OUT_BASE = "outputs/evaluation_results/phase10c_ablations"

# Note: We must pass a metadata config override if we are testing a non-fold model on a fold split.
# Actually, evaluate_retrieved_priors.py reads the split from the checkpoint's config.
# If we test 'shuffled' on fold 8, it might use its default random split unless we override.
# The user wants cross-fold reranking. For the baseline models (shuffled/random), we can just let them 
# evaluate on their default val split (which is random 15%) since they are just controls.
# But for exactness, let's just evaluate them without changing their split, or we could pass --override-val-runs (not implemented).

def run_job(mode, cond, fold):
    ckpt = CHECKPOINTS[cond][fold]
    if not os.path.exists(ckpt):
        print(f"Skipping {cond} fold {fold}: checkpoint not found")
        return None
        
    out_dir = f"{OUT_BASE}/{cond}_fold{fold}_{mode}"
    os.makedirs(out_dir, exist_ok=True)
    
    cmd = [
        "python", EVAL_SCRIPT,
        "--checkpoint", ckpt,
        "--index-prefix", INDEX_PREFIX,
        "--common-embeddings", COMMON_EMBS,
        "--vlm-attributes", VLM_ATTRS,
        "--common-probe", COMMON_PROBE,
        "--output-dir", out_dir,
        "--rerank-mode", mode,
        "--top-k", "5",
        "--num-grid-examples", "15" # Just enough to check grids for cosine_attr_div
    ]
    
    # Run evaluation
    print(f"Running {cond} fold {fold} mode {mode}...")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    report_path = f"{out_dir}/retrieval_evaluation.json"
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            return json.load(f)["metrics"]
    return None

def main():
    results = {m: {c: [] for c in CONDITIONS} for m in MODES}
    
    for mode in MODES:
        for cond in CONDITIONS:
            for fold in FOLDS:
                metrics = run_job(mode, cond, fold)
                if metrics:
                    results[mode][cond].append(metrics)
                    
    # Aggregation
    print("\n\n=== PHASE 10C AGGREGATE RESULTS ===")
    
    for mode in MODES:
        print(f"\n--- Reranking Mode: {mode.upper()} ---")
        for cond in CONDITIONS:
            runs = results[mode][cond]
            if not runs: continue
            
            top1_sim = np.mean([r["clip_similarity"]["top1"] for r in runs])
            top1_attr = np.mean([r["attribute_agreement"]["top1_mean"] for r in runs])
            
            print(f"{cond:<10} | Top-1 Cosine: {top1_sim:.4f} | Top-1 Attr Agreement: {top1_attr:.4f}")

    print("\n\n=== PER-ATTRIBUTE GAIN TABLE (cosine_attr_div) ===")
    target_mode = "cosine_attr_div"
    
    attrs = [
        "is_animate", "human_visible", "face_visible", "animal_visible", 
        "furry", "soft_texture", "indoor_outdoor", "dominant_color"
    ]
    
    print(f"{'Attribute':<20} | {'Real':<8} | {'Shuffled':<8} | {'Random':<8} | {'Delta(R-S)':<8}")
    print("-" * 65)
    
    real_runs = results[target_mode]["real"]
    shuff_runs = results[target_mode]["shuffled"]
    rand_runs = results[target_mode]["random"]
    
    for attr in attrs:
        real_val = np.mean([r["attribute_agreement"]["per_attribute_top1"][attr] for r in real_runs]) if real_runs else 0
        shuff_val = np.mean([r["attribute_agreement"]["per_attribute_top1"][attr] for r in shuff_runs]) if shuff_runs else 0
        rand_val = np.mean([r["attribute_agreement"]["per_attribute_top1"][attr] for r in rand_runs]) if rand_runs else 0
        delta = real_val - shuff_val
        
        print(f"{attr:<20} | {real_val:.4f}   | {shuff_val:.4f}   | {rand_val:.4f}   | {delta:+.4f}")

    print("\nDone.")

if __name__ == "__main__":
    main()

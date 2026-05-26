#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
import pandas as pd
from collections import Counter

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vlm-json", default="data/processed/clip_embeddings/vlm_attributes.json")
    p.add_argument("--metadata", default="data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv,data/processed/semantic_epochs/zuna_tight1s_sub02_runs01_40/all_runs_metadata.csv,data/processed/semantic_epochs/zuna_tight1s_sub03_runs01_40/all_runs_metadata.csv,data/processed/semantic_epochs/zuna_tight1s_sub04_runs01_40/all_runs_metadata.csv")
    p.add_argument("--output-dir", default="outputs/vlm_audit")
    return p.parse_args()

def main():
    args = parse_args()
    
    # Load VLM attributes
    vlm_path = Path(args.vlm_json)
    if not vlm_path.exists():
        print(f"Error: {vlm_path} not found.")
        return
        
    with open(vlm_path, "r") as f:
        vlm_data = json.load(f)
    print(f"Loaded {len(vlm_data)} image annotations from {vlm_path}.")
    
    # Load all metadata files
    metadata_paths = [p.strip() for p in args.metadata.split(",")]
    dfs = []
    subject_images = {}
    for p in metadata_paths:
        p = Path(p)
        if not p.exists():
            print(f"Warning: Metadata file {p} not found.")
            continue
        df = pd.read_csv(p)
        dfs.append(df)
        sub_name = p.parent.name.split("_")[-3] # e.g. sub01 from zuna_tight1s_sub01_runs01_40
        subject_images[sub_name] = set(df["image_id"].unique())
        
    if not dfs:
        print("Error: No metadata files loaded.")
        return
        
    all_df = pd.concat(dfs, ignore_index=True)
    all_unique_images = set(all_df["image_id"].unique())
    print(f"Total unique images in metadata across all subjects: {len(all_unique_images)}")
    
    # Task Coverage Report
    coverage_report = {}
    total_annotated = 0
    missing_images = []
    
    for img_id in all_unique_images:
        if img_id in vlm_data:
            total_annotated += 1
        else:
            missing_images.append(img_id)
            
    coverage_report["total_images_in_metadata"] = len(all_unique_images)
    coverage_report["total_annotated"] = total_annotated
    coverage_report["overall_coverage_pct"] = (total_annotated / len(all_unique_images)) * 100 if all_unique_images else 0.0
    
    # Subject coverage breakdown
    subject_coverage = {}
    for sub, img_set in subject_images.items():
        sub_annotated = sum(1 for img in img_set if img in vlm_data)
        subject_coverage[sub] = {
            "total_images": len(img_set),
            "annotated": sub_annotated,
            "coverage_pct": (sub_annotated / len(img_set)) * 100 if img_set else 0.0
        }
    coverage_report["subject_breakdown"] = subject_coverage
    
    # Per-task class balances
    # Collect all tasks/attributes
    attributes = [
        "is_animate", "human_visible", "face_visible", "animal_visible",
        "indoor_outdoor", "natural_artificial", "scene_dominance", "real_world_size",
        "dominant_color", "main_subject_position_x", "subject_scale",
        "soft_texture", "spiky_or_pointed", "furry", "metallic",
        "tool_like", "vehicle_like", "food_like"
    ]
    
    class_balances = {}
    for attr in attributes:
        vals = []
        for img_id in all_unique_images:
            if img_id in vlm_data:
                val = vlm_data[img_id].get(attr, "missing")
                vals.append(val)
        counts = Counter(vals)
        total = sum(counts.values())
        class_balances[attr] = {
            k: {
                "count": v,
                "pct": (v / total) * 100 if total else 0.0
            } for k, v in counts.items()
        }
        
    audit_results = {
        "coverage_report": coverage_report,
        "class_balances": class_balances,
        "missing_images_count": len(missing_images)
    }
    
    # Create output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save main report
    report_path = out_dir / "vlm_audit_report.json"
    with open(report_path, "w") as f:
        json.dump(audit_results, f, indent=2)
    print(f"Saved VLM audit report to {report_path}")
    
    # Save missing image IDs
    missing_path = out_dir / "missing_images.txt"
    with open(missing_path, "w") as f:
        for img in sorted(missing_images):
            f.write(f"{img}\n")
    print(f"Saved {len(missing_images)} missing image IDs to {missing_path}")
    
    # Print a summary table to stdout
    print("\n" + "="*50)
    print("VLM AUDIT SUMMARY")
    print("="*50)
    print(f"Overall Coverage: {total_annotated} / {len(all_unique_images)} ({coverage_report['overall_coverage_pct']:.2f}%)")
    print("-" * 50)
    print(f"{'Subject':<10} | {'Annotated':<10} / {'Total':<10} | {'Coverage %':<10}")
    print("-" * 50)
    for sub, stats in subject_coverage.items():
        print(f"{sub:<10} | {stats['annotated']:<10} / {stats['total_images']:<10} | {stats['coverage_pct']:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()

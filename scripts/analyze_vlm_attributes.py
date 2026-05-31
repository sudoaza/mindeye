#!/usr/bin/env python3
"""Audit VLM attribute JSON: image coverage, per-attribute unclear rate, missing keys.

See docs/VLM_ATTRIBUTES.md for schema tiers and backfill procedure.

Usage:
  python scripts/analyze_vlm_attributes.py \\
    --vlm-json outputs/common_probe/vlm_attributes_runs01_40.json \\
    --output-dir outputs/vlm_audit
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindseye.models.common_probe import (
    ALL_VLM_ATTRIBUTE_NAMES,
    CALIBRATION_ATTRIBUTE_NAMES,
    TIER1_ATTRIBUTE_NAMES,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--vlm-json",
        default="outputs/common_probe/vlm_attributes_runs01_40.json",
        help="Path to vlm_attributes JSON",
    )
    p.add_argument(
        "--metadata",
        default=(
            "data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv,"
            "data/processed/semantic_epochs/zuna_tight1s_sub02_runs01_40/all_runs_metadata.csv,"
            "data/processed/semantic_epochs/zuna_tight1s_sub03_runs01_40/all_runs_metadata.csv,"
            "data/processed/semantic_epochs/zuna_tight1s_sub04_runs01_40/all_runs_metadata.csv"
        ),
        help="Comma-separated metadata CSVs",
    )
    p.add_argument("--output-dir", default="outputs/vlm_audit")
    return p.parse_args()


def main():
    args = parse_args()
    vlm_path = Path(args.vlm_json)
    if not vlm_path.exists():
        print(f"Error: {vlm_path} not found.")
        return

    with open(vlm_path, "r") as f:
        vlm_data = json.load(f)
    print(f"Loaded {len(vlm_data)} image annotations from {vlm_path}.")

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
        parts = p.parent.name.split("_")
        sub_name = parts[-3] if len(parts) >= 3 else p.parent.name
        subject_images[sub_name] = set(df["image_id"].astype(str).unique())

    if not dfs:
        print("Error: No metadata files loaded.")
        return

    all_df = pd.concat(dfs, ignore_index=True)
    all_unique_images = set(all_df["image_id"].astype(str).unique())
    print(f"Total unique images in metadata: {len(all_unique_images)}")

    missing_images = [img for img in all_unique_images if img not in vlm_data]
    total_annotated = len(all_unique_images) - len(missing_images)

    subject_coverage = {}
    for sub, img_set in subject_images.items():
        sub_annotated = sum(1 for img in img_set if img in vlm_data)
        subject_coverage[sub] = {
            "total_images": len(img_set),
            "annotated": sub_annotated,
            "coverage_pct": (sub_annotated / len(img_set)) * 100 if img_set else 0.0,
        }

    def _tier_stats(attr_names: tuple[str, ...]) -> dict:
        per_attr = {}
        for attr in attr_names:
            counts: Counter = Counter()
            for img_id in all_unique_images:
                if img_id not in vlm_data:
                    counts["image_missing"] += 1
                    continue
                val = vlm_data[img_id].get(attr, "key_missing")
                counts[val] += 1
            n = sum(counts.values())
            n_unclear = counts.get("unclear", 0) + counts.get("key_missing", 0)
            n_missing_key = counts.get("key_missing", 0)
            per_attr[attr] = {
                "counts": dict(counts),
                "non_unclear_pct": ((n - n_unclear) / n * 100) if n else 0.0,
                "unclear_pct": (counts.get("unclear", 0) / n * 100) if n else 0.0,
                "key_missing_pct": (n_missing_key / n * 100) if n else 0.0,
            }
        return per_attr

    tier1_stats = _tier_stats(TIER1_ATTRIBUTE_NAMES)
    calibration_stats = _tier_stats(CALIBRATION_ATTRIBUTE_NAMES)

    cal_keys_present_pct = 0.0
    if all_unique_images:
        n_with_all_cal = sum(
            1
            for img_id in all_unique_images
            if img_id in vlm_data
            and all(a in vlm_data[img_id] for a in CALIBRATION_ATTRIBUTE_NAMES)
        )
        cal_keys_present_pct = n_with_all_cal / len(all_unique_images) * 100

    audit_results = {
        "vlm_json": str(vlm_path),
        "coverage_report": {
            "total_images_in_metadata": len(all_unique_images),
            "total_annotated": total_annotated,
            "overall_coverage_pct": (total_annotated / len(all_unique_images)) * 100
            if all_unique_images
            else 0.0,
            "calibration_keys_all_present_pct": cal_keys_present_pct,
            "subject_breakdown": subject_coverage,
        },
        "tier1_attributes": tier1_stats,
        "calibration_attributes": calibration_stats,
        "missing_images_count": len(missing_images),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "vlm_audit_report.json"
    with open(report_path, "w") as f:
        json.dump(audit_results, f, indent=2)
    print(f"Saved audit → {report_path}")

    missing_path = out_dir / "missing_images.txt"
    with open(missing_path, "w") as f:
        for img in sorted(missing_images):
            f.write(f"{img}\n")

    print("\n" + "=" * 60)
    print("VLM AUDIT SUMMARY")
    print("=" * 60)
    print(
        f"Image coverage: {total_annotated} / {len(all_unique_images)} "
        f"({audit_results['coverage_report']['overall_coverage_pct']:.2f}%)"
    )
    print(f"All 11 calibration keys present: {cal_keys_present_pct:.2f}% of metadata images")
    print("-" * 60)
    print(f"{'Attribute':<28} | {'non-unclear %':<12} | {'key missing %':<12}")
    print("-" * 60)
    for attr in ALL_VLM_ATTRIBUTE_NAMES:
        block = tier1_stats if attr in TIER1_ATTRIBUTE_NAMES else calibration_stats
        s = block[attr]
        print(
            f"{attr:<28} | {s['non_unclear_pct']:>10.1f}% | {s['key_missing_pct']:>10.1f}%"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()

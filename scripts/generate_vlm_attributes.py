#!/usr/bin/env python3
"""Generate semantic attributes for images using Qwen2-VL.

Outputs a JSON dict: image_id → {attr: label_string}.

Attribute tiers (see docs/VLM_ATTRIBUTES.md):
  tier1       — 18 natural-image semantics (original prompt)
  calibration — 11 Phase 11A material/shape axes (backfill)
  all         — 29 attributes (full ATTRIBUTE_SCHEMAS)

Examples:

  # Full annotation (new bank)
  python scripts/generate_vlm_attributes.py \\
    --tier all --metadata ... --image-dir data/raw/nod/stimuli/ImageNet \\
    --output outputs/common_probe/vlm_attributes_runs01_40.json

  # Backfill only missing calibration keys (merge into existing JSON)
  python scripts/generate_vlm_attributes.py \\
    --tier calibration --merge \\
    --metadata ... --image-dir data/raw/nod/stimuli/ImageNet \\
    --output outputs/common_probe/vlm_attributes_runs01_40.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindseye.models.common_probe import (
    ALL_VLM_ATTRIBUTE_NAMES,
    CALIBRATION_ATTRIBUTE_NAMES,
    TIER1_ATTRIBUTE_NAMES,
    unclear_fallback,
    vlm_json_schema_lines,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", required=True, help="Crop metadata CSV or comma-separated list")
    p.add_argument("--image-dir", required=True, help="ImageNet stimuli directory")
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--tier",
        choices=("tier1", "calibration", "all"),
        default="all",
        help="Which attribute set to request from the VLM (default: all)",
    )
    p.add_argument(
        "--merge",
        action="store_true",
        default=False,
        help="Merge parsed attrs into existing JSON entries (required for calibration backfill)",
    )
    p.add_argument(
        "--no-merge",
        action="store_true",
        help="Replace entire per-image dict on each successful parse (default for tier1-only runs)",
    )
    return p.parse_args()


def _attrs_for_tier(tier: str) -> tuple[str, ...]:
    if tier == "tier1":
        return TIER1_ATTRIBUTE_NAMES
    if tier == "calibration":
        return CALIBRATION_ATTRIBUTE_NAMES
    return ALL_VLM_ATTRIBUTE_NAMES


def _build_system_prompt(attr_names: tuple[str, ...]) -> str:
    schema_body = ",\n".join(vlm_json_schema_lines(attr_names))
    return f"""You are an expert image annotator. Analyze the image and provide a JSON response describing its semantic and visual attributes based exactly on the provided schema.

Return ONLY a valid JSON object matching this schema:
{{
{schema_body}
}}

Use "unclear" when the attribute cannot be determined reliably. Do not invent values outside the allowed options."""


def main():
    args = parse_args()
    merge = args.merge and not args.no_merge
    required_attrs = _attrs_for_tier(args.tier)
    system_prompt = _build_system_prompt(required_attrs)

    print(f"Tier={args.tier}  attributes={len(required_attrs)}  merge={merge}")

    metadata_paths = [p.strip() for p in args.metadata.split(",")]
    dfs = [pd.read_csv(p) for p in metadata_paths]
    df = pd.concat(dfs, ignore_index=True)
    unique_images = df["image_id"].unique()

    image_dir = Path(args.image_dir)
    image_paths = {}
    for img_id in unique_images:
        p = image_dir / f"{img_id}.JPEG"
        if p.exists():
            image_paths[img_id] = str(p)

    print(f"Found {len(image_paths)} images to process out of {len(unique_images)} unique image_ids")

    output_path = Path(args.output)
    results: dict = {}
    if output_path.exists():
        with open(output_path, "r") as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} existing annotations.")

    images_to_process = []
    for k in image_paths.keys():
        if k not in results:
            images_to_process.append(k)
        else:
            missing = any(attr not in results[k] for attr in required_attrs)
            if missing:
                images_to_process.append(k)
    print(f"Remaining images to process: {len(images_to_process)}")

    if not images_to_process:
        print("Done — all images have required keys for this tier.")
        return

    print("Loading Qwen2-VL...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    chunks = [
        images_to_process[i : i + args.batch_size]
        for i in range(0, len(images_to_process), args.batch_size)
    ]

    for chunk in tqdm(chunks):
        messages_batch = []
        for img_id in chunk:
            messages_batch.append(
                [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_paths[img_id]},
                            {"type": "text", "text": "Analyze the image and output the requested JSON."},
                        ],
                    },
                ]
            )

        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_batch
        ]
        image_inputs, video_inputs = process_vision_info(messages_batch)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs, max_new_tokens=256, temperature=0.1, do_sample=False
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        for img_id, out_text in zip(chunk, output_texts):
            try:
                match = re.search(r"\{.*\}", out_text.strip(), re.DOTALL)
                json_str = match.group(0) if match else out_text.strip()
                parsed = json.loads(json_str)
            except Exception as e:
                print(f"Error parsing output for {img_id}: {e}\nOutput was: {out_text}")
                parsed = unclear_fallback(required_attrs)

            if merge and img_id in results:
                results[img_id].update(parsed)
            else:
                results[img_id] = parsed

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"Saved {len(results)} annotations → {output_path}")


if __name__ == "__main__":
    main()

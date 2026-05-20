#!/usr/bin/env python3
"""Generate Tier 1 semantic attributes for images using Qwen2-VL.
Outputs a JSON dictionary mapping image_id to its attributes.
"""

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True, help="Path to crop metadata CSV")
    p.add_argument("--image-dir", required=True, help="Base directory of stimuli images")
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument("--batch-size", type=int, default=4)
    return p.parse_args()

SYSTEM_PROMPT = """You are an expert image annotator. Analyze the image and provide a JSON response describing its semantic and visual attributes based exactly on the provided schema.

Return ONLY a valid JSON object matching this schema:
{
  "is_animate": "yes" | "no" | "unclear",
  "human_visible": "yes" | "no" | "unclear",
  "face_visible": "yes" | "no" | "unclear",
  "animal_visible": "yes" | "no" | "unclear",
  "indoor_outdoor": "indoor" | "outdoor" | "mixed" | "unclear",
  "natural_artificial": "natural" | "artificial" | "mixed" | "unclear",
  "scene_dominance": "isolated_object" | "object_with_background" | "full_scene" | "unclear",
  "real_world_size": "tiny" | "small" | "medium" | "large" | "huge" | "unclear",
  "dominant_color": "red" | "blue" | "green" | "yellow" | "orange" | "purple" | "pink" | "brown" | "black" | "white" | "gray" | "multicolor" | "unclear",
  "lighting_condition": "bright_daylight" | "indoor_warm" | "dim_dark" | "high_contrast" | "studio_artificial" | "unclear",
  "object_presence": "single_object" | "multiple_objects" | "no_clear_object" | "unclear",
  "contrast_level": "high" | "low" | "unclear"
}"""

def main():
    args = parse_args()
    
    df = pd.read_csv(args.metadata)
    # image_id looks like n03594734_45507
    unique_images = df["image_id"].unique()
    
    # Map image_id to actual file path
    image_dir = Path(args.image_dir)
    image_paths = {}
    for img_id in unique_images:
        p = image_dir / f"{img_id}.JPEG"
        if p.exists():
            image_paths[img_id] = str(p)
            
    print(f"Found {len(image_paths)} images to process out of {len(unique_images)} unique image_ids")
    
    # Load previous if exists to resume
    output_path = Path(args.output)
    results = {}
    if output_path.exists():
        with open(output_path, "r") as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} existing annotations.")
        
    images_to_process = [k for k in image_paths.keys() if k not in results]
    print(f"Remaining images to process: {len(images_to_process)}")
    
    if not images_to_process:
        print("Done!")
        return

    # Load Model
    print("Loading Qwen2-VL...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    
    # Create chunks
    chunks = [images_to_process[i:i + args.batch_size] for i in range(0, len(images_to_process), args.batch_size)]
    
    import re
    
    for chunk in tqdm(chunks):
        messages_batch = []
        for img_id in chunk:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_paths[img_id],
                        },
                        {"type": "text", "text": "Analyze the image and output the requested JSON."},
                    ],
                }
            ]
            messages_batch.append(messages)
            
        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_batch]
        
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
            generated_ids = model.generate(**inputs, max_new_tokens=200, temperature=0.1, do_sample=False)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        for img_id, out_text in zip(chunk, output_texts):
            # Parse JSON
            try:
                # Find JSON block if wrapped in markdown
                match = re.search(r'\{.*\}', out_text.strip(), re.DOTALL)
                if match:
                    json_str = match.group(0)
                else:
                    json_str = out_text.strip()
                parsed = json.loads(json_str)
                results[img_id] = parsed
            except Exception as e:
                print(f"Error parsing output for {img_id}: {e}\nOutput was: {out_text}")
                # Provide a fallback
                results[img_id] = {
                  "is_animate": "unclear",
                  "human_visible": "unclear",
                  "face_visible": "unclear",
                  "animal_visible": "unclear",
                  "indoor_outdoor": "unclear",
                  "natural_artificial": "unclear",
                  "scene_dominance": "unclear",
                  "real_world_size": "unclear",
                  "dominant_color": "unclear",
                  "lighting_condition": "unclear",
                  "object_presence": "unclear",
                  "contrast_level": "unclear"
                }
                
        # Save every chunk to avoid data loss
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()

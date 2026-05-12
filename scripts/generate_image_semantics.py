#!/usr/bin/env python3
import argparse
import json
import sys
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata-csv", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--out-parquet", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda")
    return p.parse_args()

def build_caption_core_text(row: dict) -> str:
    return (
        f"{row['short_caption']} "
        f"{row['detailed_caption']} "
        f"Composition: {row['composition_caption']} "
        f"Attributes: {row['attribute_caption']}."
    )

def extract_json(text: str) -> dict:
    match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    try:
        # Fallback if no markdown block
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != 0:
            return json.loads(text[start:end])
    except Exception:
        pass
    return {}

def main():
    args = parse_args()
    
    df = pd.read_csv(args.metadata_csv)
    
    unique_images = df[["image_id", "class"]].drop_duplicates().reset_index(drop=True)
    if args.limit:
        unique_images = unique_images.head(args.limit)

    print(f"Found {len(unique_images)} unique images.")
    
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    
    prompt_text = """You are generating visual-semantic training targets for an EEG-to-image decoding model.

Describe only visible content. Do not infer hidden context, identity, brand, location, story, intention, politics, emotion of people, or events outside the frame. Do not say "image", "photo", "picture", or "visual stimulus" inside the captions.

Return valid JSON only with exactly these keys:

{
  "short_caption": "string",
  "detailed_caption": "string",
  "composition_caption": "string",
  "attribute_caption": "string",
  "objects": ["string"],
  "scene": "string",
  "setting": "string",
  "spatial_layout": "string",
  "dominant_colors": ["string"],
  "materials_textures": ["string"],
  "lighting": "string",
  "viewpoint": "string",
  "action_or_state": "string",
  "mood": "string",
  "uncertainties": ["string"]
}

Field requirements:

short_caption:
- One sentence.
- Name the main visible subject and broad setting.
- Example style: "A brown dog standing on green grass outdoors."

detailed_caption:
- Two sentences maximum.
- Mention main subject, secondary visible objects, background, colors, pose/state, and scene context.
- Keep it concrete and visual.

composition_caption:
- One sentence.
- Describe spatial layout: center/left/right, foreground/background, scale, framing, viewpoint, and object relations.
- Example: "The main subject is centered in the foreground with a grassy background filling most of the frame."

attribute_caption:
- One compact comma-separated phrase list.
- Include object type, color, material/texture, setting, lighting, viewpoint, and action/state.
- Example: "dog, brown fur, green grass, outdoor daylight, centered subject, side view, standing"

objects:
- List visible object categories only.
- Use generic nouns, not proper names.

scene:
- Broad scene type, e.g. animal scene, indoor room, outdoor field, street, forest, water scene, food scene, vehicle scene, person scene, close-up object, abstract/unclear.

setting:
- Concrete environment or background, e.g. grassy field, kitchen table, urban street, forest floor, plain background.

spatial_layout:
- Short phrase describing layout and object placement.

dominant_colors:
- 3 to 6 visible dominant colors.

materials_textures:
- Visible materials/textures such as fur, grass, metal, wood, glass, fabric, skin, water, stone, plastic.

lighting:
- Use concrete visible lighting such as daylight, indoor light, dim light, bright light, backlit, shadowed, unclear.

viewpoint:
- close-up, wide shot, frontal view, side view, top-down, low angle, overhead, unclear.

action_or_state:
- Visible state only: standing, sitting, flying, parked, resting, open, closed, arranged, unclear.

mood:
- Visual atmosphere only: neutral, calm, bright, cluttered, dramatic, natural, playful, unclear.

uncertainties:
- List visible ambiguities. Use [] if none.

Rules:
- Be concise.
- Prefer nouns and visual adjectives.
- Avoid poetic language.
- Avoid abstract interpretation unless directly visible.
- If the image is unclear, say "unclear" in relevant fields."""

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    
    out_f = open(args.out_jsonl, "w")
    results = []
    
    for _, row in tqdm(unique_images.iterrows(), total=len(unique_images)):
        image_id = row["image_id"]
        class_label = row["class"]
        
        # Determine image path. If not found exactly, try with glob
        img_path = Path(args.image_root) / class_label / f"{image_id}.JPEG"
        if not img_path.exists():
            # Try without class subfolder
            img_path = Path(args.image_root) / f"{image_id}.JPEG"
        
        if not img_path.exists():
            # Sometimes image_id includes the class, or has .png
            matches = list(Path(args.image_root).rglob(f"*{image_id}*.*"))
            if matches:
                img_path = matches[0]
            else:
                print(f"Warning: image {image_id} not found in {args.image_root}")
                continue
                
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{img_path}"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(args.device)
        
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
        parsed = extract_json(output_text)
        
        if not parsed:
            # Try to populate a fallback
            parsed = {
                "short_caption": "",
                "detailed_caption": "",
                "composition_caption": "",
                "attribute_caption": "",
                "objects": [],
                "scene": "",
                "setting": "",
                "spatial_layout": "",
                "dominant_colors": [],
                "materials_textures": [],
                "lighting": "",
                "viewpoint": "",
                "action_or_state": "",
                "mood": "",
                "uncertainties": []
            }
            empty_or_failed = True
        else:
            empty_or_failed = False
            
        # Ensure all keys exist
        for key in ["short_caption", "detailed_caption", "composition_caption", "attribute_caption", "objects", "scene", "setting", "spatial_layout", "dominant_colors", "materials_textures", "lighting", "viewpoint", "action_or_state", "mood", "uncertainties"]:
            if key not in parsed:
                parsed[key] = "" if not key.endswith("s") else []

        embedding_text = build_caption_core_text(parsed)
        
        res = {
            "image_id": image_id,
            "image_path": str(img_path),
            "class_label": class_label,
            "vlm_model": args.model,
            "short_caption": parsed["short_caption"],
            "detailed_caption": parsed["detailed_caption"],
            "composition_caption": parsed["composition_caption"],
            "attribute_caption": parsed["attribute_caption"],
            "objects": parsed["objects"],
            "scene": parsed["scene"],
            "setting": parsed["setting"],
            "spatial_layout": parsed["spatial_layout"],
            "dominant_colors": parsed["dominant_colors"],
            "materials_textures": parsed["materials_textures"],
            "lighting": parsed["lighting"],
            "viewpoint": parsed["viewpoint"],
            "action_or_state": parsed["action_or_state"],
            "mood": parsed["mood"],
            "uncertainties": parsed["uncertainties"],
            "embedding_text": embedding_text,
            "quality_flags": {
                "mentions_uncertainty": len(parsed["uncertainties"]) > 0,
                "empty_or_failed": empty_or_failed,
                "too_generic": len(parsed["detailed_caption"]) < 20,
            }
        }
        
        out_f.write(json.dumps(res) + "\n")
        out_f.flush()
        results.append(res)
        
    out_f.close()
    
    res_df = pd.DataFrame(results)
    res_df.to_parquet(args.out_parquet, index=False)
    print(f"Saved {len(results)} image semantics to {args.out_parquet}")

if __name__ == "__main__":
    main()

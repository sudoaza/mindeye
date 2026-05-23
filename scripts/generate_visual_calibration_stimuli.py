#!/usr/bin/env python3
import os
import csv
import math
import random
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

def parse_args():
    p = argparse.ArgumentParser(description="Generate visual calibration stimuli battery")
    p.add_argument("--output-dir", default="data/stimuli/calibration", help="Base output directory")
    p.add_argument("--repeats-multiplier", type=int, default=2, help="Multiplier for trial repeats (default: 2)")
    p.add_argument("--no-diffusion", action="store_true", help="Disable Stable Diffusion generation, use procedural fallback")
    p.add_argument("--num-blocks", type=int, default=4, help="Number of experimental blocks to distribute trials")
    p.add_argument("--seed", type=int, default=42, help="Random seed for generation and trial ordering")
    return p.parse_args()

def set_seed(seed):
    random.seed(seed)

def compute_shape_points(shape_name, center, target_area=70000):
    cx, cy = center
    if shape_name == "circle":
        r = math.sqrt(target_area / math.pi)
        return [cx - r, cy - r, cx + r, cy + r]
    
    elif shape_name == "square":
        s = math.sqrt(target_area)
        return [cx - s/2, cy - s/2, cx + s/2, cy + s/2]
        
    elif shape_name == "triangle":
        # Equilateral triangle area: (sqrt(3)/4) * s^2
        s = math.sqrt(4 * target_area / math.sqrt(3))
        h = s * math.sqrt(3) / 2
        # Center of mass is 1/3 height from bottom
        return [
            (cx, cy - 2*h/3),
            (cx - s/2, cy + h/3),
            (cx + s/2, cy + h/3)
        ]
        
    elif shape_name == "hexagon":
        # Area = 3 * sqrt(3) / 2 * R^2
        r = math.sqrt(2 * target_area / (3 * math.sqrt(3)))
        pts = []
        for i in range(6):
            angle = math.radians(60 * i)
            pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        return pts
        
    elif shape_name == "octagon":
        # Area = 2 * (1 + sqrt(2)) * s^2 = 2 * sin(45) * 8 / 2 * R^2 = 2 * sqrt(2) * R^2
        # R = math.sqrt(A / (2 * sqrt(2)))
        r = math.sqrt(target_area / (2 * math.sqrt(2)))
        pts = []
        for i in range(8):
            angle = math.radians(45 * i)
            pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        return pts

    elif shape_name == "diamond":
        # Rhombus with diagonals d1 = d2 = d. Area = 0.5 * d^2 -> d = sqrt(2A)
        d = math.sqrt(2 * target_area)
        half = d / 2
        return [
            (cx, cy - half),
            (cx + half, cy),
            (cx, cy + half),
            (cx - half, cy)
        ]
        
    elif shape_name == "star":
        # 5-point star. Let Outer radius be Ro, Inner radius Ri = 0.382 * Ro (golden ratio)
        # Area = 5 * Ro * Ri * sin(36 degrees)
        # Ro = sqrt(A / (5 * 0.382 * sin(36)))
        sin_36 = math.sin(math.radians(36))
        ro = math.sqrt(target_area / (5 * 0.382 * sin_36))
        ri = 0.382 * ro
        pts = []
        for i in range(10):
            r = ro if i % 2 == 0 else ri
            angle = math.radians(36 * i - 90) # Start pointing straight up
            pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        return pts

    elif shape_name == "cross":
        # Cross of width w, length L. Area = 2 * w * L - w^2.
        # Let's set w = 85. L = (A + w^2) / (2 * w)
        w = 85
        L = (target_area + w**2) / (2 * w)
        half_w = w / 2
        half_L = L / 2
        # Return list of rect points: [horiz_rect, vert_rect]
        return [
            [cx - half_L, cy - half_w, cx + half_L, cy + half_w],
            [cx - half_w, cy - half_L, cx + half_w, cy + half_L]
        ]
    else:
        raise ValueError(f"Unknown shape: {shape_name}")

def draw_matched_shape(draw, shape_name, points, color):
    if shape_name in ("circle", "square"):
        draw.ellipse(points, fill=color) if shape_name == "circle" else draw.rectangle(points, fill=color)
    elif shape_name == "cross":
        draw.rectangle(points[0], fill=color)
        draw.rectangle(points[1], fill=color)
    else:
        draw.polygon(points, fill=color)

def generate_procedural_texture(img_size, tex_name, bg_color):
    img = Image.new("RGB", img_size, bg_color)
    draw = ImageDraw.Draw(img)
    w, h = img_size
    
    if tex_name == "fur":
        # Draw many fine lines to simulate hair/fur
        for _ in range(1500):
            x1 = random.randint(0, w)
            y1 = random.randint(0, h)
            length = random.randint(10, 30)
            angle = math.radians(random.gauss(45, 10))
            x2 = x1 + length * math.cos(angle)
            y2 = y1 + length * math.sin(angle)
            gray = random.randint(80, 180)
            draw.line([(x1, y1), (x2, y2)], fill=(gray, gray, gray), width=2)
        img = img.filter(ImageFilter.GaussianBlur(1))
        
    elif tex_name == "grass":
        # Draw green-hued spiky blades
        for _ in range(2000):
            x = random.randint(0, w)
            y = random.randint(10, h)
            length = random.randint(15, 35)
            angle = math.radians(random.gauss(-90, 15)) # Pointing upwards
            x2 = x + length * math.cos(angle)
            y2 = y + length * math.sin(angle)
            g = random.randint(100, 200)
            r = random.randint(30, 80)
            draw.line([(x, y), (x2, y2)], fill=(r, g, 40), width=2)
            
    elif tex_name == "metal":
        # Brushed metal gradient + noise
        for x in range(w):
            val = int(140 + 20 * math.sin(x / 50.0))
            draw.line([(x, 0), (x, h)], fill=(val, val, val))
        # Add fine horizontal scratches
        for _ in range(500):
            y = random.randint(0, h)
            x1 = random.randint(0, w - 50)
            x2 = x1 + random.randint(10, 50)
            gray = random.choice([100, 220])
            draw.line([(x1, y), (x2, y)], fill=(gray, gray, gray), width=1)
            
    elif tex_name == "wood":
        # Concentric ring lines + wood color
        for y in range(h):
            dist = math.sqrt((w/2 - y)**2 + (h/2 - y)**2)
            ring = int(130 + 15 * math.sin(dist / 12.0))
            draw.line([(0, y), (w, y)], fill=(ring, ring - 40, ring - 80))
            
    elif tex_name == "fabric":
        # Grid pattern
        img = Image.new("RGB", img_size, (180, 150, 130))
        draw = ImageDraw.Draw(img)
        for i in range(0, w, 4):
            draw.line([(i, 0), (i, h)], fill=(120, 100, 80), width=1)
            draw.line([(0, i), (w, i)], fill=(120, 100, 80), width=1)
            
    elif tex_name == "water":
        # Wavy patterns
        for y in range(h):
            for x in range(w):
                val = int(128 + 40 * math.sin(x / 15.0) * math.cos(y / 15.0))
                img.putpixel((x, y), (val - 50, val, val + 50))
                
    elif tex_name == "stone":
        # Noise + cells
        for y in range(0, h, 4):
            for x in range(0, w, 4):
                gray = random.randint(80, 160)
                draw.rectangle([x, y, x+3, y+3], fill=(gray, gray, gray))
        img = img.filter(ImageFilter.GaussianBlur(1.5))
        
    elif tex_name == "plastic":
        # Solid smooth color with a light gradient
        for y in range(h):
            val = int(140 - 20 * (y / h))
            draw.line([(0, y), (w, y)], fill=(val, val, val + 20))
            
    return img

def main():
    args = parse_args()
    set_seed(args.seed)
    
    output_dir = Path(args.output_dir)
    metadata_file = output_dir / "calibration_metadata.csv"
    trials_file = output_dir / "calibration_trials.csv"
    
    # Ensure folders exist
    for sub in ["color_patches", "shapes", "textures", "spatial", "animacy"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)
        
    metadata_rows = []
    image_size = (500, 500)
    bg_color = (128, 128, 128)
    
    def add_meta(img_id, stim_type, **kwargs):
        row = {
            "image_id": img_id,
            "stimulus_type": stim_type,
            "dominant_color": kwargs.get("dominant_color", "unclear"),
            "shape": kwargs.get("shape", "unclear"),
            "main_subject_position_x": kwargs.get("main_subject_position_x", "center"),
            "subject_scale": kwargs.get("subject_scale", "close_up"),
            "soft_texture": kwargs.get("soft_texture", "unclear"),

            "spiky_or_pointed": kwargs.get("spiky_or_pointed", "unclear"),
            "furry": kwargs.get("furry", "unclear"),
            "metallic": kwargs.get("metallic", "unclear"),
            "tool_like": kwargs.get("tool_like", "unclear"),
            "vehicle_like": kwargs.get("vehicle_like", "unclear"),
            "food_like": kwargs.get("food_like", "unclear"),
            "is_animate": kwargs.get("is_animate", "unclear"),
            "face_visible": kwargs.get("face_visible", "unclear"),
            "animal_visible": kwargs.get("animal_visible", "unclear"),
            "warm_vs_cool": kwargs.get("warm_vs_cool", "unclear"),
            "bright_vs_dark": kwargs.get("bright_vs_dark", "unclear"),
            "round_or_curved": kwargs.get("round_or_curved", "unclear"),
            "angular_or_geometric": kwargs.get("angular_or_geometric", "unclear"),
            "symmetrical": kwargs.get("symmetrical", "unclear"),
            "single_object": kwargs.get("single_object", "unclear"),
            "glossy": kwargs.get("glossy", "unclear"),
            "rough": kwargs.get("rough", "unclear"),
            "smooth": kwargs.get("smooth", "unclear"),
            "transparent": kwargs.get("transparent", "unclear"),
            "organic_texture": kwargs.get("organic_texture", "unclear"),
        }
        metadata_rows.append(row)
        
    print("[Calibration] Initializing visual calibration stimuli generation...")
    
    # 1. Colors Block
    print("[Calibration] Generating Color Patches...")
    COLORS = {
        "red": ((255, 50, 50), "warm"),
        "orange": ((255, 140, 0), "warm"),
        "yellow": ((255, 230, 0), "warm"),
        "green": ((50, 180, 50), "cool"),
        "blue": ((40, 60, 240), "cool"),
        "purple": ((130, 40, 200), "cool"),
        "pink": ((255, 120, 200), "warm"),
        "brown": ((110, 60, 20), "warm"),
        "black": ((10, 10, 10), "neutral"),
        "white": ((245, 245, 245), "neutral"),
        "gray": ((150, 150, 150), "neutral"),
    }
    for i, (c_name, (rgb, temp)) in enumerate(COLORS.items(), 1):
        img = Image.new("RGB", image_size, bg_color)
        draw = ImageDraw.Draw(img)
        # Draw matched central color patch
        draw.rectangle([100, 100, 400, 400], fill=rgb)
        img_id = f"calib_color_{i:03d}"
        img.save(output_dir / "color_patches" / f"{img_id}.jpg")
        add_meta(
            img_id, "color_patch",
            dominant_color=c_name,
            warm_vs_cool=temp,
            single_object="yes",
            symmetrical="yes",
            angular_or_geometric="yes",
        )

    # 2. Shapes Block
    print("[Calibration] Generating Matched-Area Shapes...")
    SHAPES = [
        ("circle", "yes", "no"),
        ("square", "no", "yes"),
        ("triangle", "no", "yes"),
        ("hexagon", "no", "yes"),
        ("octagon", "no", "yes"),
        ("diamond", "no", "yes"),
        ("star", "no", "yes"),
        ("cross", "no", "yes")
    ]
    for i, (shape_name, round_val, angular_val) in enumerate(SHAPES, 1):
        img = Image.new("RGB", image_size, bg_color)
        draw = ImageDraw.Draw(img)
        pts = compute_shape_points(shape_name, (250, 250), target_area=70000)
        draw_matched_shape(draw, shape_name, pts, (200, 200, 200))
        img_id = f"calib_shape_{i:03d}"
        img.save(output_dir / "shapes" / f"{img_id}.jpg")
        add_meta(
            img_id, "shape",
            shape=shape_name,
            round_or_curved=round_val,
            angular_or_geometric=angular_val,
            dominant_color="gray",
            single_object="yes",
            symmetrical="yes"
        )

    # 3. Spatial Block
    print("[Calibration] Generating Spatial Targets...")
    # 9 positions on a 3x3 grid
    coords = [
        ((120, 120), "left"), ((250, 120), "center"), ((380, 120), "right"),
        ((120, 250), "left"), ((250, 250), "center"), ((380, 250), "right"),
        ((120, 380), "left"), ((250, 380), "center"), ((380, 380), "right")
    ]
    for i, (pt, pos_x) in enumerate(coords, 1):
        img = Image.new("RGB", image_size, bg_color)
        draw = ImageDraw.Draw(img)
        # Small white circular dot target (radius 20)
        draw.ellipse([pt[0]-20, pt[1]-20, pt[0]+20, pt[1]+20], fill=(255, 255, 255))
        img_id = f"calib_spatial_{i:03d}"
        img.save(output_dir / "spatial" / f"{img_id}.jpg")
        add_meta(
            img_id, "spatial",
            dominant_color="white",
            main_subject_position_x=pos_x,
            single_object="yes",
            round_or_curved="yes"
        )

    # 4. Textures Block
    print("[Calibration] Generating Texture Patches...")
    TEXTURE_PROMPTS = {
        "fur": ("macro photo of animal fur, highly detailed texture patch, flat 2d", "yes", "yes"),
        "grass": ("macro photo of green grass, flat lawn texture patch", "no", "yes"),
        "metal": ("photo of brushed aluminum metal sheet surface texture", "no", "no"),
        "wood": ("photo of natural wood grain surface texture, flat plank", "no", "no"),
        "fabric": ("macro photo of woven cotton fabric texture, textile pattern", "no", "yes"),
        "water": ("photo of clear blue water surface ripples texture", "no", "no"),
        "stone": ("photo of rough gray stone granite texture surface", "no", "no"),
        "plastic": ("photo of smooth matte colored plastic surface", "no", "no")
    }
    
    use_sd = False
    pipeline = None
    if not args.no_diffusion:
        try:
            import torch
            from diffusers import StableDiffusionPipeline
            if torch.cuda.is_available():
                print("[Calibration] Initializing Stable Diffusion Pipeline on GPU...")
                pipeline = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16,
                    safety_checker=None
                ).to("cuda")
                use_sd = True
            else:
                print("[Calibration] No GPU available, falling back to procedural textures.")
        except Exception as e:
            print(f"[Calibration] Failed to load Stable Diffusion ({e}). Falling back to procedural textures.")
            
    for i, (tex_name, (prompt, soft, furry)) in enumerate(TEXTURE_PROMPTS.items(), 1):
        img_id = f"calib_texture_{i:03d}"
        img_path = output_dir / "textures" / f"{img_id}.jpg"
        
        if img_path.exists():
            print(f"[Calibration] Texture '{tex_name}' already exists, skipping generation.")
        elif use_sd and pipeline is not None:
            # Generate via SD
            print(f"[SD] Generating texture '{tex_name}'...")
            with torch.inference_mode():
                image = pipeline(prompt, num_inference_steps=20, guidance_scale=7.5).images[0]
            # Center crop to 500x500
            image = image.resize((500, 500))
            image.save(img_path)
        else:
            # Procedural fallback
            img = generate_procedural_texture(image_size, tex_name, bg_color)
            img.save(img_path)
            
        add_meta(
            img_id, "texture",
            organic_texture="no" if tex_name in ("metal", "plastic") else "yes",
            soft_texture=soft,
            furry=furry,
            rough="yes" if tex_name in ("stone", "wood", "grass") else "no",
            smooth="yes" if tex_name in ("plastic", "metal", "water") else "no",
            metallic="yes" if tex_name == "metal" else "no"
        )

    # 5. Face/Animacy Block
    print("[Calibration] Generating Face and Animacy Stimuli...")
    ANIMACY_PROMPTS = {
        "human_face": ("close up portrait of a human face, neutral expression, plain gray background, centered", "yes", "yes", "no"),
        "animal_face": ("close up portrait of an animal face, dog or cat, neutral, plain gray background, centered", "yes", "yes", "yes"),
        "human_body": ("photograph of a person standing, full body, plain gray background, centered", "yes", "no", "no"),
        "animal_body": ("photograph of a single animal standing, full body, plain gray background, centered", "yes", "no", "yes"),
        "plant": ("photograph of a green leaf or plant, plain gray background, centered", "no", "no", "no"),
        "inanimate_object": ("photograph of an everyday object, a chair or cup, plain gray background, centered", "no", "no", "no")
    }
    
    for i, (cat_name, (prompt, animate, face, animal)) in enumerate(ANIMACY_PROMPTS.items(), 1):
        img_id = f"calib_animacy_{i:03d}"
        img_path = output_dir / "animacy" / f"{img_id}.jpg"
        
        if img_path.exists():
            print(f"[Calibration] Animacy category '{cat_name}' already exists, skipping generation.")
        elif use_sd and pipeline is not None:
            # Generate via SD
            print(f"[SD] Generating animacy category '{cat_name}'...")
            with torch.inference_mode():
                image = pipeline(prompt, num_inference_steps=20, guidance_scale=7.5).images[0]
            image = image.resize((500, 500))
            image.save(img_path)
        else:
            # Draw placeholder shapes
            img = Image.new("RGB", image_size, bg_color)
            draw = ImageDraw.Draw(img)
            if cat_name == "human_face":
                draw.ellipse([150, 150, 350, 350], fill=(240, 200, 170))
                draw.ellipse([210, 210, 230, 230], fill=(0,0,0))
                draw.ellipse([270, 210, 290, 290], fill=(0,0,0))
                draw.arc([200, 260, 300, 310], 0, 180, fill=(0,0,0), width=4)
            elif cat_name == "animal_face":
                draw.ellipse([160, 180, 340, 340], fill=(130, 90, 60))
                draw.polygon([(160,180), (140,120), (200,160)], fill=(130, 90, 60))
                draw.polygon([(340,180), (360,120), (300,160)], fill=(130, 90, 60))
            elif cat_name == "human_body":
                draw.rectangle([230, 200, 270, 450], fill=(50, 100, 180))
                draw.ellipse([220, 120, 280, 180], fill=(240, 200, 170))
            elif cat_name == "animal_body":
                draw.rectangle([180, 250, 320, 350], fill=(130, 90, 60))
                draw.rectangle([190, 350, 210, 450], fill=(130, 90, 60))
                draw.rectangle([290, 350, 310, 450], fill=(130, 90, 60))
                draw.ellipse([300, 200, 350, 260], fill=(130, 90, 60))
            elif cat_name == "plant":
                draw.rectangle([240, 250, 260, 450], fill=(120, 80, 40))
                draw.ellipse([180, 100, 320, 280], fill=(40, 160, 40))
            else:
                draw.rectangle([180, 180, 320, 420], fill=(200, 80, 80))
            img.save(img_path)
            
        add_meta(
            img_id, "animacy",
            is_animate=animate,
            face_visible=face,
            animal_visible=animal,
            single_object="yes"
        )

        
    # Write metadata CSV
    meta_headers = list(metadata_rows[0].keys())
    with open(metadata_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=meta_headers)
        w.writeheader()
        w.writerows(metadata_rows)
    print(f"[Calibration] Wrote {len(metadata_rows)} metadata rows to {metadata_file}")
    
    # 6. Generate calibration_trials.csv
    print("[Calibration] Constructing experimental trials protocol...")
    trials = []
    trial_idx = 1
    
    # Define trigger mappings
    # Colors: 101-111
    # Shapes: 201-208
    # Textures: 301-308
    # Spatial: 401-409
    # Animacy: 501-506
    trigger_offsets = {
        "color_patch": 100,
        "shape": 200,
        "texture": 300,
        "spatial": 400,
        "animacy": 500
    }
    
    # We will loop over each stimulus type and repeat it based on repeats-multiplier
    # Standard repeats per stimulus type:
    # color: 8, shape: 8, texture: 8, spatial: 6, animacy: 8
    base_repeats = {
        "color_patch": 8,
        "shape": 8,
        "texture": 8,
        "spatial": 6,
        "animacy": 8
    }
    
    for r_idx in range(args.repeats_multiplier):
        for row in metadata_rows:
            img_id = row["image_id"]
            stim_type = row["stimulus_type"]
            
            # Determine which stimulus idx we are dealing with (1-based index from image_id)
            stim_num = int(img_id.split("_")[-1])
            expected_trigger = trigger_offsets[stim_type] + stim_num
            
            repeats_for_stim = base_repeats[stim_type]
            for rep in range(repeats_for_stim):
                duration = random.randint(300, 500)
                jitter = random.randint(800, 1200)
                trial_row = {
                    "trial_id": f"trial_{trial_idx:04d}",
                    "image_id": img_id,
                    "condition": stim_type,
                    "repeat": rep + r_idx * repeats_for_stim,
                    "duration_ms": duration,
                    "fixation_ms": 500,
                    "jitter_ms": jitter,
                    "expected_trigger": expected_trigger,
                    "npz_file": "calibration_epochs.npz",
                    "subject": "sub-01",
                    "run": 1,
                    "class": "unclear",
                    "anchor_sample": 51.2
                }
                # Merge all attributes from row
                for k, v in row.items():
                    if k not in ("image_id", "stimulus_type"):
                        trial_row[k] = v
                trials.append(trial_row)
                trial_idx += 1
                
    # Shuffle trials to avoid order bias
    random.shuffle(trials)
    
    # Distribute trials across blocks
    num_blocks = args.num_blocks
    for idx, t in enumerate(trials):
        t["block"] = (idx % num_blocks) + 1
        
    # Re-sort by block first to present them sequentially block-by-block
    trials = sorted(trials, key=lambda x: (x["block"], x["trial_id"]))
    
    # Write trials CSV
    trial_headers = list(trials[0].keys())
    with open(trials_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=trial_headers)
        w.writeheader()
        w.writerows(trials)

        
    print(f"[Calibration] Wrote {len(trials)} trials to {trials_file} distributed across {num_blocks} blocks.")
    print("[Calibration] Stimuli generation successfully complete!")

if __name__ == "__main__":
    main()

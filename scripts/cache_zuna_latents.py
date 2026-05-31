import os
import sys
import argparse
import glob
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import json

# Ensure import paths work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Ensure ZUNA imports are loaded so we can override lingua.transformer
try:
    import zuna
    zuna_path = os.path.dirname(zuna.__file__)
    sys.path.append(os.path.join(zuna_path, 'inference', 'AY2l', 'lingua'))
except ImportError:
    pass

import lingua.transformer
# Bypass torch.compile for flex_attention to avoid lowering bug:
# AttributeError: 'Symbol' object has no attribute 'get_device'
lingua.transformer.flex_attention_comp = lingua.transformer.flex_attention

from mindseye.zuna.latent_extractor import ZunaLatentExtractor

def main():
    parser = argparse.ArgumentParser(description="Cache ZUNA encoder latents from 5s epochs.")
    parser.add_argument("--epochs-dir", type=str, required=True, help="Directory containing zuna_full5s_backaligned npz and metadata")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save cached latents")
    parser.add_argument("--layers", type=str, default="all", help="Layers to cache (comma-separated, e.g. 'layer_8,post_mmd') or 'all'")
    parser.add_argument("--max-trials", type=int, default=None, help="Maximum number of trials to cache (for Phase 0.5 sweep)")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for extraction")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load all_runs_metadata.csv
    metadata_path = os.path.join(args.epochs_dir, "all_runs_metadata.csv")
    if not os.path.exists(metadata_path):
        # Look for individual metadata files and merge them
        metadata_files = glob.glob(os.path.join(args.epochs_dir, "*_metadata.csv"))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata files found in {args.epochs_dir}")
        print(f"Found {len(metadata_files)} individual metadata files. Merging...")
        dfs = [pd.read_csv(f) for f in metadata_files]
        metadata_df = pd.concat(dfs, ignore_index=True)
    else:
        print(f"Loading metadata from {metadata_path}")
        metadata_df = pd.read_csv(metadata_path)

    # If max_trials is set, truncate metadata
    if args.max_trials is not None:
        metadata_df = metadata_df.iloc[:args.max_trials].copy()
        print(f"Truncated metadata to {args.max_trials} trials for debug sweep.")

    # Initialize Extractor
    print(f"Initializing ZunaLatentExtractor on {args.device}...")
    extractor = ZunaLatentExtractor(device=args.device)
    print("Extractor initialized.")

    # Parse layers list
    all_supported_layers = ["layer_4", "layer_8", "layer_12", "layer_16", "pre_mmd", "post_mmd"]
    if args.layers == "all":
        layers_to_cache = all_supported_layers
    else:
        layers_to_cache = [l.strip() for l in args.layers.split(",") if l.strip() in all_supported_layers]
        
    print(f"Layers to cache: {layers_to_cache}")

    # Group metadata by NPZ file to load them efficiently
    grouped = metadata_df.groupby("npz_file")
    
    latent_records = []
    n_channels = 62
    
    for npz_filename, group_df in tqdm(grouped, desc="Processing runs"):
        npz_path = os.path.join(args.epochs_dir, npz_filename)
        if not os.path.exists(npz_path):
            # Check if there is a zuna prefix or similar
            # Try to resolve relative path issues
            alternative_path = glob.glob(os.path.join(args.epochs_dir, f"*{npz_filename}*"))
            if alternative_path:
                npz_path = alternative_path[0]
            else:
                print(f"Warning: npz file {npz_filename} not found in {args.epochs_dir}. Skipping.")
                continue
                
        # Load NPZ file
        # Contains: eeg [N, 62, 1281], sfreq, times, ch_names
        data = np.load(npz_path, allow_pickle=True)
        eeg_all = torch.tensor(data["eeg"], dtype=torch.float32)
        ch_names = data["ch_names"]
        
        # Load original raw/resampled channel positions from ZUNA preprocessing outputs
        # ZUNA electrode positions are in 10-05 montage, which matches the standard coordinates.
        # Let's get them from standard_1005 montage or ZUNA's own database.
        # Standard ZUNA preprocessing maps channels to 3D positions in self.xyz_extremes space.
        # Since the NPZ files do not contain channel_positions directly, we can read standard 3D positions
        # using mne.channels.make_standard_montage or similar.
        # Let's check how montage is retrieved.
        import mne
        montage = mne.channels.make_standard_montage("standard_1005")
        
        # Match channel names and get 3D coordinates
        ch_pos_dict = montage.get_positions()["ch_pos"]
        ch_coords = []
        for name in ch_names:
            # Clean name (e.g. strip whitespace, uppercase)
            clean_name = name.strip()
            # MNE standard name mappings might differ slightly
            if clean_name in ch_pos_dict:
                pos = ch_pos_dict[clean_name]
            elif clean_name.upper() in ch_pos_dict:
                pos = ch_pos_dict[clean_name.upper()]
            else:
                pos = [0.0, 0.0, 0.0]  # fallback
            ch_coords.append(pos)
            
        ch_pos_template = torch.tensor(ch_coords, dtype=torch.float32) # [n_channels, 3]
        
        # Process in batches
        num_epochs = len(group_df)
        
        for idx_start in range(0, num_epochs, args.batch_size):
            idx_end = min(idx_start + args.batch_size, num_epochs)
            batch_df = group_df.iloc[idx_start:idx_end]
            
            # Epoch indices corresponding to NPZ. Since we know they match index-by-index:
            epoch_indices = batch_df.index.tolist()
            # Wait, the index of batch_df is the index in metadata_df, but we want the local row index in group_df!
            # Since group_df is a subset of metadata_df, its row indices might not be contiguous.
            # But the order of rows in group_df corresponds exactly to the order of epochs in the NPZ!
            # So the epoch index in the NPZ is the relative position of the row within group_df.
            local_indices = [group_df.index.get_loc(idx) for idx in batch_df.index]
            
            eeg_batch = eeg_all[local_indices] # [B, 62, 1281]
            
            # Trim 1281 to exactly 1280 samples for ZUNA
            eeg_batch = eeg_batch[:, :, :1280] # [B, 62, 1280]
            
            # Replicate channel positions template to [B, n_channels, 3]
            ch_pos_batch = ch_pos_template.unsqueeze(0).repeat(len(eeg_batch), 1, 1) # [B, 62, 3]
            
            # Extract latents
            res = extractor(eeg_batch.to(args.device), ch_pos_batch.to(args.device))
            
            # Save per-trial records
            for b_idx in range(len(eeg_batch)):
                row = batch_df.iloc[b_idx]
                
                # Use correct columns from metadata: subject, run, and local row index
                subject_id = int(row.get('subject', 1))
                run_id = int(row.get('run', 1))
                trial_id = local_indices[b_idx]
                
                sample_id = f"sub-{subject_id:02d}_run-{run_id:02d}_trial-{trial_id:03d}"
                
                image_id = str(row.get('image_id', 'MISSING'))
                class_id = str(row.get('class_id', 'MISSING'))
                onset_offset_s = float(row.get('event_offset_s', 3.0))
                onset_tc = int(row.get('anchor_sample', 768) / 32) # e.g. 768 / 32 = 24
                
                record = {
                    "sample_id": sample_id,
                    "subject_id": subject_id,
                    "run_id": run_id,
                    "trial_id": trial_id,
                    "image_id": image_id,
                    "class_id": class_id,
                    "onset_offset_s": onset_offset_s,
                    "onset_tc": onset_tc,
                }
                
                # Save each requested layer as float16 to save disk space
                for l in layers_to_cache:
                    # res[l] shape: [B, N, dim]
                    record[l] = res[l][b_idx].cpu().half()
                    
                latent_records.append(record)

    # Save to disk
    # Extract metadata-only records
    metadata_records = []
    for r in latent_records:
        meta_r = {k: v for k, v in r.items() if k not in all_supported_layers}
        metadata_records.append(meta_r)
        
    out_meta_path = os.path.join(args.output_dir, "metadata.pt")
    print(f"Saving {len(metadata_records)} metadata records to {out_meta_path}...")
    torch.save(metadata_records, out_meta_path)
    
    # Save each layer separately as {sample_id: tensor} dict
    for l in layers_to_cache:
        out_layer_path = os.path.join(args.output_dir, f"latents_{l}.pt")
        print(f"Saving layer '{l}' to {out_layer_path}...")
        layer_dict = {r["sample_id"]: r[l] for r in latent_records}
        torch.save(layer_dict, out_layer_path)
        
    # Also save combined latents.pt only if max-trials is small (to avoid OOM on full dataset)
    if args.max_trials is not None and args.max_trials <= 200:
        out_pt_path = os.path.join(args.output_dir, "latents.pt")
        print(f"Saving combined latent records to {out_pt_path}...")
        torch.save(latent_records, out_pt_path)
    
    # Save metadata.json
    meta_info = {
        "n_channels": n_channels,
        "tc": int(1280 // extractor.model_args.num_fine_time_pts),
        "tf": int(extractor.model_args.num_fine_time_pts),
        "D": int(extractor.model_args.encoder_output_dim),
        "onset_tc": 24,
        "n_trials": len(latent_records),
        "cached_layers": layers_to_cache
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(meta_info, f, indent=2)
        
    print("✓ Caching complete!")

if __name__ == "__main__":
    main()

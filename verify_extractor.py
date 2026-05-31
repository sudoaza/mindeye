import torch
import sys
import os

# Ensure import path includes src/
sys.path.append("/workspace/mindeye/src")

# Ensure ZUNA imports are loaded so we can override lingua.transformer
import zuna
zuna_path = os.path.dirname(zuna.__file__)
sys.path.append(os.path.join(zuna_path, 'inference', 'AY2l', 'lingua'))

import lingua.transformer
# Bypass torch.compile for flex_attention to avoid lowering bug:
# AttributeError: 'Symbol' object has no attribute 'get_device'
lingua.transformer.flex_attention_comp = lingua.transformer.flex_attention

from mindseye.zuna.latent_extractor import ZunaLatentExtractor

def main():
    print("Initializing ZunaLatentExtractor on cuda...")
    extractor = ZunaLatentExtractor(device="cuda")
    print("ZunaLatentExtractor initialized successfully!")
    
    # Load input PT file
    pt_path = "/workspace/mindeye/data/processed/zuna_output/2_pt_input/ds000000_000000_000001_d02_00038_62_1280.pt"
    print(f"Loading input file: {pt_path}")
    pt_data = torch.load(pt_path)
    
    # Select first batch/epoch
    eeg = pt_data['data'][0].unsqueeze(0).to("cuda") # [1, 62, 1280]
    ch_pos = pt_data['channel_positions'][0].unsqueeze(0).to("cuda") # [1, 62, 3]
    
    print(f"Input shapes: eeg={eeg.shape}, ch_pos={ch_pos.shape}")
    
    # Run extractor
    print("Running extractor forward pass...")
    res = extractor(eeg, ch_pos)
    
    print("\n--- Extractor Output Diagnostics ---")
    for key, val in res.items():
        if key == "metadata":
            print(f"metadata: {val}")
        elif isinstance(val, torch.Tensor):
            nan_count = torch.isnan(val).sum().item()
            inf_count = torch.isinf(val).sum().item()
            print(f"{key}: shape={list(val.shape)}, dtype={val.dtype}, mean={val.float().mean().item():.4f}, std={val.float().std().item():.4f}, NaNs={nan_count}, Infs={inf_count}")

if __name__ == "__main__":
    main()

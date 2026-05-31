import torch
import sys
import os

sys.path.append("/workspace/mindeye/src")

from mindseye.zuna.latent_extractor import ZunaLatentExtractor

def main():
    extractor = ZunaLatentExtractor(device="cuda")
    print("max_seqlen of encoder rope_embeddings:", extractor.model.encoder.rope_embeddings.max_seqlen)
    print("freqs_cis shape of encoder rope_embeddings:", extractor.model.encoder.rope_embeddings.freqs_cis.shape)
    
    # Load input PT file
    pt_path = "/workspace/mindeye/data/processed/zuna_output/2_pt_input/ds000000_000000_000001_d02_00038_62_1280.pt"
    pt_data = torch.load(pt_path)
    
    # Check max values of discretized positions
    eeg = pt_data['data'][0].unsqueeze(0).to("cuda") # [1, 62, 1280]
    ch_pos = pt_data['channel_positions'][0].unsqueeze(0).to("cuda") # [1, 62, 3]
    
    tf = extractor.model_args.num_fine_time_pts
    tc = 1280 // tf
    
    # Run discretize_chan_pos with 100 bins
    from apps.AY2latent_bci.eeg_data import discretize_chan_pos
    cp_discrete = discretize_chan_pos(ch_pos[0].cpu(), extractor.xyz_extremes.cpu(), 100)
    print("cp_discrete min/max:", cp_discrete.min().item(), cp_discrete.max().item())
    
    # Run discretize_chan_pos with 50 bins? Or what does the model have?
    print("Model args num_bins:", getattr(extractor.model_args, 'num_bins_discretize_xyz_chan_pos', 'MISSING'))

if __name__ == "__main__":
    main()

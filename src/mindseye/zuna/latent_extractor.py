import os
import sys
import json
import torch
from torch import nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safe_load

# Ensure ZUNA imports work by adding its internal path
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

from lingua.args import dataclass_from_dict
from apps.AY2latent_bci.transformer import DecoderTransformerArgs, EncoderDecoder
from apps.AY2latent_bci.eeg_data import discretize_chan_pos, chop_and_reshape_signals


class ZunaLatentExtractor(nn.Module):
    """
    Frozen ZUNA encoder that exposes intermediate and final layer latents.
    Exposes:
      - layer_{k} for k in {4, 8, 12, 16}: hidden states [B, N, 1024]
      - pre_mmd: latent before bottleneck [B, N, 32]
      - post_mmd: latent after MMD bottleneck [B, N, 32]
    """
    PROBE_LAYERS = [4, 8, 12, 16]

    def __init__(self, hf_repo: str = "Zyphra/ZUNA", device: str = "cuda"):
        super().__init__()
        self.device = device
        self.hf_repo = hf_repo
        
        # Download config and weights
        config_path = hf_hub_download(repo_id=hf_repo, filename="config.json")
        with open(config_path, "r") as f:
            cfig = json.load(f)
            
        self.model_args = dataclass_from_dict(DecoderTransformerArgs, cfig["model"])
        
        weights_path = hf_hub_download(repo_id=hf_repo, filename="model-00001-of-00001.safetensors", token=False)
        sd_st_raw = safe_load(weights_path, device="cpu")
        sd_st = {k.removeprefix("model."): v for k, v in sd_st_raw.items()}
        
        self.model = EncoderDecoder(self.model_args)
        self.model.load_state_dict(sd_st, strict=True)
        self.model = self.model.to(device)
        self.model.eval()
        
        # Freeze all parameters
        for p in self.model.parameters():
            p.requires_grad = False
            
        # Hook registration for intermediate layers
        self.intermediate_outputs = {}
        self.hooks = []
        
        # Set up xyz extremes for discretize_chan_pos (twelves used by ZUNA inference)
        self.xyz_extremes = torch.tensor([ 
            [-0.12, -0.12, -0.12], 
            [ 0.12,  0.12,  0.12]
        ], device=device)
        self.num_bins = 50
        
        self._register_hooks()

    def _register_hooks(self):
        for idx in self.PROBE_LAYERS:
            layer_idx = idx - 1
            layer = self.model.encoder.layers[layer_idx]
            
            def make_hook(layer_name):
                def hook_fn(module, input_tensor, output_tensor):
                    self.intermediate_outputs[layer_name] = output_tensor
                return hook_fn
                
            hook = layer.register_forward_hook(make_hook(f"layer_{idx}"))
            self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def forward(self, eeg: torch.Tensor, ch_pos: torch.Tensor) -> dict:
        """
        Args:
            eeg:    [B, n_channels, 1280]  — 5s epoch at 256Hz
            ch_pos: [B, n_channels, 3]     — 3D electrode positions (metres)
        Returns:
            dict of latents
        """
        B, n_channels, n_timepoints = eeg.shape
        assert n_timepoints == 1280, f"Expected 1280 timepoints, got {n_timepoints}"
        
        # ZUNA-matched *global* normalization. The official pipeline applies a single
        # global z-score (one mean/std scalar across all channels+time) offline, then
        # divides by data_norm=10 at inference to land at std~=0.1 with a +-1 clip.
        # Per-channel z-scoring (the previous behavior) forces every channel to unit
        # variance and therefore ERASES the cross-channel amplitude topography — the very
        # thing that carries the visual evoked response (occipital >> frontal). That
        # flattening was destroying the stimulus signal (raw-EEG probe decodes coarse
        # category at +5.5% over baseline; the per-channel-z'd ZUNA latents were at/below
        # baseline). We reproduce ZUNA's scheme per epoch: global z-score -> std ~= 1, then
        # *0.1 -> std ~= 0.1, clamp to +-1. This preserves relative channel/time amplitudes.
        gmean = eeg.mean(dim=(-2, -1), keepdim=True)
        gstd = eeg.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        eeg = ((eeg - gmean) / gstd) * 0.1
        eeg = eeg.clamp(-1.0, 1.0)
        
        self.intermediate_outputs.clear()
        
        tf = self.model_args.num_fine_time_pts
        tc = 1280 // tf
        orig_seqlen = n_channels * tc
        
        # Discretize channel positions
        ch_pos_discrete = []
        for b in range(B):
            discrete_pos = discretize_chan_pos(ch_pos[b].cpu(), self.xyz_extremes.cpu(), self.num_bins).to(self.device)
            ch_pos_discrete.append(discrete_pos)
        ch_pos_discrete = torch.stack(ch_pos_discrete) # [B, n_channels, 3]
        
        # Reshape each sample in batch
        eeg_reshaped_list = []
        chan_pos_reshaped_list = []
        chan_pos_discrete_reshaped_list = []
        chan_id_reshaped_list = []
        tc_reshaped_list = []
        seq_lens_list = []
        
        for b in range(B):
            eeg_b, cp_b, cpd_b, cid_b, tc_b, sl_b = chop_and_reshape_signals(
                eeg_signal=eeg[b].cpu(),
                chan_pos=ch_pos[b].cpu(),
                chan_pos_discrete=ch_pos_discrete[b].cpu(),
                tf=tf,
                use_coarse_time="B"
            )
            eeg_reshaped_list.append(eeg_b.to(self.device))
            chan_pos_reshaped_list.append(cp_b.to(self.device))
            chan_pos_discrete_reshaped_list.append(cpd_b.to(self.device))
            chan_id_reshaped_list.append(cid_b.to(self.device))
            tc_reshaped_list.append(tc_b.to(self.device))
            seq_lens_list.append(sl_b)
            
        encoder_input = torch.cat(eeg_reshaped_list, dim=0) # [B * orig_seqlen, tf]
        chan_pos_batch = torch.cat(chan_pos_reshaped_list, dim=0)
        chan_pos_discrete_batch = torch.cat(chan_pos_discrete_reshaped_list, dim=0)
        t_coarse_batch = torch.cat(tc_reshaped_list, dim=0)
        seq_lens = torch.tensor(seq_lens_list, device=self.device)
        
        encoder_input = encoder_input.unsqueeze(0) # [1, B * orig_seqlen, tf]
        
        # Setup tok_idx for RoPE
        if self.model.tok_idx_type == "{x,y,z,tc}" and self.model.rope_dim == 4:
            tok_idx = torch.cat((chan_pos_discrete_batch.unsqueeze(0), t_coarse_batch.unsqueeze(0)), dim=2)
        elif self.model.tok_idx_type == "t_coarse" and self.model.rope_dim == 1:
            tok_idx = t_coarse_batch.unsqueeze(0)
        else:
            tok_idx = torch.hstack([torch.arange(sl, device=self.device) for sl in seq_lens_list]).unsqueeze(0).unsqueeze(-1)
            
        do_idx = torch.zeros(encoder_input.shape[1], dtype=torch.bool, device=self.device)
        
        # Run encoder
        with torch.no_grad():
            post_mmd, _ = self.model.encoder(
                token_values=encoder_input,
                seq_lens=seq_lens,
                tok_idx=tok_idx,
                do_idx=do_idx
            )
            
        # Post-process intermediate outputs: extract registers
        res = {}
        for idx in self.PROBE_LAYERS:
            layer_name = f"layer_{idx}"
            h_interleaved = self.intermediate_outputs[layer_name] # [1, B * orig_seqlen * 2, 1024]
            h_reshaped = h_interleaved.reshape(1, B * orig_seqlen, 2, -1)
            registers = h_reshaped[:, :, 0, :] # [1, B * orig_seqlen, 1024]
            
            # Split back into batch of size B
            registers = registers.squeeze(0).view(B, orig_seqlen, -1) # [B, orig_seqlen, 1024]
            res[layer_name] = registers
            
        # Extract pre-mmd by projecting layer_16 register outputs through encoder projection + norm
        # In EncoderTransformer: logits = self.output(self.norm(h))
        # res["layer_16"] is already registers of shape [B, orig_seqlen, 1024]
        # Let's project it exactly like EncoderTransformer.forward does:
        # ZUNA's norm is RMSNorm(args.dim, eps=args.norm_eps)
        layer_16_reg = res["layer_16"].unsqueeze(0) # [1, B * orig_seqlen, 1024]
        with torch.no_grad():
            pre_mmd_reg = self.model.encoder.output(self.model.encoder.norm(layer_16_reg))
            
        res["pre_mmd"] = pre_mmd_reg.squeeze(0).view(B, orig_seqlen, -1) # [B, orig_seqlen, 32]
        res["post_mmd"] = post_mmd.squeeze(0).view(B, orig_seqlen, -1) # [B, orig_seqlen, 32]
        
        # Spatial view: reshaped post_mmd
        res["spatial"] = res["post_mmd"].view(B, n_channels, tc, -1)
        
        res["metadata"] = {
            "N": orig_seqlen,
            "tc": tc,
            "tf": tf,
            "n_channels": n_channels,
            "onset_tc": 24,
            "layer_dims": {
                "layer_4": 1024,
                "layer_8": 1024,
                "layer_12": 1024,
                "layer_16": 1024,
                "pre_mmd": 32,
                "post_mmd": 32
            }
        }
        
        return res

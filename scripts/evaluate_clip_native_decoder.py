import argparse
import sys
import os
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindseye.generation.clip_native_backend import ClipNativeDecoderBackend
from mindseye.models.eeg_encoder import EEGClipEncoder, TemporalAttnEncoder, DualHeadTemporalAttnEncoder
from mindseye.datasets.semantic_pairs import ZunaClipPairDataset, SemanticPairConfig

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="Path to trained EEG model directory")
    p.add_argument("--num-samples", type=int, default=4, help="Number of images to generate")
    p.add_argument("--output", default="outputs/clip_native_eval.png", help="Output grid path")
    return p.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    run_dir = Path(args.run_dir)
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")
    
    with open(config_path) as f:
        config = json.load(f)["setup"]
        
    print(f"Loading EEG Encoder from {run_dir}...")
    model_name = config.get("model", "cnn")
    if model_name in {"temporal_attn", "temporal_attn_small"}:
        if config.get("dual_head", False):
            model = DualHeadTemporalAttnEncoder(
                n_channels=config["eeg_shape"][0],
                embedding_dim=config["embedding_dim"],
                hidden_dim=config["hidden_dim"],
                n_layers=config.get("n_layers", 2 if model_name == "temporal_attn_small" else 4),
                n_heads=config.get("n_heads", 4 if model_name == "temporal_attn_small" else 8),
                dropout=config["dropout"],
                stem_dropout1d=config["stem_dropout1d"],
            ).to(device)
        else:
            model = TemporalAttnEncoder(
                n_channels=config["eeg_shape"][0],
                embedding_dim=config["embedding_dim"],
                hidden_dim=config["hidden_dim"],
                n_layers=config.get("n_layers", 2 if model_name == "temporal_attn_small" else 4),
                n_heads=config.get("n_heads", 4 if model_name == "temporal_attn_small" else 8),
                dropout=config["dropout"],
                stem_dropout1d=config["stem_dropout1d"],
            ).to(device)
    else:
        model = EEGClipEncoder(
            n_channels=config["eeg_shape"][0],
            n_times=config["eeg_shape"][1],
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            dropout=config["dropout"],
            stem_dropout1d=config["stem_dropout1d"],
        ).to(device)
    
    checkpoint_path = run_dir / "best.pt"
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("Loading Validation Data...")
    dataset_config = SemanticPairConfig(
        common_embeddings_pt="data/processed/clip_embeddings/decode_common_embeddings.pt",
        metadata_csv="data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv",
        epochs_dir="data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40",
        window_mode="tight1s",
        target_mode="real",
        input_domain="zuna"
    )
    dataset = ZunaClipPairDataset(dataset_config)
    
    # Simple random split as used in training (val_items=596 implies 85/15 split)
    torch.manual_seed(42)
    indices = torch.randperm(len(dataset)).tolist()
    val_indices = indices[-config["val_items"]:]
    
    # Pick a few samples
    sample_indices = val_indices[:args.num_samples]
    
    batch_eeg = []
    batch_target_raw = []
    for idx in sample_indices:
        item = dataset[idx]
        batch_eeg.append(item["eeg"])
        # For oracle unCLIP generation, we must use raw embeddings
        batch_target_raw.append(item.get("target_raw", item["target"]))
        
    batch_eeg = torch.stack(batch_eeg).to(device)
    oracle_embeds = torch.stack(batch_target_raw).to(device)
    
    print("Loading ClipNativeDecoderBackend...")
    backend = ClipNativeDecoderBackend(device=device)
    
    with torch.inference_mode():
        # Real EEG predicted embeddings
        if config.get("dual_head", False):
            # For dual head model, predicted raw embedding = pred_unit * pred_norm
            pred_unit, pred_norm = model(batch_eeg, return_norm=True)
            if config.get("use_fixed_mean_norm", False):
                mean_norm = config.get("mean_train_norm", 1.0)
                real_eeg_embeds = pred_unit * mean_norm
            else:
                real_eeg_embeds = pred_unit * pred_norm
            
            shuffled_eeg = torch.roll(batch_eeg, shifts=1, dims=0)
            shuff_unit, shuff_norm = model(shuffled_eeg, return_norm=True)
            if config.get("use_fixed_mean_norm", False):
                shuffled_embeds = shuff_unit * mean_norm
            else:
                shuffled_embeds = shuff_unit * shuff_norm
        else:
            # If standard model, use output directly
            real_eeg_embeds = model(batch_eeg)
            shuffled_eeg = torch.roll(batch_eeg, shifts=1, dims=0)
            shuffled_embeds = model(shuffled_eeg)
        
        # Random embeddings
        random_embeds = torch.randn_like(oracle_embeds)
        
        print("Generating images...")
        oracle_imgs = backend.generate_from_embeds(oracle_embeds)
        real_eeg_imgs = backend.generate_from_embeds(real_eeg_embeds)
        shuffled_imgs = backend.generate_from_embeds(shuffled_embeds)
        random_imgs = backend.generate_from_embeds(random_embeds)
        
    print("Constructing grid...")
    all_tensors = []
    for i in range(args.num_samples):
        all_tensors.append(TF.to_tensor(oracle_imgs[i]))
        all_tensors.append(TF.to_tensor(real_eeg_imgs[i]))
        all_tensors.append(TF.to_tensor(shuffled_imgs[i]))
        all_tensors.append(TF.to_tensor(random_imgs[i]))
        
    grid = make_grid(all_tensors, nrow=4, padding=2)
    grid_img = TF.to_pil_image(grid)
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid_img.save(out_path)
    print(f"Saved evaluation grid to {out_path}")
    print("Columns: Oracle | Real EEG | Shuffled | Random")

if __name__ == "__main__":
    main()

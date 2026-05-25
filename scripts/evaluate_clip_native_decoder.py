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
from mindseye.datasets.semantic_pairs import ZunaClipPairDataset, SemanticPairConfig, split_indices

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="Path to trained EEG model directory")
    p.add_argument("--num-samples", type=int, default=8, help="Number of images to generate")
    p.add_argument("--output", default="outputs/clip_native_eval.png", help="Output grid path")
    p.add_argument("--k", type=int, default=5, help="k for soft kNN retrieval")
    p.add_argument("--temperature", type=float, default=0.05, help="Temperature for soft kNN softmax")
    p.add_argument("--stimuli-dir", default="data/raw/nod/stimuli/ImageNet",
                   help="Path to ImageNet stimulus images (*.JPEG)")
    return p.parse_args()

def retrieve_soft_knn(z_pred_unit, train_unit_bank, train_raw_bank, k=5, temp=0.05):
    z_pred_norm = F.normalize(z_pred_unit, dim=-1)
    train_unit_norm = F.normalize(train_unit_bank, dim=-1)
    sim = torch.mm(z_pred_norm, train_unit_norm.t()) # [batch, n_train]
    topk_sim, topk_idx = sim.topk(k, dim=-1)
    weights = F.softmax(topk_sim / temp, dim=-1) # [batch, k]
    
    retrieved_raw = []
    for b in range(len(z_pred_unit)):
        w_raw = (train_raw_bank[topk_idx[b]] * weights[b].unsqueeze(-1)).sum(dim=0)
        retrieved_raw.append(w_raw)
        
    return torch.stack(retrieved_raw)

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    run_dir = Path(args.run_dir)
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")
    
    with open(config_path) as f:
        root_config = json.load(f)
        config = root_config.get("setup", {})
        
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
        input_domain="zuna",
        add_event_marker=root_config.get("add_event_marker", False)
    )
    dataset = ZunaClipPairDataset(dataset_config)
    
    # Use split_indices to perfectly replicate training split
    val_fraction = config.get("val_fraction", 0.15)
    seed = config.get("seed", 13)
    train_indices, val_indices = split_indices(len(dataset), val_fraction=val_fraction, seed=seed)
    
    print(f"Constructing retrieval banks from {len(train_indices)} training samples...")
    train_decode_unit = []
    train_target_raw = []
    for idx in train_indices:
        item = dataset[idx]
        train_decode_unit.append(item["target"])
        train_target_raw.append(item.get("target_raw", torch.zeros_like(item["target"])))
        
    train_decode_unit = torch.stack(train_decode_unit).to(device)
    train_target_raw = torch.stack(train_target_raw).to(device)
    
    # Pick a few samples
    sample_indices = val_indices[:args.num_samples]
    
    batch_eeg = []
    batch_target_raw = []
    batch_image_ids = []
    for idx in sample_indices:
        item = dataset[idx]
        batch_eeg.append(item["eeg"])
        # For oracle unCLIP generation, we must use raw embeddings
        batch_target_raw.append(item.get("target_raw", item["target"]))
        batch_image_ids.append(item.get("image_id", ""))

    batch_eeg = torch.stack(batch_eeg).to(device)
    oracle_embeds = torch.stack(batch_target_raw).to(device)
    
    print("Loading ClipNativeDecoderBackend...")
    backend = ClipNativeDecoderBackend(device=device)
    
    with torch.inference_mode():
        # Predict unit embeddings from EEG
        if config.get("dual_head", False):
            pred_unit, _ = model(batch_eeg, return_norm=True)
            shuffled_eeg = torch.roll(batch_eeg, shifts=1, dims=0)
            shuff_unit, _ = model(shuffled_eeg, return_norm=True)
        else:
            pred_unit = model(batch_eeg)
            shuffled_eeg = torch.roll(batch_eeg, shifts=1, dims=0)
            shuff_unit = model(shuffled_eeg)
            
        # Retrieve target_raw using soft kNN
        real_eeg_embeds = retrieve_soft_knn(pred_unit, train_decode_unit, train_target_raw, k=args.k, temp=args.temperature)
        shuffled_embeds = retrieve_soft_knn(shuff_unit, train_decode_unit, train_target_raw, k=args.k, temp=args.temperature)
        
        # Random embeddings
        random_unit = torch.randn_like(pred_unit)
        random_embeds = retrieve_soft_knn(random_unit, train_decode_unit, train_target_raw, k=args.k, temp=args.temperature)
        
        print("Generating images...")
        oracle_imgs = backend.generate_from_embeds(oracle_embeds)
        real_eeg_imgs = backend.generate_from_embeds(real_eeg_embeds)
        shuffled_imgs = backend.generate_from_embeds(shuffled_embeds)
        random_imgs = backend.generate_from_embeds(random_embeds)
        
    print("Constructing grid...")
    stimuli_dir = Path(args.stimuli_dir)
    all_tensors = []
    for i in range(args.num_samples):
        # Column 1: actual stimulus image (target)
        img_id = batch_image_ids[i] if i < len(batch_image_ids) else ""
        stim_path = stimuli_dir / f"{img_id}.JPEG"
        if stim_path.exists():
            stim_img = Image.open(stim_path).convert("RGB").resize((512, 512))
        else:
            stim_img = Image.new("RGB", (512, 512), (30, 30, 30))  # dark placeholder
        all_tensors.append(TF.to_tensor(stim_img))
        # Columns 2-5: oracle, real EEG kNN, shuffled kNN, random kNN
        all_tensors.append(TF.to_tensor(oracle_imgs[i]))
        all_tensors.append(TF.to_tensor(real_eeg_imgs[i]))
        all_tensors.append(TF.to_tensor(shuffled_imgs[i]))
        all_tensors.append(TF.to_tensor(random_imgs[i]))

    grid = make_grid(all_tensors, nrow=5, padding=4)
    grid_img = TF.to_pil_image(grid)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid_img.save(out_path)
    print(f"Saved evaluation grid to {out_path}")
    print("Columns: Target | Oracle | Real EEG kNN | Shuffled kNN | Random kNN")

if __name__ == "__main__":
    main()

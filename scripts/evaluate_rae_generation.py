#!/usr/bin/env python3
import argparse
import sys
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindseye.generation.rae_backend import RaeDecoderBackend
from mindseye.generation.clip_native_backend import ClipNativeDecoderBackend
from mindseye.models.eeg_encoder import EEGClipEncoder, TemporalAttnEncoder, DualHeadTemporalAttnEncoder
from mindseye.datasets.semantic_pairs import ZunaClipPairDataset, SemanticPairConfig, split_indices
from mindseye.models.common_probe import CommonProbeModel, ATTRIBUTE_SCHEMAS, IGNORE_INDEX

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="Path to trained EEG model directory")
    p.add_argument("--num-samples", type=int, default=100, help="Number of validation samples to evaluate")
    p.add_argument("--batch-size", type=int, default=10, help="Batch size for generation")
    p.add_argument("--k", type=int, default=5, help="k for soft kNN retrieval")
    p.add_argument("--temperature", type=float, default=0.05, help="Temperature for soft kNN softmax")
    p.add_argument("--common-probe", default="outputs/decode_probe_v2/common_probe.pt", help="Path to pretrained attribute probe model")
    p.add_argument("--stimuli-dir", default="data/raw/nod/stimuli/ImageNet", help="Path to ImageNet stimulus images")
    p.add_argument("--output-dir", default="outputs/rae_generation_eval", help="Output directory")
    p.add_argument("--target-key", default="image_id_to_rae_centered_unit", help="Key for target RAE vectors")
    return p.parse_args()

def retrieve_soft_knn_tokens(z_pred_unit, train_unit_bank, train_tokens_bank, image_ids, k=5, temp=0.05):
    """
    Perform soft-kNN retrieval to blend RAE spatial tokens.
    
    Args:
        z_pred_unit: Predicted normalized embeddings [B, 768]
        train_unit_bank: Training normalized target embeddings [N_train, 768]
        train_tokens_bank: Dictionary mapping image_id -> token tensor [768, 16, 16]
        image_ids: List of training image IDs corresponding to rows of train_unit_bank
        
    Returns:
        blended_tokens: Stacked tensor of shape [B, 768, 16, 16]
    """
    sim = torch.mm(z_pred_unit, train_unit_bank.t()) # [B, N_train]
    topk_sim, topk_idx = sim.topk(k, dim=-1)
    weights = F.softmax(topk_sim / temp, dim=-1) # [B, k]
    
    device = z_pred_unit.device
    retrieved = []
    for b in range(len(z_pred_unit)):
        # Retrieve tokens and combine
        accum = None
        for i in range(k):
            idx = topk_idx[b, i].item()
            img_id = image_ids[idx]
            tok = train_tokens_bank[img_id].to(device).float()
            w = weights[b, i]
            if accum is None:
                accum = tok * w
            else:
                accum += tok * w
        retrieved.append(accum)
        
    return torch.stack(retrieved)

def run_bootstrap_ci(values, num_iterations=1000, ci=95):
    if len(values) == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    boot_means = []
    n = len(values)
    values = np.array(values)
    for _ in range(num_iterations):
        boot_sample = np.random.choice(values, size=n, replace=True)
        boot_means.append(np.mean(boot_sample))
    boot_means = np.array(boot_means)
    lower_pct = (100 - ci) / 2.0
    upper_pct = 100 - lower_pct
    return {
        "mean": float(np.mean(values)),
        "ci_lower": float(np.percentile(boot_means, lower_pct)),
        "ci_upper": float(np.percentile(boot_means, upper_pct))
    }

def run_bootstrap_ci_multitask(matches_list, num_iterations=1000, ci=95):
    boot_means = []
    n = len(matches_list)
    for _ in range(num_iterations):
        indices = np.random.choice(n, size=n, replace=True)
        boot_matches = []
        for idx in indices:
            boot_matches.extend([m for m in matches_list[idx] if m is not None])
        if len(boot_matches) > 0:
            boot_means.append(np.mean(boot_matches))
        else:
            boot_means.append(0.0)
            
    boot_means = np.array(boot_means)
    all_matches = [m for sample in matches_list for m in sample if m is not None]
    mean_val = np.mean(all_matches) if all_matches else 0.0
    lower_pct = (100 - ci) / 2.0
    upper_pct = 100 - lower_pct
    return {
        "mean": float(mean_val),
        "ci_lower": float(np.percentile(boot_means, lower_pct)),
        "ci_upper": float(np.percentile(boot_means, upper_pct))
    }

def run_paired_bootstrap(real_values, baseline_values, num_iterations=1000, ci=95):
    if len(real_values) == 0 or len(real_values) != len(baseline_values):
        return {"mean_delta": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_value_empirical": 1.0}
    
    real_vals = np.array(real_values)
    base_vals = np.array(baseline_values)
    deltas = real_vals - base_vals
    
    boot_means = []
    n = len(deltas)
    for _ in range(num_iterations):
        boot_sample = np.random.choice(deltas, size=n, replace=True)
        boot_means.append(np.mean(boot_sample))
        
    boot_means = np.array(boot_means)
    lower_pct = (100 - ci) / 2.0
    upper_pct = 100 - lower_pct
    
    p_val = float(np.mean(boot_means <= 0))
    
    return {
        "mean_delta": float(np.mean(deltas)),
        "ci_lower": float(np.percentile(boot_means, lower_pct)),
        "ci_upper": float(np.percentile(boot_means, upper_pct)),
        "p_value_empirical": p_val
    }

def run_paired_bootstrap_multitask(real_matches, baseline_matches, num_iterations=1000, ci=95):
    n = len(real_matches)
    boot_means = []
    
    real_sample_means = []
    base_sample_means = []
    for r, b in zip(real_matches, baseline_matches):
        r_valid = [m for m in r if m is not None]
        b_valid = [m for m in b if m is not None]
        real_sample_means.append(np.mean(r_valid) if r_valid else 0.0)
        base_sample_means.append(np.mean(b_valid) if b_valid else 0.0)
        
    real_vals = np.array(real_sample_means)
    base_vals = np.array(base_sample_means)
    deltas = real_vals - base_vals
    
    for _ in range(num_iterations):
        boot_sample = np.random.choice(deltas, size=n, replace=True)
        boot_means.append(np.mean(boot_sample))
        
    boot_means = np.array(boot_means)
    lower_pct = (100 - ci) / 2.0
    upper_pct = 100 - lower_pct
    p_val = float(np.mean(boot_means <= 0))
    
    return {
        "mean_delta": float(np.mean(deltas)),
        "ci_lower": float(np.percentile(boot_means, lower_pct)),
        "ci_upper": float(np.percentile(boot_means, upper_pct)),
        "p_value_empirical": p_val
    }

def apply_rae_target_transform(global_vec, bank_metadata, target_key):
    device = global_vec.device
    is_1d = (global_vec.ndim == 1)
    if is_1d:
        global_vec = global_vec.unsqueeze(0)
        
    if "centered" in target_key or "whitened" in target_key:
        if "rae_center_mean" not in bank_metadata:
            raise KeyError("bank_metadata is missing 'rae_center_mean'. Did you run add_rae_transforms.py?")
        rae_center_mean = bank_metadata["rae_center_mean"].to(device)
        v_centered = global_vec - rae_center_mean
    else:
        v_centered = global_vec
        
    if "whitened" in target_key:
        if "rae_pca_components" not in bank_metadata or "rae_pca_eigenvalues" not in bank_metadata:
            raise KeyError("bank_metadata is missing whitening parameters.")
        rae_pca_components = bank_metadata["rae_pca_components"].to(device)
        rae_pca_eigenvalues = bank_metadata["rae_pca_eigenvalues"].to(device)
        whitening_eps = bank_metadata.get("whitening_eps", 1e-5)
        
        v_whitened = torch.matmul(v_centered, rae_pca_components)
        v_transformed = v_whitened / torch.sqrt(rae_pca_eigenvalues + whitening_eps)
    else:
        v_transformed = v_centered
        
    v_transformed_unit = F.normalize(v_transformed, dim=-1)
    
    if is_1d:
        v_transformed_unit = v_transformed_unit.squeeze(0)
        
    return v_transformed_unit

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    run_dir = Path(args.run_dir)
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")
        
    with open(config_path) as f:
        root_config = json.load(f)
        config = root_config.get("setup", {})
        
    print(f"Loading EEG Encoder from {run_dir}...")
    model_name = config.get("model", "cnn")
    saved_subject_to_id = config.get("subject_to_id", None)
    if saved_subject_to_id:
        num_subjects = len(saved_subject_to_id)
        subject_list = list(saved_subject_to_id.keys())
    else:
        num_subjects = len(config.get("subjects_loaded", [1]))
        subject_list = config.get("subjects_loaded", None)
        
    no_film = config.get("no_film", False)
    no_subject_heads = config.get("no_subject_heads", False)
    
    if saved_subject_to_id is None:
        num_subjects = 1
        no_film = True
        no_subject_heads = True
        
    # Build model architecture
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
                num_subjects=num_subjects,
                no_film=no_film,
                no_subject_heads=no_subject_heads,
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
                num_subjects=num_subjects,
                no_film=no_film,
                no_subject_heads=no_subject_heads,
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
        
    checkpoint = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    target_center_eval = checkpoint.get("target_center", None)
    if target_center_eval is not None:
        target_center_eval = target_center_eval.to(device)
        print("Detected target_center in checkpoint. Centering will be applied to predictions.")
    else:
        target_center_eval = None
        print("No target_center detected in checkpoint. Skipping centering.")
    
    # Load dataset
    common_pt = root_config.get("common_embeddings", config.get("common_embeddings", "data/processed/rae_embeddings/rae_dinov2_base_sub01_04_runs01_40.pt"))
    metadata_paths = root_config.get("metadata", config.get("metadata", ""))
    epochs_dir_paths = root_config.get("epochs_dir", config.get("epochs_dir", ""))
    
    print("Loading Validation Dataset...")
    dataset_config = SemanticPairConfig(
        common_embeddings_pt=common_pt,
        metadata_csv=metadata_paths,
        epochs_dir=epochs_dir_paths,
        window_mode="tight1s",
        target_mode="real",
        input_domain="zuna",
        target_space="rae_unit",
        target_key=args.target_key,
        vlm_attributes_json="data/processed/clip_embeddings/vlm_attributes.json",
        add_event_marker=root_config.get("add_event_marker", True),
        subject_list=subject_list
    )
    dataset = ZunaClipPairDataset(dataset_config)
    
    val_fraction = root_config.get("val_fraction", config.get("val_fraction", 0.15))
    seed = root_config.get("seed", config.get("seed", 13))
    train_indices, val_indices = split_indices(len(dataset), val_fraction=val_fraction, seed=seed)
    
    num_eval = min(args.num_samples, len(val_indices))
    eval_val_indices = val_indices[:num_eval]
    print(f"Loaded dataset. Train: {len(train_indices)}, Val: {len(val_indices)}. Evaluating: {num_eval}")
    
    # Load RAE latent bank table
    print("Loading RAE target embeddings bank...")
    table = torch.load(common_pt, map_location="cpu")
    image_id_to_rae_tokens = table["image_id_to_rae_tokens"]
    
    # Build training retrieval banks
    train_unit_list = []
    train_image_ids = []
    for idx in train_indices:
        row = dataset.metadata.iloc[idx]
        img_id = str(row["image_id"])
        target = dataset._get_targets(idx)
        train_unit_list.append(target)
        train_image_ids.append(img_id)
        
    train_unit_bank = torch.stack(train_unit_list).to(device)
    
    # Load RAE and CLIP backends
    rae_backend = RaeDecoderBackend(device=device, apply_patch=True)
    rae_backend.load()
    
    clip_backend = None
    probe_model = None
    if args.common_probe and Path(args.common_probe).exists():
        clip_backend = ClipNativeDecoderBackend(device=device)
        print(f"Loading pretrained probe model from {args.common_probe}...")
        probe_specs_path = Path(args.common_probe).parent / "task_specs.json"
        with open(probe_specs_path, "r") as f:
            active_task_specs = json.load(f)
            
        # RAE latents have dimension 768
        probe_model = CommonProbeModel(
            embedding_dim=768,
            task_specs=active_task_specs
        ).to(device)
        probe_model.load_state_dict(torch.load(args.common_probe, map_location=device))
        probe_model.eval()
        for p in probe_model.parameters():
            p.requires_grad = False
            
    # Extract predicted unit embeddings from val set
    print("Running EEG encoder inference...")
    val_eeg = []
    val_targets = []
    val_subjects = []
    val_image_ids = []
    val_probe_targets = []
    for idx in eval_val_indices:
        item = dataset[idx]
        val_eeg.append(item["eeg"])
        val_targets.append(item["target"])
        val_subjects.append(item["subject_id"])
        val_image_ids.append(item["image_id"])
        val_probe_targets.append(item["probe_targets"])
        
    val_eeg = torch.stack(val_eeg).to(device)
    val_targets = torch.stack(val_targets).to(device)
    val_subjects = torch.tensor(val_subjects, device=device).long()
    
    with torch.no_grad():
        kwargs = {"subject_id": val_subjects} if getattr(model, "subject_embed", None) is not None else {}
        shuffled_subjects = torch.roll(val_subjects, shifts=1, dims=0)
        shuffled_kwargs = {"subject_id": shuffled_subjects} if getattr(model, "subject_embed", None) is not None else {}
        
        pred_raw = model(val_eeg, **kwargs)
        if target_center_eval is not None:
            pred_unit = F.normalize(pred_raw - target_center_eval, dim=-1)
        else:
            pred_unit = F.normalize(pred_raw, dim=-1)
            
        shuffled_eeg = torch.roll(val_eeg, shifts=1, dims=0)
        shuff_raw = model(shuffled_eeg, **shuffled_kwargs)
        if target_center_eval is not None:
            shuff_unit = F.normalize(shuff_raw - target_center_eval, dim=-1)
        else:
            shuff_unit = F.normalize(shuff_raw, dim=-1)
            
        random_unit = F.normalize(torch.randn_like(pred_unit), dim=-1)

    # --- 1. RAE-native Full-bank Retrieval (Top-10 and MRR) ---
    print("Computing full-bank retrieval in RAE space...")
    all_target_list = []
    all_image_ids = list(table[args.target_key].keys())
    for img_id in all_image_ids:
        all_target_list.append(table[args.target_key][img_id])
    full_bank = F.normalize(torch.stack(all_target_list).to(device), dim=-1) # [N_bank, 768]
    
    fb_logits = pred_unit @ full_bank.T # [N_val, 15891]
    tgt_fb_logits = val_targets @ full_bank.T
    correct_idx = tgt_fb_logits.argmax(dim=-1)
    
    fb_sorted = fb_logits.argsort(dim=-1, descending=True)
    fb_rank = (fb_sorted == correct_idx[:, None]).nonzero(as_tuple=False)[:, 1].float()
    
    rae_fb_top1 = (fb_rank < 1).float().mean().item()
    rae_fb_top5 = (fb_rank < 5).float().mean().item()
    rae_fb_top10 = (fb_rank < 10).float().mean().item()
    rae_fb_mrr = (1.0 / (fb_rank + 1.0)).mean().item()

    N_bank = full_bank.size(0)
    fb_median_rank = torch.median(fb_rank + 1).item()
    
    # Calculate MRR chance
    # Sum of 1/i for i=1 to N
    mrr_chance = sum(1.0 / i for i in range(1, N_bank + 1)) / N_bank
    
    fb_random_top10_expected = 10.0 / N_bank
    fb_top10_enrichment = rae_fb_top10 / fb_random_top10_expected if fb_random_top10_expected > 0 else 0.0
    
    print(f"Full-Bank Size (N): {N_bank}")
    print(f"Full-Bank Top-1:  {rae_fb_top1:.5f} (Chance: {1/N_bank:.5f})")
    print(f"Full-Bank Top-5:  {rae_fb_top5:.5f} (Chance: {5/N_bank:.5f})")
    print(f"Full-Bank Top-10: {rae_fb_top10:.5f} (Chance: {fb_random_top10_expected:.5f}) | Enrichment: {fb_top10_enrichment:.2f}x")
    print(f"Full-Bank MRR:    {rae_fb_mrr:.5f} (Chance: {mrr_chance:.5f})")
    print(f"Full-Bank Median: {fb_median_rank:.1f} / {N_bank} ({(fb_median_rank/N_bank)*100:.2f}%)")

    # --- 2. Soft-kNN retrieval and decoding ---
    conditions = {
        "oracle": [image_id_to_rae_tokens[img_id].to(device).float() for img_id in val_image_ids],
        "real": retrieve_soft_knn_tokens(pred_unit, train_unit_bank, image_id_to_rae_tokens, train_image_ids, k=args.k, temp=args.temperature),
        "shuffled": retrieve_soft_knn_tokens(shuff_unit, train_unit_bank, image_id_to_rae_tokens, train_image_ids, k=args.k, temp=args.temperature),
        "random": retrieve_soft_knn_tokens(random_unit, train_unit_bank, image_id_to_rae_tokens, train_image_ids, k=args.k, temp=args.temperature),
    }
    
    # We evaluate RAE-native generated-to-target cosine, CLIP cosine and attribute matches
    rae_generation_cosines = {c: [] for c in conditions}
    clip_generation_cosines = {c: [] for c in conditions}
    attribute_matches = {c: [[] for _ in range(num_eval)] for c in conditions}
    
    # Grid variables
    grid_targets = []
    grid_oracle = []
    grid_real = []
    grid_shuffled = []
    grid_random = []
    
    for cond_name, tokens_list in conditions.items():
        print(f"\nDecoding images for condition: {cond_name}...")
        for b_start in tqdm(range(0, num_eval, args.batch_size), desc="Generating batches"):
            b_end = min(b_start + args.batch_size, num_eval)
            
            if isinstance(tokens_list, list):
                batch_tokens = torch.stack(tokens_list[b_start:b_end])
            else:
                batch_tokens = tokens_list[b_start:b_end]
                
            gen_images = rae_backend.generate_from_embeds(batch_tokens)
            
            # Extract RAE global embeddings from generated images to evaluate RAE-native cosine
            # (Use PIL images batch)
            gen_rae = rae_backend.extract_rae_latent(gen_images)
            gen_rae_unit = apply_rae_target_transform(gen_rae["global"].float(), table, args.target_key)
            
            # Compare with true target RAE unit
            for idx_in_batch in range(len(gen_images)):
                global_idx = b_start + idx_in_batch
                tgt_unit = val_targets[global_idx]  # [768]
                rae_cos = F.cosine_similarity(gen_rae_unit[idx_in_batch], tgt_unit, dim=0).item()
                rae_generation_cosines[cond_name].append(rae_cos)
                
            # Secondary CLIP & attribute probe evaluation
            if clip_backend is not None:
                gen_clip = clip_backend.extract_teacher_embeds(gen_images, normalize=True).float() # [B, 1024]
                
                # Retrieve true target CLIP embeddings
                target_clip_list = []
                for idx_in_batch in range(len(gen_images)):
                    global_idx = b_start + idx_in_batch
                    img_id = val_image_ids[global_idx]
                    stim_path = Path(args.stimuli_dir) / f"{img_id}.JPEG"
                    if not stim_path.exists():
                        stim_path = Path(args.stimuli_dir) / f"{img_id}.png"
                    if stim_path.exists():
                        target_clip_list.append(clip_backend.extract_teacher_embeds(Image.open(stim_path).convert("RGB").resize((256, 256)), normalize=True).squeeze(0))
                    else:
                        target_clip_list.append(torch.zeros(gen_clip.shape[-1], device=device))
                        
                target_clip = torch.stack(target_clip_list)
                clip_cos = F.cosine_similarity(gen_clip, target_clip, dim=-1).cpu().numpy()
                clip_generation_cosines[cond_name].extend(clip_cos.tolist())
                
                if probe_model is not None:
                    probe_logits_dict = probe_model(gen_rae_unit)
                    for idx_in_batch in range(len(gen_images)):
                        global_idx = b_start + idx_in_batch
                        gt_targets = val_probe_targets[global_idx]
                        
                        sample_matches = []
                        for task_name, logits in probe_logits_dict.items():
                            gt_val = gt_targets.get(task_name, IGNORE_INDEX)
                            if gt_val == IGNORE_INDEX:
                                continue
                            pred_val = logits[idx_in_batch].argmax().item()
                            sample_matches.append(float(pred_val == gt_val))
                            
                        attribute_matches[cond_name][global_idx] = sample_matches
                        
            # Save first 8 samples for visual grid
            if cond_name == "oracle":
                for idx_in_batch in range(len(gen_images)):
                    global_idx = b_start + idx_in_batch
                    if len(grid_targets) < 8:
                        # load target image
                        img_id = val_image_ids[global_idx]
                        stim_path = Path(args.stimuli_dir) / f"{img_id}.JPEG"
                        if not stim_path.exists():
                            stim_path = Path(args.stimuli_dir) / f"{img_id}.png"
                        if stim_path.exists():
                            t_img = Image.open(stim_path).convert("RGB").resize((256, 256))
                        else:
                            t_img = Image.new("RGB", (256, 256), (30, 30, 30))
                        grid_targets.append(TF.to_tensor(t_img))
                        grid_oracle.append(TF.to_tensor(gen_images[idx_in_batch]))
            elif cond_name == "real":
                for img in gen_images:
                    if len(grid_real) < 8:
                        grid_real.append(TF.to_tensor(img))
            elif cond_name == "shuffled":
                for img in gen_images:
                    if len(grid_shuffled) < 8:
                        grid_shuffled.append(TF.to_tensor(img))
            elif cond_name == "random":
                for img in gen_images:
                    if len(grid_random) < 8:
                        grid_random.append(TF.to_tensor(img))

    # Calculate statistics
    rae_results = {c: run_bootstrap_ci(rae_generation_cosines[c]) for c in conditions}
    rae_paired = {
        "real_vs_shuffled": run_paired_bootstrap(rae_generation_cosines["real"], rae_generation_cosines["shuffled"]),
        "real_vs_random": run_paired_bootstrap(rae_generation_cosines["real"], rae_generation_cosines["random"])
    }
    
    clip_results = {c: run_bootstrap_ci(clip_generation_cosines[c]) for c in conditions}
    attribute_results = {c: run_bootstrap_ci_multitask(attribute_matches[c]) for c in conditions}
    attribute_paired = {
        "real_vs_shuffled": run_paired_bootstrap_multitask(attribute_matches["real"], attribute_matches["shuffled"]),
        "real_vs_random": run_paired_bootstrap_multitask(attribute_matches["real"], attribute_matches["random"])
    }
    
    print("\n" + "="*80)
    print("RAE NATIVE GENERATION EVALUATION RESULTS")
    print("="*80)
    print(f"{'Condition':<15} | {'RAE Cosine similarity (Mean ± 95% CI)':<40}")
    print("-"*80)
    for c in ["oracle", "real", "shuffled", "random"]:
        mean = rae_results[c]["mean"]
        low = rae_results[c]["ci_lower"]
        high = rae_results[c]["ci_upper"]
        print(f"{c:<15} | {mean:.5f} ({low:.5f} to {high:.5f})")
    
    print("-"*80)
    print("PAIRED SIGNIFICANCE (Real vs Baseline Deltas)")
    print("-"*80)
    for name, p_data in rae_paired.items():
        print(f"{name:<15} | Mean Delta: {p_data['mean_delta']:+.5f} (CI: {p_data['ci_lower']:+.5f} to {p_data['ci_upper']:+.5f}) | p-value: {p_data['p_value_empirical']:.4f}")
    print("="*80)
    
    # Save metrics to JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = {
        "rae_native_full_bank_retrieval": {
            "top1": rae_fb_top1,
            "top5": rae_fb_top5,
            "top10": rae_fb_top10,
            "mrr": rae_fb_mrr,
            "median_rank": fb_median_rank,
            "n_bank": N_bank,
            "top10_chance": fb_random_top10_expected,
            "top10_enrichment": fb_top10_enrichment
        },
        "rae_native_generation_cosine": rae_results,
        "rae_native_paired_significance": rae_paired,
        "clip_space_generation_cosine": clip_results,
        "attribute_agreement": attribute_results,
        "attribute_paired_significance": attribute_paired
    }
    
    with open(output_dir / "generation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    # Construct visual grid (Columns: Target | Oracle | Real EEG kNN | Shuffled kNN | Random kNN)
    all_grid_tensors = []
    for i in range(len(grid_targets)):
        all_grid_tensors.append(grid_targets[i])
        all_grid_tensors.append(grid_oracle[i])
        all_grid_tensors.append(grid_real[i])
        all_grid_tensors.append(grid_shuffled[i])
        all_grid_tensors.append(grid_random[i])
        
    grid = make_grid(all_grid_tensors, nrow=5, padding=4)
    grid_pil = TF.to_pil_image(grid)
    grid_pil.save(output_dir / "generation_grid.png")
    
    print(f"Saved generation grid to {output_dir / 'generation_grid.png'}")
    print(f"Saved generation metrics to {output_dir / 'generation_metrics.json'}")

if __name__ == "__main__":
    main()

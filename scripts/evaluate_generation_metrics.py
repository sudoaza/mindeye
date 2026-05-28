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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindseye.generation.clip_native_backend import ClipNativeDecoderBackend
from mindseye.models.eeg_encoder import EEGClipEncoder, TemporalAttnEncoder, DualHeadTemporalAttnEncoder
from mindseye.datasets.semantic_pairs import ZunaClipPairDataset, SemanticPairConfig, split_indices
from mindseye.models.common_probe import CommonProbeModel, ATTRIBUTE_SCHEMAS, IGNORE_INDEX

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="Path to trained EEG model directory")
    p.add_argument("--num-samples", type=int, default=100, help="Number of validation samples to evaluate")
    p.add_argument("--batch-size", type=int, default=10, help="Batch size for unCLIP generation")
    p.add_argument("--k", type=int, default=5, help="k for soft kNN retrieval")
    p.add_argument("--temperature", type=float, default=0.05, help="Temperature for soft kNN softmax")
    p.add_argument("--common-probe", default="outputs/decode_probe_v2/common_probe.pt", help="Path to pretrained attribute probe model")
    p.add_argument("--metadata", default=None, help="Override metadata CSV path(s)")
    p.add_argument("--epochs-dir", default=None, help="Override epochs directory path(s)")
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

def run_bootstrap_ci(data_dict, num_iterations=1000, ci=95):
    """
    Run bootstrap resampling to compute mean and confidence intervals.
    data_dict: dict mapping condition -> numpy array of shape [num_samples]
    """
    results = {}
    lower_pct = (100 - ci) / 2.0
    upper_pct = 100 - lower_pct
    
    for cond, values in data_dict.items():
        if len(values) == 0:
            continue
        boot_means = []
        n = len(values)
        for _ in range(num_iterations):
            boot_sample = np.random.choice(values, size=n, replace=True)
            boot_means.append(np.mean(boot_sample))
            
        boot_means = np.array(boot_means)
        mean_val = np.mean(values)
        lower_val = np.percentile(boot_means, lower_pct)
        upper_val = np.percentile(boot_means, upper_pct)
        
        results[cond] = {
            "mean": mean_val,
            "ci_lower": lower_val,
            "ci_upper": upper_val,
        }
    return results

def run_bootstrap_ci_multitask(matches_dict, num_iterations=1000, ci=95):
    """
    Bootstrap attribute matches which might have variable number of valid attributes per sample.
    matches_dict: dict mapping condition -> list of lists of matches (True/False/None)
    """
    results = {}
    lower_pct = (100 - ci) / 2.0
    upper_pct = 100 - lower_pct
    
    for cond, samples_list in matches_dict.items():
        boot_means = []
        n = len(samples_list)
        for _ in range(num_iterations):
            # Resample indices
            indices = np.random.choice(n, size=n, replace=True)
            boot_matches = []
            for idx in indices:
                # Add all valid matches for this sample
                boot_matches.extend([m for m in samples_list[idx] if m is not None])
            if len(boot_matches) > 0:
                boot_means.append(np.mean(boot_matches))
            else:
                boot_means.append(0.0)
                
        boot_means = np.array(boot_means)
        
        # Original mean
        all_matches = []
        for sample in samples_list:
            all_matches.extend([m for m in sample if m is not None])
        mean_val = np.mean(all_matches) if all_matches else 0.0
        
        lower_val = np.percentile(boot_means, lower_pct)
        upper_val = np.percentile(boot_means, upper_pct)
        
        results[cond] = {
            "mean": mean_val,
            "ci_lower": lower_val,
            "ci_upper": upper_val,
        }
    return results

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

    # Resolve multiple subject paths from setup block if present, allowing CLI overrides
    metadata_paths = args.metadata if args.metadata is not None else root_config.get("metadata", "")
    epochs_dir_paths = args.epochs_dir if args.epochs_dir is not None else root_config.get("epochs_dir", "")

    if not metadata_paths:
        metadata_paths = config.get("metadata", "data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv")
    if not epochs_dir_paths:
        epochs_dir_paths = config.get("epochs_dir", "data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40")

    # Load subject_to_id from saved config (authoritative). Fall back to auto-detect.
    saved_subject_to_id = config.get("subject_to_id", None)
    if saved_subject_to_id:
        num_subjects = len(saved_subject_to_id)
        subject_list = list(saved_subject_to_id.keys())
        print(f"[SubjectMap] Loaded from config: {saved_subject_to_id}")
    else:
        # Legacy fallback: derive from subjects_loaded
        num_subjects = len(config.get("subjects_loaded", [1]))
        subject_list = config.get("subjects_loaded", None)
        print(f"[SubjectMap] No saved mapping, falling back to subjects_loaded len={num_subjects}")

    no_film = config.get("no_film", False)
    no_subject_heads = config.get("no_subject_heads", False)

    # Pre-adapter checkpoints (no subject_to_id saved) don't have adapter keys.
    # Force adapter-free mode so model is built without those params and loads cleanly.
    is_legacy_checkpoint = saved_subject_to_id is None
    if is_legacy_checkpoint:
        num_subjects = 1
        no_film = True
        no_subject_heads = True
        print("[SubjectMap] Legacy checkpoint (no subject_to_id): forcing num_subjects=1, no_film=True, no_subject_heads=True")

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
    
    checkpoint_path = run_dir / "best.pt"
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("Loading Validation Dataset...")
    dataset_config = SemanticPairConfig(
        common_embeddings_pt="data/processed/clip_embeddings/decode_common_embeddings.pt",
        metadata_csv=metadata_paths,
        epochs_dir=epochs_dir_paths,
        window_mode="tight1s",
        target_mode="real",
        input_domain="zuna",
        vlm_attributes_json="data/processed/clip_embeddings/vlm_attributes.json",
        add_event_marker=root_config.get("add_event_marker", True),
        subject_list=subject_list,
    )
    dataset = ZunaClipPairDataset(dataset_config)
    
    # Use split_indices to replicate training/validation split
    val_fraction = config.get("val_fraction", 0.15)
    seed = config.get("seed", 13)
    train_indices, val_indices = split_indices(len(dataset), val_fraction=val_fraction, seed=seed)
    
    print(f"Loaded dataset: {len(dataset)} items. Train: {len(train_indices)}, Val: {len(val_indices)}")
    
    # Limit val samples to evaluate
    num_eval = min(args.num_samples, len(val_indices))
    eval_val_indices = val_indices[:num_eval]
    print(f"Evaluating {num_eval} validation samples...")
    
    print("Constructing retrieval banks from training set...")
    train_decode_unit = []
    train_target_raw = []
    for idx in tqdm(train_indices, desc="Building retrieval banks"):
        item = dataset[idx]
        train_decode_unit.append(item["target"])
        train_target_raw.append(item.get("target_raw", torch.zeros_like(item["target"])))
        
    train_decode_unit = torch.stack(train_decode_unit).to(device)
    train_target_raw = torch.stack(train_target_raw).to(device)
    
    # Load CommonProbeModel
    probe_model = None
    if args.common_probe:
        print(f"Loading pretrained probe model from {args.common_probe}...")
        probe_specs_path = Path(args.common_probe).parent / "task_specs.json"
        if not probe_specs_path.exists():
            raise FileNotFoundError(f"Missing task_specs.json at {probe_specs_path}")
        with open(probe_specs_path, "r") as f:
            active_task_specs = json.load(f)
            
        probe_model = CommonProbeModel(
            embedding_dim=dataset.embedding_dim,
            task_specs=active_task_specs
        ).to(device)
        probe_model.load_state_dict(torch.load(args.common_probe, map_location=device))
        probe_model.eval()
        for p in probe_model.parameters():
            p.requires_grad = False
            
    print("Loading ClipNativeDecoderBackend (Stable unCLIP)...")
    backend = ClipNativeDecoderBackend(device=device)
    
    # Extract prediction embeddings for val set
    print("Computing EEG predictions...")
    val_eeg = []
    val_target_raw = []
    val_probe_targets = []
    val_subjects = []
    for idx in eval_val_indices:
        item = dataset[idx]
        val_eeg.append(item["eeg"])
        val_target_raw.append(item["target_raw"])
        val_probe_targets.append(item["probe_targets"])
        val_subjects.append(item["subject_id"])
        
    val_eeg = torch.stack(val_eeg).to(device)
    val_target_raw = torch.stack(val_target_raw).to(device)
    val_subjects = torch.tensor(val_subjects, device=device).long()
    
    with torch.no_grad():
        kwargs = {"subject_id": val_subjects} if getattr(model, "subject_embed", None) is not None else {}
        shuffled_subjects = torch.roll(val_subjects, shifts=1, dims=0)
        shuffled_kwargs = {"subject_id": shuffled_subjects} if getattr(model, "subject_embed", None) is not None else {}
        
        if config.get("dual_head", False):
            pred_unit, _ = model(val_eeg, return_norm=True, **kwargs)
            shuffled_eeg = torch.roll(val_eeg, shifts=1, dims=0)
            shuff_unit, _ = model(shuffled_eeg, return_norm=True, **shuffled_kwargs)
        else:
            pred_unit = model(val_eeg, **kwargs)
            shuffled_eeg = torch.roll(val_eeg, shifts=1, dims=0)
            shuff_unit = model(shuffled_eeg, **shuffled_kwargs)
            
        # Retrieve target_raw via soft kNN
        real_embeds = retrieve_soft_knn(pred_unit, train_decode_unit, train_target_raw, k=args.k, temp=args.temperature)
        shuffled_embeds = retrieve_soft_knn(shuff_unit, train_decode_unit, train_target_raw, k=args.k, temp=args.temperature)
        
        random_unit = torch.randn_like(pred_unit)
        random_embeds = retrieve_soft_knn(random_unit, train_decode_unit, train_target_raw, k=args.k, temp=args.temperature)
        
    conditions = {
        "oracle": val_target_raw,
        "real": real_embeds,
        "shuffled": shuffled_embeds,
        "random": random_embeds,
    }
    
    # We will generate and evaluate in batches
    cosine_sims = {c: [] for c in conditions}
    attribute_matches = {c: [[] for _ in range(num_eval)] for c in conditions}
    
    for cond_name, embeds in conditions.items():
        print(f"\nEvaluating condition: {cond_name}...")
        for b_start in tqdm(range(0, num_eval, args.batch_size), desc="Generating batches"):
            b_end = min(b_start + args.batch_size, num_eval)
            batch_embeds = embeds[b_start:b_end]
            
            # Generate images from embeddings
            gen_images = backend.generate_from_embeds(batch_embeds, num_inference_steps=20, watermark=False)
            
            # Extract CLIP embeddings from generated images
            gen_clip = backend.extract_teacher_embeds(gen_images, normalize=True) # [B, 1024]
            gen_clip = gen_clip.float()  # cast to float32 (pipeline outputs float16)
            
            # Compare with target_raw
            target_raw_batch = val_target_raw[b_start:b_end].to(device)
            target_raw_batch_norm = F.normalize(target_raw_batch.float(), dim=-1)
            
            cos = F.cosine_similarity(gen_clip, target_raw_batch_norm, dim=-1).cpu().numpy()
            cosine_sims[cond_name].extend(cos.tolist())
            
            # Predict attributes
            if probe_model is not None:
                # Probe expects visual embedding of shape [B, 1024]
                probe_logits_dict = probe_model(gen_clip)
                
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
                    
    # Clean up GPU memory
    del backend
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Bootstrapping
    print("\n=== Running Bootstrapping for 95% Confidence Intervals ===")
    
    cosine_results = run_bootstrap_ci({c: np.array(cosine_sims[c]) for c in conditions})
    attribute_results = run_bootstrap_ci_multitask(attribute_matches)
    
    # Print results table
    print("\n" + "="*80)
    print(f"QUANTITATIVE GENERATION EVALUATION RESULTS ({num_eval} samples)")
    print("="*80)
    print(f"{'Condition':<15} | {'CLIP Cosine similarity (Mean ± 95% CI)':<40} | {'Attribute Agreement (Mean ± 95% CI)':<40}")
    print("-"*100)
    
    for c in ["oracle", "real", "shuffled", "random"]:
        cos_mean = cosine_results[c]["mean"]
        cos_low = cosine_results[c]["ci_lower"]
        cos_high = cosine_results[c]["ci_upper"]
        
        attr_mean = attribute_results[c]["mean"]
        attr_low = attribute_results[c]["ci_lower"]
        attr_high = attribute_results[c]["ci_upper"]
        
        print(f"{c:<15} | {cos_mean:.5f} ({cos_low:.5f} to {cos_high:.5f}) | {attr_mean:.2%} ({attr_low:.2%} to {attr_high:.2%})")
        
    print("="*80)

    # --- Per-subject breakdown ---
    subject_ids_np = val_subjects.cpu().numpy() if hasattr(val_subjects, "cpu") else np.array(val_subjects)
    unique_subj_ids = sorted(set(subject_ids_np.tolist()))
    # Build reverse map: adapter id -> subject name
    id_to_subj = {v: k for k, v in (saved_subject_to_id or {}).items()}

    per_subject_results = {}
    if len(unique_subj_ids) > 1:
        print("\n" + "="*80)
        print("PER-SUBJECT BREAKDOWN")
        print("="*80)
        for sid in unique_subj_ids:
            mask = (subject_ids_np == sid)
            subj_name = id_to_subj.get(sid, f"sub-{sid}")
            for c in ["real", "shuffled"]:
                subj_cos = np.array(cosine_sims[c])[mask]
                if len(subj_cos) == 0:
                    continue
                subj_attr = [attribute_matches[c][i] for i in range(len(subject_ids_np)) if mask[i]]
                all_attr = [m for sample in subj_attr for m in sample if m is not None]
                attr_mean = np.mean(all_attr) if all_attr else 0.0
                print(f"  {subj_name} | {c:10s} | cosine={subj_cos.mean():.5f} | attr_agree={attr_mean:.2%} | n={mask.sum()}")
            per_subject_results[subj_name] = {
                c: {
                    "cosine_mean": float(np.array(cosine_sims[c])[mask].mean()) if mask.sum() > 0 else 0.0,
                    "n": int(mask.sum()),
                } for c in ["oracle", "real", "shuffled", "random"]
            }
        print("="*80)

    # Save results to a json file in the run directory
    results = {
        "cosine_similarity": cosine_results,
        "attribute_agreement": attribute_results,
        "per_subject": per_subject_results,
        "subject_to_id": saved_subject_to_id or {},
    }
    out_json_path = run_dir / "generation_evaluation_metrics.json"
    with open(out_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved quantitative generation metrics to {out_json_path}")

if __name__ == "__main__":
    main()

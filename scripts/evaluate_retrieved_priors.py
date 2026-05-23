#!/usr/bin/env python3
"""Evaluate retrieved visual priors from EEG predictions.
Computes CLIP cosine similarity, coarse VLM attribute agreement, and generates retrieval grids.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindseye.embeddings.faiss_index import FAISSIndex
from mindseye.datasets.semantic_pairs import SemanticPairConfig, ZunaClipPairDataset, split_indices

COARSE_ATTRIBUTES = [
    "is_animate",
    "human_visible",
    "face_visible",
    "animal_visible",
    "indoor_outdoor",
    "natural_artificial",
    "dominant_color",
    "soft_texture",
    "spiky_or_pointed",
    "furry",
]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained model checkpoint (.pt)",
    )
    p.add_argument(
        "--index-prefix",
        required=True,
        help="Prefix path of the FAISS index and metadata",
    )
    p.add_argument(
        "--common-embeddings",
        required=True,
        help="Path to common_embeddings.pt containing target image embeddings",
    )
    p.add_argument(
        "--vlm-attributes",
        required=True,
        help="Path to vlm_attributes_runs01_40.json",
    )
    p.add_argument(
        "--common-probe",
        default=None,
        help="Path to pretrained common_probe.pt checkpoint for attribute-constrained reranking",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save the evaluation results and retrieval grids",
    )
    p.add_argument(
        "--rerank-mode",
        type=str,
        default="cosine_attr_div",
        choices=["cosine", "attr", "cosine_attr", "cosine_attr_div"],
        help="Reranking mode to use (default: cosine_attr_div)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of nearest neighbors to retrieve (default: 10)",
    )
    p.add_argument(
        "--num-grid-examples",
        type=int,
        default=15,
        help="Number of validation trials to render in visual grid",
    )
    p.add_argument(
        "--stimuli-root",
        default="data/raw/nod/stimuli/ImageNet",
        help="Root directory of stimulus ImageNet images",
    )
    p.add_argument(
        "--device",
        default=None,
        help="cuda, cpu, or omitted for auto",
    )
    p.add_argument("--metadata", default=None, help="Override crop metadata CSV path")
    p.add_argument("--epochs-dir", default=None, help="Override crop NPZ epochs directory")
    return p.parse_args()


def _parse_val_runs(value: str) -> set[int]:
    return {int(x.strip()) for x in value.split(",") if x.strip()}


def _split_by_run(dataset, val_runs: set[int]) -> tuple[list[int], list[int]]:
    if "run" not in dataset.metadata.columns:
        raise ValueError("run split requires a 'run' column in metadata")
    train_idx: list[int] = []
    val_idx: list[int] = []
    for idx, run in enumerate(dataset.metadata["run"].astype(int).tolist()):
        (val_idx if run in val_runs else train_idx).append(idx)
    return train_idx, val_idx


def _resolve_image(path_str: str, stimuli_root: Path) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    stem = path.stem
    for ext in ("", ".JPEG", ".jpg", ".png", ".jpeg"):
        candidate = stimuli_root / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    candidate = Path.cwd() / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not resolve stimulus image: {path_str}")


def _fit_image(image, size: int):
    from PIL import Image
    image = image.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def _draw_label(draw, xy: tuple[int, int], text: str) -> None:
    draw.text(xy, text[:28], fill=(0, 0, 0))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    setup = checkpoint.get("setup", {})
    
    checkpoint_path = Path(args.checkpoint)
    config_json_path = checkpoint_path.parent / "config.json"
    train_config = {}
    if config_json_path.exists():
        with open(config_json_path, "r") as f:
            train_config = json.load(f)
            
    metadata_csv = args.metadata or train_config.get("metadata") or setup.get("metadata")
    epochs_dir = args.epochs_dir or train_config.get("epochs_dir") or setup.get("epochs_dir")
    
    if not metadata_csv or not epochs_dir:
        raise ValueError("Could not resolve dataset paths. Please supply --metadata and --epochs-dir overrides.")
        
    print(f"Loading validation dataset from metadata={metadata_csv}...")
    dataset = ZunaClipPairDataset(
        SemanticPairConfig(
            metadata_csv=metadata_csv,
            epochs_dir=epochs_dir,
            common_embeddings_pt=args.common_embeddings,
            epochs_dir_raw=train_config.get("epochs_dir_raw"),
            epochs_dir_resample=train_config.get("epochs_dir_resample"),
            vlm_attributes_json=args.vlm_attributes,
            input_domain=train_config.get("input_domain") or setup.get("input_domain", "zuna"),
            target_mode=train_config.get("target_mode") or setup.get("target_mode", "real"),
            window_mode=train_config.get("window_mode") or setup.get("window_mode", "crop"),
            target_space=train_config.get("target_space") or setup.get("target_space", "common"),
            add_event_marker=train_config.get("add_event_marker", setup.get("add_event_marker", False)),
            augment_eeg=False,
        )
    )
    
    # Resolve validation indices
    split_mode = train_config.get("split_mode") or setup.get("split_mode", "random")
    seed = train_config.get("seed") or setup.get("seed", 13)
    val_fraction = train_config.get("val_fraction") or setup.get("val_fraction", 0.15)
    val_runs_val = train_config.get("val_runs") or setup.get("val_runs")
    
    if split_mode == "run":
        if not val_runs_val:
            raise ValueError("run split requested but no validation runs were found in checkpoint or training config")
        val_runs = _parse_val_runs(val_runs_val) if isinstance(val_runs_val, str) else set(val_runs_val)
        _, val_idx = _split_by_run(dataset, val_runs)
    else:
        _, val_idx = split_indices(len(dataset), val_fraction=val_fraction, seed=seed)
        
    print(f"Recreated validation split: {len(val_idx)} validation trials.")
    
    # Reconstruct and load model
    model_type = setup.get("model") or train_config.get("model", "cnn")
    n_channels, n_times = dataset.eeg_shape
    
    print(f"Recreating encoder model architecture: '{model_type}'...")
    if model_type in {"spatial_temporal", "spatial_temporal_small"}:
        from mindseye.models.spatial_temporal_encoder import build_spatial_temporal_encoder
        preset = "small" if model_type == "spatial_temporal_small" else "medium"
        overrides = {}
        for key in ("hidden_dim", "n_layers", "n_heads", "dropout"):
            val = train_config.get(key)
            if val is not None:
                overrides[key] = val
        overrides["stem_dropout"] = train_config.get("stem_dropout1d", 0.15)
        overrides["spatial_mixing"] = not train_config.get("no_spatial_mixing", False)
        
        model = build_spatial_temporal_encoder(
            preset,
            n_channels=n_channels,
            embedding_dim=dataset.embedding_dim,
            ch_names=getattr(dataset, "ch_names", None),
            **overrides,
        ).to(device)
    elif model_type in {"temporal_attn", "temporal_attn_small"}:
        from mindseye.models.eeg_encoder import TemporalAttnEncoder
        hidden_dim = train_config.get("hidden_dim")
        n_layers = train_config.get("n_layers")
        n_heads = train_config.get("n_heads")
        dropout = train_config.get("dropout")
        if model_type == "temporal_attn_small":
            hidden_dim = hidden_dim or 128
            n_layers = n_layers or 2
            n_heads = n_heads or 4
            dropout = 0.35 if dropout is None else dropout
        else:
            hidden_dim = hidden_dim or 256
            n_layers = n_layers or 4
            n_heads = n_heads or 8
            dropout = 0.2 if dropout is None else dropout
        model = TemporalAttnEncoder(
            n_channels=n_channels,
            embedding_dim=dataset.embedding_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            stem_dropout1d=train_config.get("stem_dropout1d", 0.15),
        ).to(device)
    else:
        from mindseye.models.eeg_encoder import EEGClipEncoder
        hidden_dim = train_config.get("hidden_dim") or 256
        dropout = train_config.get("dropout")
        dropout = 0.2 if dropout is None else dropout
        model = EEGClipEncoder(
            n_channels=n_channels,
            n_times=n_times,
            embedding_dim=dataset.embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            stem_dropout1d=train_config.get("stem_dropout1d", 0.15),
        ).to(device)
        
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    probe_model = None
    if args.common_probe:
        print(f"Loading Common Probe from {args.common_probe}...")
        from mindseye.models.common_probe import CommonProbeModel, ATTRIBUTE_SCHEMAS
        probe_specs_path = Path(args.common_probe).parent / "task_specs.json"
        if not probe_specs_path.exists():
            raise FileNotFoundError(f"Active tasks specification not found at {probe_specs_path}")
        with open(probe_specs_path, "r") as f:
            task_specs = json.load(f)
        probe_model = CommonProbeModel(
            embedding_dim=dataset.embedding_dim,
            task_specs=task_specs,
        ).to(device)
        probe_model.load_state_dict(torch.load(args.common_probe, map_location=device))
        probe_model.eval()
        for param in probe_model.parameters():
            param.requires_grad = False
    
    # Load target center if available
    target_center = checkpoint.get("target_center")
    if target_center is not None:
        target_center = target_center.to(device).float()
    
    # Load FAISS index
    print(f"Loading FAISS index from {args.index_prefix}...")
    index = FAISSIndex.load(args.index_prefix)
    
    # Load common embeddings to perform direct target similarity calculations
    print(f"Loading target embeddings from {args.common_embeddings}...")
    emb_dict = torch.load(args.common_embeddings, map_location="cpu")
    if "image_id_to_common" in emb_dict:
        image_id_to_common = emb_dict["image_id_to_common"]
    elif "image_id_to_image" in emb_dict:
        image_id_to_common = emb_dict["image_id_to_image"]
    else:
        image_id_to_common = emb_dict
        
    # Load VLM attributes
    print(f"Loading VLM attributes from {args.vlm_attributes}...")
    with open(args.vlm_attributes, "r") as f:
        vlm_attributes = json.load(f)
        
    # Run predictions
    from torch.utils.data import DataLoader, Subset
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=64, shuffle=False)
    
    print("Computing predicted EEG embeddings...")
    preds = []
    ground_truth_ids = []
    predicted_attributes = []
    
    with torch.no_grad():
        for batch in val_loader:
            eeg = batch["eeg"].to(device).float()
            subject_id = batch.get("subject_id", None)
            if subject_id is not None:
                subject_id = subject_id.to(device)
            kwargs = {"subject_id": subject_id} if "spatial_temporal" in type(model).__name__.lower() or "spatialtemporal" in type(model).__name__.lower() else {}
            pred = model(eeg, **kwargs)
            if isinstance(pred, tuple):
                pred = pred[0]
            preds.append(pred.cpu())
            ground_truth_ids.extend(batch["image_id"])
            
            if probe_model is not None:
                logits_dict = probe_model(F.normalize(pred, dim=-1))
                batch_size = pred.shape[0]
                batch_attrs = [{} for _ in range(batch_size)]
                for attr in COARSE_ATTRIBUTES:
                    if attr in logits_dict:
                        preds_idx = logits_dict[attr].argmax(dim=-1).cpu().numpy()
                        for i in range(batch_size):
                            batch_attrs[i][attr] = ATTRIBUTE_SCHEMAS[attr][preds_idx[i]]
                predicted_attributes.extend(batch_attrs)
            
    preds = torch.cat(preds)
    
    # Reranking requires querying more candidates initially (e.g., top-200)
    coarse_k = 200 if probe_model is not None else args.top_k
    print(f"Querying FAISS index for top-{coarse_k} visual priors...")
    distances, retrieved_ids = index.search(preds, k=coarse_k)
    
    # Evaluate CLIP similarity and attribute agreement per trial
    clip_similarities = []
    attribute_matches = {attr: [] for attr in COARSE_ATTRIBUTES}
    attribute_matches["class_family"] = []
    
    results = []
    
    for i in range(len(val_idx)):
        gt_id = ground_truth_ids[i]
        gt_emb = image_id_to_common[gt_id].float()
        gt_emb_norm = gt_emb / torch.linalg.norm(gt_emb)
        gt_attrs = vlm_attributes.get(gt_id, {})
        gt_wordnet = gt_id.split("_")[0]
        
        candidates = []
        for rank_200 in range(coarse_k):
            ret_id = retrieved_ids[i][rank_200]
            ret_emb = image_id_to_common[ret_id].float()
            ret_emb_norm = ret_emb / torch.linalg.norm(ret_emb)
            sim_gt = float(torch.dot(gt_emb_norm, ret_emb_norm).item())
            ret_attrs = vlm_attributes.get(ret_id, {})
            
            # Feature agreement with PREDICTED attributes
            attr_agreements = []
            if predicted_attributes:
                for attr in COARSE_ATTRIBUTES:
                    pred_val = predicted_attributes[i].get(attr, "unclear")
                    ret_val = ret_attrs.get(attr, "unclear")
                    if pred_val != "unclear" and ret_val != "unclear":
                        attr_agreements.append(1.0 if pred_val == ret_val else 0.0)
            attr_score = sum(attr_agreements) / len(attr_agreements) if attr_agreements else 0.0
            
            candidates.append({
                "image_id": ret_id,
                "pred_cosine": float(distances[i][rank_200]),
                "sim_gt": sim_gt,
                "attr_score": attr_score,
                "ret_attrs": ret_attrs,
                "original_rank": rank_200,
            })
            
        # Semantic Montage Reranking
        montage_selected = []
        selected_classes = set()
        
        if probe_model is not None:
            while len(montage_selected) < args.top_k and candidates:
                best_score = -999.0
                best_idx = -1
                for c_idx, c in enumerate(candidates):
                    wordnet_class = c["image_id"].split("_")[0]
                    diversity_reward = 1.0 if wordnet_class not in selected_classes else 0.0
                    
                    if args.rerank_mode == "cosine":
                        score = c["pred_cosine"]
                    elif args.rerank_mode == "attr":
                        score = c["attr_score"]
                    elif args.rerank_mode == "cosine_attr":
                        score = 0.60 * c["pred_cosine"] + 0.25 * c["attr_score"]
                    elif args.rerank_mode == "cosine_attr_div":
                        score = 0.60 * c["pred_cosine"] + 0.25 * c["attr_score"] + 0.15 * diversity_reward
                    else:
                        score = c["pred_cosine"]
                        
                    if score > best_score:
                        best_score = score
                        best_idx = c_idx
                best_cand = candidates.pop(best_idx)
                selected_classes.add(best_cand["image_id"].split("_")[0])
                montage_selected.append(best_cand)
        else:
            montage_selected = candidates[:args.top_k]
            
        # Overwrite retrieved_ids and distances to reflect the reranked top-K so grids work
        retrieved_ids[i][:args.top_k] = [c["image_id"] for c in montage_selected]
        distances[i][:args.top_k] = [c["pred_cosine"] for c in montage_selected]
            
        trial_clip_sims = []
        trial_attr_matches = {attr: [] for attr in COARSE_ATTRIBUTES}
        trial_attr_matches["class_family"] = []
        
        retrieved_list = []
        for rank in range(args.top_k):
            c = montage_selected[rank]
            ret_id = c["image_id"]
            ret_attrs = c["ret_attrs"]
            
            trial_clip_sims.append(c["sim_gt"])
            
            # Compute Attribute Agreement vs Ground Truth
            for attr in COARSE_ATTRIBUTES:
                gt_val = gt_attrs.get(attr, "unclear")
                ret_val = ret_attrs.get(attr, "unclear")
                trial_attr_matches[attr].append(1.0 if gt_val == ret_val else 0.0)
            
            ret_wordnet = ret_id.split("_")[0]
            trial_attr_matches["class_family"].append(1.0 if gt_wordnet == ret_wordnet else 0.0)
            
            retrieved_list.append({
                "rank": rank + 1,
                "image_id": ret_id,
                "score": c.get("pred_cosine", 0.0),
                "clip_similarity": c["sim_gt"],
                "attributes": {attr: ret_attrs.get(attr, "unclear") for attr in COARSE_ATTRIBUTES}
            })
            
        clip_similarities.append(trial_clip_sims)
        for attr in COARSE_ATTRIBUTES:
            attribute_matches[attr].append(trial_attr_matches[attr])
        attribute_matches["class_family"].append(trial_attr_matches["class_family"])
            
        results.append({
            "trial_index": int(val_idx[i]),
            "ground_truth_image_id": gt_id,
            "ground_truth_attributes": {attr: gt_attrs.get(attr, "unclear") for attr in COARSE_ATTRIBUTES},
            "retrieved_results": retrieved_list
        })
        
    # Convert lists to numpy arrays for aggregate calculations
    clip_similarities = np.array(clip_similarities)  # [N, top_k]
    
    top1_sim = clip_similarities[:, 0].mean()
    top5_sim = clip_similarities[:, :5].mean(axis=1).mean()
    top10_sim = clip_similarities[:, :10].mean(axis=1).mean() if args.top_k >= 10 else 0.0
    
    attr_top1 = {}
    attr_top5 = {}
    attr_top10 = {}
    
    overall_top1_match = []
    overall_top5_match = []
    overall_top10_match = []
    
    ALL_EVAL_ATTRS = COARSE_ATTRIBUTES + ["class_family"]
    
    for attr in ALL_EVAL_ATTRS:
        matches = np.array(attribute_matches[attr])  # [N, top_k]
        
        attr_top1[attr] = matches[:, 0].mean()
        attr_top5[attr] = matches[:, :5].mean(axis=1).mean()
        if args.top_k >= 10:
            attr_top10[attr] = matches[:, :10].mean(axis=1).mean()
            overall_top10_match.append(matches[:, :10].mean(axis=1))
        
        overall_top1_match.append(matches[:, 0])
        overall_top5_match.append(matches[:, :5].mean(axis=1))
        
    mean_top1_attr = np.mean(overall_top1_match)
    mean_top5_attr = np.mean(overall_top5_match)
    mean_top10_attr = np.mean(overall_top10_match) if args.top_k >= 10 else 0.0
    
    summary_metrics = {
        "clip_similarity": {
            "top1": float(top1_sim),
            "top5": float(top5_sim),
            "top10": float(top10_sim)
        },
        "attribute_agreement": {
            "top1_mean": float(mean_top1_attr),
            "top5_mean": float(mean_top5_attr),
            "top10_mean": float(mean_top10_attr),
            "per_attribute_top1": {a: float(attr_top1[a]) for a in ALL_EVAL_ATTRS},
            "per_attribute_top5": {a: float(attr_top5[a]) for a in ALL_EVAL_ATTRS},
            "per_attribute_top10": {a: float(attr_top10.get(a, 0.0)) for a in ALL_EVAL_ATTRS}
        }
    }
    
    print("\n================ EVALUATION METRICS ================")
    print(f"Checkpoint: {args.checkpoint}")
    print("\nCLIP Cosine Similarity:")
    print(f"  Top-1:  {top1_sim:.4f}")
    print(f"  Top-5:  {top5_sim:.4f}")
    if args.top_k >= 10: print(f"  Top-10: {top10_sim:.4f}")
    print("\nMean Attribute Agreement:")
    print(f"  Top-1:  {mean_top1_attr:.4f}")
    print(f"  Top-5:  {mean_top5_attr:.4f}")
    if args.top_k >= 10: print(f"  Top-10: {mean_top10_attr:.4f}")
    print("\nPer-Attribute Top-1 Agreement:")
    for a in ALL_EVAL_ATTRS:
        print(f"  {a:<20}: {attr_top1[a]:.4f}")
    print("====================================================")
    
    # Save JSON report
    report = {
        "checkpoint": str(args.checkpoint),
        "metrics": summary_metrics,
        "trials": results
    }
    
    report_path = output_dir / "retrieval_evaluation.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Successfully saved evaluation report to: {report_path}")
    
    # Save conditioning object for downstream diffusion
    conditioning_priors = {
        "z_pred_common": preds,
        "predicted_attributes": predicted_attributes,
        "trials": []
    }
    for r in results:
        conditioning_priors["trials"].append({
            "trial_index": r["trial_index"],
            "ground_truth_image_id": r["ground_truth_image_id"],
            "top_k_priors": [item["image_id"] for item in r["retrieved_results"]],
            "prior_scores": [item["score"] for item in r["retrieved_results"]],
        })
    cond_path = output_dir / "conditioning_priors.pt"
    torch.save(conditioning_priors, cond_path)
    print(f"Successfully saved conditioning priors to: {cond_path}")
    
    # Generate visual retrieval grids
    print(f"Generating visual retrieval grids for first {args.num_grid_examples} validation trials...")
    try:
        from PIL import Image, ImageDraw
        stimuli_root = Path(args.stimuli_root)
        
        # Get target image paths from index or metadata
        if "image_path" in emb_dict:
            bank_image_paths = {emb_dict["image_id"][idx]: emb_dict["image_path"][idx] for idx in range(len(emb_dict["image_id"]))}
        else:
            bank_image_paths = {img_id: f"{img_id}.png" for img_id in image_id_to_common.keys()}
            
        thumb = 160
        label_h = 34
        margin = 12
        cols = 6  # 1 GT + 5 retrieved
        rows = min(args.num_grid_examples, len(val_idx))
        width = cols * thumb + (cols + 1) * margin
        height = rows * (thumb + label_h) + (rows + 1) * margin
        grid = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(grid)
        
        for r in range(rows):
            gt_id = ground_truth_ids[r]
            gt_path = bank_image_paths.get(gt_id, f"{gt_id}.png")
            gt_img_resolved = _resolve_image(gt_path, stimuli_root)
            
            # Draw GT image
            img_gt = _fit_image(Image.open(gt_img_resolved), thumb)
            grid.paste(img_gt, (margin, margin + r * (thumb + label_h + margin)))
            _draw_label(draw, (margin, margin + r * (thumb + label_h + margin) + thumb + 2), f"GT: {gt_id}")
            
            for c in range(1, cols):
                ret_id = retrieved_ids[r][c - 1]
                ret_path = bank_image_paths.get(ret_id, f"{ret_id}.png")
                ret_img_resolved = _resolve_image(ret_path, stimuli_root)
                
                img_ret = _fit_image(Image.open(ret_img_resolved), thumb)
                x = margin + c * (thumb + margin)
                y = margin + r * (thumb + label_h + margin)
                grid.paste(img_ret, (x, y))
                
                score = distances[r][c - 1]
                _draw_label(draw, (x, y + thumb + 2), f"Top-{c} ({score:.2f}) {ret_id}")
                
        grid_path = output_dir / "retrieval_grid.jpg"
        grid.save(grid_path, quality=92)
        print(f"Successfully saved visual retrieval grid to: {grid_path}")
        
    except Exception as e:
        print(f"Warning: Failed to generate visual grid: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

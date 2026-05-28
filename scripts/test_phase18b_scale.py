#!/usr/bin/env python3
"""Phase 18B scale correction test.

Tests three inference-only rescaling strategies for EEG-predicted codes
and measures expander scale sensitivity using oracle targets.

Rescale variants (no retraining):
    A. global scalar:   pred *= (target_std / pred_std)
    B. per-sample norm: pred *= target_norm_mean / pred.norm(dim=-1, keepdim=True)
    C. per-site norm:   pred[:, :, h, w] *= target_site_norm_mean[h,w] / pred[:, :, h, w].norm(dim=1)

Expander scale sensitivity (oracle targets scaled by factor):
    target_code * {0.25, 0.5, 1.0, 2.0, 5.0}

Reports for each variant:
    - pred_std / target_std  (scale ratio)
    - expanded_token_cosine  (EEG-decoded vs oracle tokens)
    - EEG - shuffled gap     (information signal)

Usage (on pod):
    python scripts/test_phase18b_scale.py \\
        --run-dir outputs/20260528_164952_zuna_real_phase18b_eeg_to_rae_3x3 \\
        --bottleneck-checkpoint outputs/rae_bottleneck/spatial_768x3x3/best.pt \\
        --rae-bank data/processed/rae_embeddings/rae_dinov2_base_sub01_04_runs01_40.pt \\
        --codes-bank data/processed/rae_embeddings/rae_bottleneck_codes_3x3.pt \\
        --num-samples 200 \\
        --output-dir outputs/phase18b_scale_test \\
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindseye.generation.rae_backend import RaeDecoderBackend
from mindseye.models.rae_token_bottleneck import build_bottleneck
from mindseye.datasets.semantic_pairs import ZunaClipPairDataset, SemanticPairConfig, split_indices


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--bottleneck-checkpoint", required=True)
    p.add_argument("--rae-bank", required=True)
    p.add_argument("--codes-bank", required=True)
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--output-dir", default="outputs/phase18b_scale_test")
    p.add_argument("--device", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(values: list[float], n: int = 1000, ci: int = 95) -> dict:
    if not values:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    arr = np.array(values)
    boot = [np.mean(np.random.choice(arr, len(arr))) for _ in range(n)]
    lo = (100 - ci) / 2
    return {"mean": float(arr.mean()),
            "ci_lower": float(np.percentile(boot, lo)),
            "ci_upper": float(np.percentile(boot, 100 - lo))}


def expanded_token_cosine_batch(codes_flat: torch.Tensor,
                                code_shape: tuple,
                                bottleneck,
                                image_ids: list[str],
                                image_id_to_rae_tokens: dict,
                                device: torch.device) -> list[float]:
    """Expand flat codes [N, D] → [N, 768, 16, 16] and compute cosine vs oracle tokens."""
    cosines = []
    c, h, w = code_shape
    for i, img_id in enumerate(image_ids):
        if img_id not in image_id_to_rae_tokens:
            continue
        code_sp = codes_flat[i:i+1].reshape(1, c, h, w).to(device)
        with torch.no_grad():
            expanded = bottleneck.expand(code_sp)   # [1, 768, 16, 16]
        oracle = image_id_to_rae_tokens[img_id].to(device).float().unsqueeze(0)  # [1, 768, 16, 16]
        cos = F.cosine_similarity(
            expanded.reshape(1, 768, -1),
            oracle.reshape(1, 768, -1), dim=-1
        ).mean().item()
        cosines.append(float(cos))
    return cosines


def load_eeg_model(run_dir: Path, embedding_dim: int, n_channels: int, device: torch.device):
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    root_cfg = cfg.get("setup", cfg)

    model_type = root_cfg.get("model") or "temporal_attn_small"
    hidden_dim = int(root_cfg.get("hidden_dim") or 128)
    n_layers   = int(root_cfg.get("n_layers") or 2)
    n_heads    = int(root_cfg.get("n_heads") or 4)
    dropout    = float(root_cfg.get("dropout") or 0.35)
    num_subj   = int(root_cfg.get("num_subjects") or 1)

    from mindseye.models.eeg_encoder import TemporalAttnEncoder
    model = TemporalAttnEncoder(
        n_channels=n_channels,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        num_subjects=num_subj,
        normalize_output=False,
    )
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = Path(args.run_dir)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    root_cfg = cfg.get("setup", cfg)

    ds_config = SemanticPairConfig(
        metadata_csv=root_cfg.get("metadata_csv") or cfg.get("metadata"),
        epochs_dir=root_cfg.get("epochs_dir") or cfg.get("epochs_dir"),
        common_embeddings_pt=root_cfg.get("common_embeddings_pt") or cfg.get("common_embeddings"),
        target_space=root_cfg.get("target_space") or "rae_code",
        target_key=root_cfg.get("target_key") or "image_id_to_rae_code",
        window_mode=root_cfg.get("window_mode") or "tight1s",
        # add_event_marker: default False — training did not use --add-event-marker
        add_event_marker=bool(root_cfg.get("add_event_marker") or cfg.get("add_event_marker") or False),
        augment_eeg=False,
        subject_list=list(root_cfg.get("subjects_loaded") or cfg.get("subjects_loaded") or []),
    )
    dataset = ZunaClipPairDataset(ds_config)
    code_shape = getattr(dataset, "_rae_code_shape", (768, 3, 3))
    print(f"[Dataset] n={len(dataset)}, code_shape={code_shape}")

    _, val_indices = split_indices(len(dataset),
                                   val_fraction=float(root_cfg.get("val_fraction") or 0.15),
                                   seed=int(root_cfg.get("seed") or 13))
    num_eval = min(args.num_samples, len(val_indices))
    eval_indices = val_indices[:num_eval]

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    n_channels, _ = dataset.eeg_shape
    model = load_eeg_model(run_dir, dataset.embedding_dim, n_channels, device)

    bn_ckpt = torch.load(args.bottleneck_checkpoint, map_location="cpu")
    bn_arch = bn_ckpt["arch"]
    bottleneck = build_bottleneck(bn_arch).to(device)
    bottleneck.load_state_dict(bn_ckpt["state_dict"])
    bottleneck.eval()
    for p in bottleneck.parameters():
        p.requires_grad = False
    print(f"[Bottleneck] {bn_arch}")

    # ------------------------------------------------------------------
    # Banks
    # ------------------------------------------------------------------
    print("Loading RAE token bank...")
    rae_bank = torch.load(args.rae_bank, map_location="cpu")
    image_id_to_rae_tokens = rae_bank["image_id_to_rae_tokens"]

    print("Loading codes bank...")
    codes_data = torch.load(args.codes_bank, map_location="cpu")
    image_id_to_rae_code = codes_data["image_id_to_rae_code"]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    print("Running EEG inference...")
    val_eeg, val_targets, val_subjects, val_image_ids = [], [], [], []
    for idx in eval_indices:
        item = dataset[idx]
        val_eeg.append(item["eeg"])
        val_targets.append(item["target"])
        val_subjects.append(item["subject_id"])
        val_image_ids.append(str(item["image_id"]))

    val_eeg_t      = torch.stack(val_eeg).to(device)
    val_targets_t  = torch.stack(val_targets).to(device)
    val_subj_t     = torch.tensor(val_subjects, device=device).long()

    with torch.no_grad():
        kwargs = {"subject_id": val_subj_t} if getattr(model, "subject_embed", None) is not None else {}
        pred_codes = model(val_eeg_t, **kwargs)        # [N, 6912]

        shuf_eeg   = torch.roll(val_eeg_t, shifts=1, dims=0)
        shuf_subj  = torch.roll(val_subj_t, shifts=1, dims=0)
        shuf_kw    = {"subject_id": shuf_subj} if getattr(model, "subject_embed", None) is not None else {}
        shuf_codes = model(shuf_eeg, **shuf_kw)

    pred_cpu  = pred_codes.cpu().float()
    tgt_cpu   = val_targets_t.cpu().float()
    shuf_cpu  = shuf_codes.cpu().float()

    # Global stats needed for variant A and B
    pred_std        = float(pred_cpu.std())
    target_std      = float(tgt_cpu.std())
    target_norm_mean = float(tgt_cpu.norm(dim=-1).mean())
    scalar_A = target_std / pred_std
    print(f"\n[Baseline stats]")
    print(f"  pred_std={pred_std:.4f}  target_std={target_std:.4f}  scalar_A={scalar_A:.4f}")
    print(f"  target_norm_mean={target_norm_mean:.2f}")

    # Per-site target norm mean [H, W] for variant C
    c, h, w = code_shape
    tgt_s = tgt_cpu.reshape(-1, c, h, w)
    target_site_norm = tgt_s.norm(dim=1).mean(dim=0)  # [H, W]

    # ------------------------------------------------------------------
    # Define variants
    # ------------------------------------------------------------------
    def rescale_A(pred):
        return pred * scalar_A

    def rescale_B(pred):
        pred_norm = pred.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return pred * (target_norm_mean / pred_norm)

    def rescale_C(pred):
        pred_sp = pred.reshape(-1, c, h, w)
        pred_site_norm = pred_sp.norm(dim=1)          # [N, H, W]
        scale = target_site_norm / (pred_site_norm + 1e-6)  # [N, H, W]
        pred_scaled = pred_sp * scale.unsqueeze(1)    # broadcast over C
        return pred_scaled.reshape(-1, c * h * w)

    variants = {
        "unscaled":     pred_cpu,
        "A_global":     rescale_A(pred_cpu),
        "B_per_sample": rescale_B(pred_cpu),
        "C_per_site":   rescale_C(pred_cpu),
    }
    shuf_variants = {
        "unscaled":     shuf_cpu,
        "A_global":     rescale_A(shuf_cpu),
        "B_per_sample": rescale_B(shuf_cpu),
        "C_per_site":   rescale_C(shuf_cpu),
    }

    # ------------------------------------------------------------------
    # Evaluate each variant
    # ------------------------------------------------------------------
    results = {}
    for name, pred_v in variants.items():
        print(f"\n--- Variant: {name} ---")
        # Distribution stats
        p_std  = float(pred_v.std())
        p_norm = float(pred_v.norm(dim=-1).mean())
        scale_ratio = p_std / target_std
        print(f"  pred_std={p_std:.4f}  scale_ratio={scale_ratio:.3f}x  pred_norm={p_norm:.2f}")

        # Spatial cosine (code level)
        pred_sv = pred_v.reshape(-1, c, h, w)
        tgt_sv  = tgt_cpu.reshape(-1, c, h, w)
        sp_cos  = float(F.cosine_similarity(pred_sv, tgt_sv, dim=1).mean())
        print(f"  spatial_cosine={sp_cos:.4f}")

        # Expanded token cosine (primary post-expand metric)
        print(f"  Computing expanded token cosines...")
        eeg_cos_vals  = expanded_token_cosine_batch(pred_v, code_shape, bottleneck, val_image_ids, image_id_to_rae_tokens, device)
        shuf_cos_vals = expanded_token_cosine_batch(shuf_variants[name], code_shape, bottleneck, val_image_ids, image_id_to_rae_tokens, device)

        eeg_ci  = bootstrap_ci(eeg_cos_vals)
        shuf_ci = bootstrap_ci(shuf_cos_vals)
        gap     = eeg_ci["mean"] - shuf_ci["mean"]

        print(f"  expanded_token_cosine_eeg      = {eeg_ci['mean']:.4f} [{eeg_ci['ci_lower']:.4f}, {eeg_ci['ci_upper']:.4f}]")
        print(f"  expanded_token_cosine_shuffled = {shuf_ci['mean']:.4f} [{shuf_ci['ci_lower']:.4f}, {shuf_ci['ci_upper']:.4f}]")
        print(f"  EEG - shuffled gap             = {gap:+.4f}")

        results[name] = {
            "pred_std": p_std,
            "scale_ratio": scale_ratio,
            "pred_norm": p_norm,
            "spatial_cosine": sp_cos,
            "expanded_token_cosine_eeg": eeg_ci,
            "expanded_token_cosine_shuffled": shuf_ci,
            "eeg_shuffled_gap": gap,
        }

    # ------------------------------------------------------------------
    # Expander scale-sensitivity test (oracle target × factor)
    # ------------------------------------------------------------------
    print("\n\n=== Expander Scale Sensitivity (oracle target codes) ===")
    scale_factors = [0.25, 0.5, 1.0, 2.0, 5.0]
    scale_sensitivity = {}

    # Use first N val samples that have both tokens and codes
    test_ids = [img_id for img_id in val_image_ids
                if img_id in image_id_to_rae_tokens and img_id in image_id_to_rae_code][:100]
    print(f"Using {len(test_ids)} samples with oracle codes + tokens")

    for factor in scale_factors:
        print(f"\n  factor={factor}:")
        cos_vals = []
        for img_id in tqdm(test_ids, leave=False, desc=f"factor={factor}"):
            oracle_code = image_id_to_rae_code[img_id].float()  # [768, 3, 3] or [6912]
            if oracle_code.ndim == 1:
                oracle_code = oracle_code.reshape(*code_shape)
            scaled_code = (oracle_code * factor).unsqueeze(0).to(device)  # [1, 768, 3, 3]

            with torch.no_grad():
                expanded = bottleneck.expand(scaled_code)  # [1, 768, 16, 16]

            oracle_tokens = image_id_to_rae_tokens[img_id].to(device).float().unsqueeze(0)
            cos = F.cosine_similarity(
                expanded.reshape(1, 768, -1),
                oracle_tokens.reshape(1, 768, -1), dim=-1
            ).mean().item()
            cos_vals.append(float(cos))

        ci = bootstrap_ci(cos_vals)
        scale_sensitivity[str(factor)] = ci
        print(f"    expanded_token_cosine = {ci['mean']:.4f} [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

    # ------------------------------------------------------------------
    # Save and summarize
    # ------------------------------------------------------------------
    all_results = {
        "num_samples": num_eval,
        "code_shape": list(code_shape),
        "baseline_scalar_A": scalar_A,
        "target_std": target_std,
        "pred_std_unscaled": pred_std,
        "rescale_variants": results,
        "expander_scale_sensitivity": scale_sensitivity,
    }
    out_path = out_dir / "scale_test_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n\nSaved → {out_path}")

    # Summary table
    print("\n\n=== SUMMARY ===")
    print(f"{'Variant':<18} {'scale_ratio':>12} {'spatial_cos':>12} {'ex_tok_eeg':>12} {'ex_tok_shuf':>12} {'gap':>8}")
    print("-" * 76)
    for name, r in results.items():
        print(f"{name:<18} {r['scale_ratio']:>12.3f}x {r['spatial_cosine']:>12.4f} "
              f"{r['expanded_token_cosine_eeg']['mean']:>12.4f} "
              f"{r['expanded_token_cosine_shuffled']['mean']:>12.4f} "
              f"{r['eeg_shuffled_gap']:>+8.4f}")
    print("\n=== SCALE SENSITIVITY (oracle) ===")
    print(f"{'factor':<10} {'expanded_token_cosine':>22}")
    print("-" * 34)
    for factor_s, ci in scale_sensitivity.items():
        print(f"{factor_s:<10} {ci['mean']:>22.4f}")

    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()

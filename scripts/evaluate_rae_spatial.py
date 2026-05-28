#!/usr/bin/env python3
"""Phase 18B evaluation: EEG → spatial code → expander → RAE decoder → image.

Pipeline stages:
    1. EEG encoder → predicted code  [B, 6912] (flattened [768, 3, 3])
    2. Reshape to [B, 768, 3, 3]
    3. Frozen _SpatialPoolBottleneck.expand() → [B, 768, 16, 16]
    4. Frozen RaeDecoderBackend.decode() → generated image

Quantitative metrics (all RAE-native; CLIP is secondary reference only):
    - target_code_mean/std/norm      — target code distribution
    - pred_code_mean/std/norm        — EEG predicted code distribution
    - spatial_cosine (pred↔target)   — primary metric
    - expanded_token_cosine          — after bottleneck expansion
    - collapsed_channels_pct         — per-channel std ratio diagnostic
    - full_bank_mrr / top-1/5/10     — secondary retrieval in flattened code space
    - oracle/bottleneck/eeg/shuffled/random image cosines (CLIP reference)

Visual grid: target | oracle (img→RAE→decode) | bottleneck oracle (img→RAE→code→expand→decode) | EEG | shuffled EEG

Usage (on pod):
    python scripts/evaluate_rae_spatial.py \\
        --run-dir outputs/runs/<phase18b_slug> \\
        --bottleneck-checkpoint outputs/rae_bottleneck/spatial_768x3x3/best.pt \\
        --rae-bank data/processed/rae_embeddings/rae_dinov2_base_sub01_04_runs01_40.pt \\
        --codes-bank data/processed/rae_embeddings/rae_bottleneck_codes_3x3.pt \\
        --stimuli-dir data/raw/nod/stimuli/ImageNet \\
        --num-samples 200 \\
        --output-dir outputs/phase18b_rae_spatial_eval \\
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
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindseye.generation.rae_backend import RaeDecoderBackend
from mindseye.models.eeg_encoder import TemporalAttnEncoder, EEGClipEncoder, DualHeadTemporalAttnEncoder
from mindseye.models.rae_token_bottleneck import build_bottleneck
from mindseye.datasets.semantic_pairs import ZunaClipPairDataset, SemanticPairConfig, split_indices


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="Path to trained EEG encoder run directory")
    p.add_argument("--bottleneck-checkpoint", required=True,
                   help="Path to trained bottleneck checkpoint (e.g. outputs/rae_bottleneck/spatial_768x3x3/best.pt)")
    p.add_argument("--rae-bank", required=True,
                   help="Path to full RAE latent bank (image_id_to_rae_tokens) .pt file")
    p.add_argument("--codes-bank", required=True,
                   help="Path to extracted bottleneck codes .pt file (image_id_to_rae_code)")
    p.add_argument("--stimuli-dir", default="data/raw/nod/stimuli/ImageNet",
                   help="Directory of ImageNet stimuli images")
    p.add_argument("--num-samples", type=int, default=200,
                   help="Number of validation samples to evaluate")
    p.add_argument("--output-dir", default="outputs/phase18b_rae_spatial_eval",
                   help="Directory to save results")
    p.add_argument("--device", default=None, help="cuda / cpu / auto")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image_tensor(img_id: str, stimuli_dir: Path, size: int = 224) -> torch.Tensor | None:
    """Load an ImageNet stimulus as a normalized [3, H, H] float tensor."""
    for ext in (".JPEG", ".jpg", ".jpeg", ".png"):
        p = stimuli_dir / f"{img_id}{ext}"
        if p.exists():
            img = Image.open(p).convert("RGB").resize((size, size))
            t = TF.to_tensor(img)  # [3, H, W] in [0,1]
            return t
    return None


def pil_grid(tensors: list[torch.Tensor], nrow: int) -> Image.Image:
    """Build a PIL image grid from a list of [3, H, W] tensors (values 0-1)."""
    grid = make_grid(torch.stack(tensors), nrow=nrow, padding=2)
    return TF.to_pil_image(grid.clamp(0, 1))


def run_bootstrap_ci(values: list[float], num_iterations: int = 1000, ci: int = 95) -> dict:
    if not values:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    arr = np.array(values)
    boot = [np.mean(np.random.choice(arr, len(arr))) for _ in range(num_iterations)]
    lo = (100 - ci) / 2
    return {
        "mean": float(arr.mean()),
        "ci_lower": float(np.percentile(boot, lo)),
        "ci_upper": float(np.percentile(boot, 100 - lo)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load EEG model
    # ------------------------------------------------------------------
    run_dir = Path(args.run_dir)
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {run_dir}")
    with open(config_path) as f:
        cfg = json.load(f)
    root_cfg = cfg.get("setup", cfg)

    # Resolve dataset config from saved training config
    metadata_csv = root_cfg.get("metadata_csv") or cfg.get("metadata")
    epochs_dir = root_cfg.get("epochs_dir") or cfg.get("epochs_dir")
    common_embeddings_pt = root_cfg.get("common_embeddings_pt") or cfg.get("common_embeddings")
    target_space = root_cfg.get("target_space") or cfg.get("target_space", "rae_code")
    target_key = root_cfg.get("target_key") or cfg.get("target_key", "image_id_to_rae_code")
    window_mode = root_cfg.get("window_mode") or cfg.get("window_mode", "tight1s")
    add_event_marker = bool(root_cfg.get("add_event_marker") or cfg.get("add_event_marker", True))
    val_fraction = float(root_cfg.get("val_fraction") or cfg.get("val_fraction", 0.15))
    seed = int(root_cfg.get("seed") or cfg.get("seed", 13))
    subject_list = list(root_cfg.get("subjects_loaded") or cfg.get("subjects_loaded") or [])

    ds_config = SemanticPairConfig(
        metadata_csv=metadata_csv,
        epochs_dir=epochs_dir,
        common_embeddings_pt=common_embeddings_pt,
        target_space=target_space,
        target_key=target_key,
        window_mode=window_mode,
        add_event_marker=add_event_marker,
        augment_eeg=False,
        subject_list=subject_list or None,
    )
    dataset = ZunaClipPairDataset(ds_config)
    code_shape = getattr(dataset, "_rae_code_shape", None)  # e.g. (768, 3, 3)
    print(f"[Dataset] n={len(dataset)}, embedding_dim={dataset.embedding_dim}, code_shape={code_shape}")

    _, val_indices = split_indices(len(dataset), val_fraction=val_fraction, seed=seed)
    num_eval = min(args.num_samples, len(val_indices))
    eval_indices = val_indices[:num_eval]
    print(f"Evaluating {num_eval} val samples")

    # Rebuild model
    n_channels, n_times = dataset.eeg_shape
    model_type = root_cfg.get("model") or cfg.get("model", "temporal_attn_small")
    embedding_dim = dataset.embedding_dim  # = C*H*W for rae_code
    hidden_dim = int(root_cfg.get("hidden_dim") or cfg.get("hidden_dim") or 128)
    n_layers = int(root_cfg.get("n_layers") or cfg.get("n_layers") or 2)
    n_heads = int(root_cfg.get("n_heads") or cfg.get("n_heads") or 4)
    dropout = float(root_cfg.get("dropout") or cfg.get("dropout") or 0.35)
    num_subjects = int(root_cfg.get("num_subjects") or cfg.get("num_subjects") or 1)

    if "temporal_attn" in model_type:
        model = TemporalAttnEncoder(
            n_channels=n_channels,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            num_subjects=num_subjects,
            normalize_output=False,  # raw code prediction, no L2 normalization
        ).to(device)
    else:
        model = EEGClipEncoder(
            n_channels=n_channels,
            n_times=n_times,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        ).to(device)

    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[Model] Loaded EEG encoder from {run_dir / 'best.pt'}")

    # ------------------------------------------------------------------
    # Load frozen bottleneck (expander only)
    # ------------------------------------------------------------------
    bn_ckpt = torch.load(args.bottleneck_checkpoint, map_location="cpu")
    bn_arch = bn_ckpt["arch"]
    bottleneck = build_bottleneck(bn_arch)
    bottleneck.load_state_dict(bn_ckpt["state_dict"])
    bottleneck = bottleneck.to(device)
    bottleneck.eval()
    for p_param in bottleneck.parameters():
        p_param.requires_grad = False
    print(f"[Bottleneck] Loaded {bn_arch} from {args.bottleneck_checkpoint}")

    # ------------------------------------------------------------------
    # Load RAE bank and codes bank
    # ------------------------------------------------------------------
    print("Loading RAE token bank...")
    rae_bank = torch.load(args.rae_bank, map_location="cpu")
    image_id_to_rae_tokens = rae_bank["image_id_to_rae_tokens"]

    print("Loading bottleneck codes bank...")
    codes_bank = torch.load(args.codes_bank, map_location="cpu")
    image_id_to_rae_code = codes_bank["image_id_to_rae_code"]
    stored_code_shape = tuple(codes_bank.get("code_shape", [768, 3, 3]))
    print(f"  Code bank: {len(image_id_to_rae_code)} images, shape {stored_code_shape}")

    # ------------------------------------------------------------------
    # Load frozen RAE decoder
    # ------------------------------------------------------------------
    print("Loading RAE decoder backend...")
    rae_backend = RaeDecoderBackend(device=device, apply_patch=True)
    rae_backend.load()

    stimuli_dir = Path(args.stimuli_dir)

    # ------------------------------------------------------------------
    # Inference: EEG → predicted code
    # ------------------------------------------------------------------
    print("Running EEG encoder inference...")
    val_eeg = []
    val_targets = []       # flattened raw codes [N, 6912]
    val_subjects = []
    val_image_ids = []

    for idx in eval_indices:
        item = dataset[idx]
        val_eeg.append(item["eeg"])
        val_targets.append(item["target"])
        val_subjects.append(item["subject_id"])
        val_image_ids.append(str(item["image_id"]))

    val_eeg_t = torch.stack(val_eeg).to(device)
    val_targets_t = torch.stack(val_targets).to(device)
    val_subjects_t = torch.tensor(val_subjects, device=device).long()

    with torch.no_grad():
        kwargs = {"subject_id": val_subjects_t} if getattr(model, "subject_embed", None) is not None else {}
        pred_codes = model(val_eeg_t, **kwargs)   # [N, 6912]
        shuffled_eeg = torch.roll(val_eeg_t, shifts=1, dims=0)
        shuf_subjects = torch.roll(val_subjects_t, shifts=1, dims=0)
        shuf_kwargs = {"subject_id": shuf_subjects} if getattr(model, "subject_embed", None) is not None else {}
        shuff_codes = model(shuffled_eeg, **shuf_kwargs)
        random_codes = torch.randn_like(pred_codes)

    print(f"Predicted code shape: {pred_codes.shape}")

    # ------------------------------------------------------------------
    # Metric 1: Code distribution diagnostics
    # ------------------------------------------------------------------
    pred_np = pred_codes.cpu().float()
    tgt_np = val_targets_t.cpu().float()

    diag = {
        "target_code_mean": float(tgt_np.mean().item()),
        "target_code_std": float(tgt_np.std().item()),
        "target_code_norm": float(tgt_np.norm(dim=-1).mean().item()),
        "pred_code_mean": float(pred_np.mean().item()),
        "pred_code_std": float(pred_np.std().item()),
        "pred_code_norm": float(pred_np.norm(dim=-1).mean().item()),
    }
    print("\n[Distribution Diagnostics]")
    for k, v in diag.items():
        print(f"  {k}: {v:.4f}")

    # Per-position spatial cosine
    if code_shape is not None and len(code_shape) == 3:
        c, h, w = code_shape
        pred_s = pred_np.reshape(-1, c, h, w)
        tgt_s = tgt_np.reshape(-1, c, h, w)
        pos_cos_per = F.cosine_similarity(pred_s, tgt_s, dim=1)  # [N, H, W]
        spatial_cos = float(pos_cos_per.mean().item())

        # Per-channel collapse
        eps = 1e-6
        pred_ch_std = pred_s.std(dim=[0, 2, 3])  # [C]
        tgt_ch_std = tgt_s.std(dim=[0, 2, 3])
        ratio = pred_ch_std / (tgt_ch_std + eps)
        collapsed_pct = float((ratio < 0.2).float().mean().item()) * 100.0
    else:
        spatial_cos = float(F.cosine_similarity(pred_np, tgt_np, dim=-1).mean().item())
        collapsed_pct = 0.0

    diag["spatial_cosine"] = spatial_cos
    diag["collapsed_channels_pct"] = collapsed_pct
    print(f"  spatial_cosine (primary): {spatial_cos:.4f}")
    print(f"  collapsed_channels_pct:   {collapsed_pct:.1f}%")

    # ------------------------------------------------------------------
    # Metric 2: Full-bank retrieval (secondary, normalized code space)
    # ------------------------------------------------------------------
    print("\nBuilding full code bank for retrieval...")
    all_ids = sorted(image_id_to_rae_code.keys())
    full_code_bank = torch.stack([
        image_id_to_rae_code[img_id].float().reshape(-1) for img_id in all_ids
    ])  # [N_bank, 6912]
    N_bank = full_code_bank.shape[0]

    pred_norm_fb = F.normalize(pred_np, dim=-1)
    tgt_norm_fb = F.normalize(tgt_np, dim=-1)
    fb_norm = F.normalize(full_code_bank, dim=-1)

    fb_logits = pred_norm_fb @ fb_norm.T           # [N_val, N_bank]
    tgt_fb_logits = tgt_norm_fb @ fb_norm.T
    correct_idx = tgt_fb_logits.argmax(dim=-1)
    fb_sorted = fb_logits.argsort(dim=-1, descending=True)
    fb_rank = (fb_sorted == correct_idx[:, None]).nonzero(as_tuple=False)[:, 1].float()

    retrieval = {
        "full_bank_n": N_bank,
        "full_bank_top1": float((fb_rank < 1).float().mean().item()),
        "full_bank_top5": float((fb_rank < 5).float().mean().item()),
        "full_bank_top10": float((fb_rank < 10).float().mean().item()),
        "full_bank_mrr": float((1.0 / (fb_rank + 1.0)).mean().item()),
        "full_bank_random_top10_expected": 10.0 / N_bank,
        "full_bank_median_rank": float(torch.median(fb_rank + 1).item()),
    }
    print("[Full-Bank Retrieval (secondary)]")
    for k, v in retrieval.items():
        print(f"  {k}: {v:.5f}" if isinstance(v, float) else f"  {k}: {v}")

    # ------------------------------------------------------------------
    # Metric 3: Expanded token cosine + visual reconstruction
    # ------------------------------------------------------------------
    print("\nRunning expand + decode for visual grid...")
    conditions = {
        "eeg":      pred_codes,
        "shuffled": shuff_codes,
        "random":   random_codes,
    }

    # Also compute oracle expanded token cosine (image→code→expand→decode)
    expanded_token_cosines: dict[str, list[float]] = {c: [] for c in [*conditions, "bottleneck_oracle"]}

    grid_target, grid_oracle, grid_bn_oracle, grid_eeg, grid_shuffled, grid_random = [], [], [], [], [], []

    MAX_GRID = min(20, num_eval)

    with torch.no_grad():
        for i, img_id in enumerate(tqdm(val_image_ids, desc="Decoding")):
            # ---- Target image ----
            if i < MAX_GRID:
                t_img = load_image_tensor(img_id, stimuli_dir)
                grid_target.append(t_img if t_img is not None else torch.zeros(3, 224, 224))

            # ---- Oracle RAE (full tokens → decode) ----
            if img_id in image_id_to_rae_tokens:
                tokens = image_id_to_rae_tokens[img_id].to(device).float().unsqueeze(0)
                oracle_img = rae_backend.decode(tokens)  # PIL Image
                if i < MAX_GRID:
                    grid_oracle.append(TF.to_tensor(oracle_img.resize((224, 224))))

                # Bottleneck oracle: image → code → expand
                if img_id in image_id_to_rae_code:
                    bn_code = image_id_to_rae_code[img_id].to(device).float()
                    if code_shape is not None:
                        bn_code_sp = bn_code.reshape(1, *code_shape)
                    else:
                        bn_code_sp = bn_code.unsqueeze(0)
                    expanded_bn = bottleneck.expand(bn_code_sp)            # [1, 768, 16, 16]
                    bn_oracle_img = rae_backend.decode(expanded_bn)        # PIL Image
                    if i < MAX_GRID:
                        grid_bn_oracle.append(TF.to_tensor(bn_oracle_img.resize((224, 224))))
                    # Expanded token cosine: bn_oracle vs oracle tokens
                    bn_cos = F.cosine_similarity(
                        expanded_bn.reshape(1, 768, -1),
                        tokens.reshape(1, 768, -1), dim=-1
                    ).mean().item()
                    expanded_token_cosines["bottleneck_oracle"].append(float(bn_cos))
            else:
                if i < MAX_GRID:
                    grid_oracle.append(torch.zeros(3, 224, 224))
                    grid_bn_oracle.append(torch.zeros(3, 224, 224))

            # ---- EEG reconstructions ----
            for cond_name, code_bank in conditions.items():
                code_flat = code_bank[i:i+1]   # [1, 6912]
                if code_shape is not None:
                    code_sp = code_flat.reshape(1, *code_shape)
                else:
                    code_sp = code_flat.unsqueeze(0)

                expanded = bottleneck.expand(code_sp.to(device))  # [1, 768, 16, 16]
                gen_img = rae_backend.decode(expanded)

                # Expanded token cosine vs oracle tokens
                if img_id in image_id_to_rae_tokens:
                    tok_ref = image_id_to_rae_tokens[img_id].to(device).float().unsqueeze(0)
                    ex_cos = F.cosine_similarity(
                        expanded.reshape(1, 768, -1),
                        tok_ref.reshape(1, 768, -1), dim=-1
                    ).mean().item()
                    expanded_token_cosines[cond_name].append(float(ex_cos))

                if i < MAX_GRID:
                    gen_t = TF.to_tensor(gen_img.resize((224, 224)))
                    if cond_name == "eeg":
                        grid_eeg.append(gen_t)
                    elif cond_name == "shuffled":
                        grid_shuffled.append(gen_t)
                    elif cond_name == "random":
                        grid_random.append(gen_t)

    # Expanded token cosine summary
    et_summary = {}
    for cond_name, vals in expanded_token_cosines.items():
        if vals:
            et_summary[f"expanded_token_cosine_{cond_name}"] = run_bootstrap_ci(vals)
    print("\n[Expanded Token Cosines]")
    for k, v in et_summary.items():
        print(f"  {k}: mean={v['mean']:.4f} [{v['ci_lower']:.4f}, {v['ci_upper']:.4f}]")

    # ------------------------------------------------------------------
    # Visual grid
    # ------------------------------------------------------------------
    if grid_target:
        nrow = MAX_GRID
        all_rows = []
        for row_imgs in [grid_target, grid_oracle, grid_bn_oracle, grid_eeg, grid_shuffled, grid_random]:
            if len(row_imgs) == nrow:
                all_rows.extend(row_imgs)
        if all_rows:
            grid_img = pil_grid(all_rows, nrow=nrow)
            grid_path = out_dir / "generation_grid.png"
            grid_img.save(grid_path)
            print(f"\nSaved visual grid ({nrow} samples × 6 rows) → {grid_path}")
            print("  Rows: target | oracle | bn_oracle | eeg | shuffled | random")

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    metrics = {
        "num_samples": num_eval,
        "code_shape": list(stored_code_shape),
        "bottleneck_arch": bn_arch,
        "distribution_diagnostics": diag,
        "full_bank_retrieval": retrieval,
        "expanded_token_cosines": et_summary,
    }
    metrics_path = out_dir / "generation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics → {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

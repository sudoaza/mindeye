#!/usr/bin/env python3
"""Probe sanity check: evaluate the frozen CommonProbeModel on z_common and z_pred.

Proves whether:
  1. probe(z_common)   ≈ pretraining accuracy (validates the probe itself)
  2. probe(z_pred_real) > probe(z_pred_shuffled/random) (validates probe loss is working)

Usage:
    python scripts/eval_probe_sanity.py \
        --common-probe outputs/common_probe/common_probe.pt \
        --common-embeddings data/processed/clip_embeddings/common_embeddings.pt \
        --checkpoint outputs/baseline_matrix/<run_dir>/best.pt \
        --metadata data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv \
        --epochs-dir data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40 \
        --val-runs 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--common-probe", required=True, help="Path to common_probe.pt")
    p.add_argument("--common-embeddings", required=True, help="Path to common_embeddings.pt")
    p.add_argument("--checkpoint", default=None,
                   help="Path to best.pt from a trained EEG run (optional, enables z_pred evaluation)")
    p.add_argument("--metadata", required=True, help="Comma-separated metadata CSV paths")
    p.add_argument("--epochs-dir", required=True, help="Comma-separated epoch dir paths")
    p.add_argument("--val-runs", default="8", help="Validation run numbers (held-out split)")
    p.add_argument("--window-mode", default="tight1s")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default=None)
    p.add_argument("--vlm-attributes", default=None, help="Path to vlm_attributes_runs*.json")
    return p.parse_args()


def eval_probe_on_embeddings(probe_model, embeds, probe_targets_t, active_tasks, device):
    """Run the frozen probe over a tensor of embeddings and return per-task accuracy."""
    import torch
    import torch.nn.functional as F
    from mindseye.models.common_probe import IGNORE_INDEX

    probe_model.eval()
    accs = {}
    with torch.inference_mode():
        batch_size = 256
        all_logits = {task: [] for task in active_tasks}
        for i in range(0, embeds.shape[0], batch_size):
            batch = F.normalize(embeds[i:i + batch_size].to(device), dim=-1)
            logits = probe_model(batch)
            for task in active_tasks:
                all_logits[task].append(logits[task].cpu())

        for task in active_tasks:
            logits_t = torch.cat(all_logits[task])
            targets_t = probe_targets_t[task]
            mask = targets_t != IGNORE_INDEX
            if mask.sum() == 0:
                accs[task] = None
            else:
                preds = logits_t.argmax(dim=-1)
                accs[task] = float((preds[mask] == targets_t[mask]).float().mean().item())

    return accs


def main() -> None:
    args = parse_args()
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset

    from mindseye.datasets.semantic_pairs import SemanticPairConfig, ZunaClipPairDataset
    from mindseye.models.common_probe import CommonProbeModel, IGNORE_INDEX

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # ── Load probe ───────────────────────────────────────────────────────────
    probe_dir = Path(args.common_probe).parent
    task_specs_path = probe_dir / "task_specs.json"
    with open(task_specs_path) as f:
        task_specs = json.load(f)
    active_tasks = list(task_specs.keys())

    # Read embedding dim from common embeddings
    table = torch.load(args.common_embeddings, map_location="cpu")
    first_emb = next(iter(table["image_id_to_common"].values()))
    embedding_dim = first_emb.shape[-1]

    probe_model = CommonProbeModel(embedding_dim=embedding_dim, task_specs=task_specs).to(device)
    probe_model.load_state_dict(torch.load(args.common_probe, map_location=device))
    probe_model.eval()
    print(f"Loaded probe with {len(active_tasks)} tasks: {active_tasks}")

    # ── Build dataset for val split ──────────────────────────────────────────
    # Derive add_event_marker from the checkpoint's eeg_shape channel count.
    # The key may be None in old checkpoints; channel count is always reliable:
    # base tight1s = 62 channels; +1 with event marker = 63.
    ckpt_setup = {}
    add_event_marker = False
    subjects_loaded = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        _ckpt_peek = torch.load(checkpoint_path, map_location="cpu")
        ckpt_setup = _ckpt_peek.get("setup", {})
        del _ckpt_peek
        n_ch = ckpt_setup.get("eeg_shape", [62])[0]
        add_event_marker = n_ch > 62  # 63 = marker added, 62 = base channels
        print(f"Inferred add_event_marker={add_event_marker} from eeg_shape n_channels={n_ch}")
        
        config_json_path = checkpoint_path.parent / "config.json"
        if config_json_path.exists():
            with open(config_json_path, "r") as f:
                train_config = json.load(f)
                subjects_loaded = train_config.get("subjects_loaded") or ckpt_setup.get("subjects_loaded")

    if args.vlm_attributes is None and args.common_probe:
        candidate = Path(args.common_probe).parent / "vlm_attributes_runs01_40.json"
        if candidate.exists():
            args.vlm_attributes = str(candidate)
            print(f"[Dataset] Auto-detected vlm_attributes at {args.vlm_attributes}")

    config = SemanticPairConfig(
        metadata_csv=args.metadata,
        epochs_dir=args.epochs_dir,
        common_embeddings_pt=args.common_embeddings,
        vlm_attributes_json=args.vlm_attributes,
        window_mode=args.window_mode,
        target_mode="real",
        add_event_marker=add_event_marker,
        augment_eeg=False,
        subject_list=subjects_loaded,
    )
    dataset = ZunaClipPairDataset(config)

    val_runs = {int(x) for x in args.val_runs.split(",") if x.strip()}
    if "run" not in dataset.metadata.columns:
        raise ValueError("Metadata must have a 'run' column for run-based split")
    val_idx = [i for i, r in enumerate(dataset.metadata["run"].astype(int)) if r in val_runs]
    print(f"Val split: {len(val_idx)} samples from runs {sorted(val_runs)}")

    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False)

    # ── Collect z_common ground truth and probe targets ───────────────────────
    z_common_list = []
    probe_targets_raw = {task: [] for task in active_tasks}

    for batch in val_loader:
        target = batch["target_common"] if "target_common" in batch else batch["target"]
        z_common_list.append(target.cpu())
        if "probe_targets" in batch:
            for task in active_tasks:
                probe_targets_raw[task].append(batch["probe_targets"][task].cpu())

    z_common = torch.cat(z_common_list)
    probe_targets_t = {task: torch.cat(probe_targets_raw[task]) for task in active_tasks}

    n_valid = {task: int((probe_targets_t[task] != IGNORE_INDEX).sum().item()) for task in active_tasks}
    print(f"\nVal probe label coverage: { {t: n for t, n in n_valid.items()} }")

    # ── Evaluate probe on z_common (ground truth) ────────────────────────────
    print("\n" + "=" * 60)
    print("  probe(z_common) — should match pretraining accuracy")
    print("=" * 60)
    accs_zcommon = eval_probe_on_embeddings(probe_model, z_common, probe_targets_t, active_tasks, device)
    for task, acc in accs_zcommon.items():
        print(f"  {task:<30s}  {f'{acc:.1%}' if acc is not None else 'N/A (no labels)'}")

    # ── Evaluate probe on z_pred from EEG checkpoint (if provided) ───────────
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        setup = ckpt.get("setup", {})
        input_domain = setup.get("input_domain", "zuna")
        target_mode = setup.get("target_mode", "real")
        print(f"\nLoaded checkpoint: input_domain={input_domain}, target_mode={target_mode}")

        # Rebuild encoder from checkpoint
        model_name = setup.get("model", "temporal_attn_small")
        n_channels, n_times = setup["eeg_shape"]
        emb_dim = setup["embedding_dim"]
        target_center = ckpt.get("target_center", None)
        if target_center is not None:
            target_center = target_center.to(device)

        num_subjects = len(subjects_loaded) if subjects_loaded else 1
        if model_name in {"temporal_attn", "temporal_attn_small"}:
            from mindseye.models.eeg_encoder import TemporalAttnEncoder
            model = TemporalAttnEncoder(
                n_channels=n_channels,
                embedding_dim=emb_dim,
                hidden_dim=setup.get("hidden_dim", 128),
                n_layers=setup.get("n_layers", 2),
                n_heads=setup.get("n_heads", 4),
                dropout=setup.get("dropout", 0.35),
                stem_dropout1d=setup.get("stem_dropout1d", 0.15),
                num_subjects=num_subjects,
            ).to(device)
        else:
            from mindseye.models.eeg_encoder import EEGClipEncoder
            model = EEGClipEncoder(
                n_channels=n_channels,
                n_times=n_times,
                embedding_dim=emb_dim,
            ).to(device)

        model.load_state_dict(ckpt["model_state"])
        model.eval()

        z_pred_list = []
        with torch.inference_mode():
            for batch in val_loader:
                eeg = batch["eeg"].to(device).float()
                subject_id = batch.get("subject_id", None)
                if subject_id is not None:
                    subject_id = subject_id.to(device)
                kwargs = {"subject_id": subject_id} if getattr(model, "subject_embed", None) is not None else {}
                pred = model(eeg, **kwargs)
                if isinstance(pred, tuple):
                    pred = pred[0]
                z_pred_list.append(pred.cpu())
        z_pred = torch.cat(z_pred_list)

        # Diagnostics: geometry of z_pred vs z_common
        import torch.nn.functional as F
        z_pred_norm = F.normalize(z_pred, dim=-1)
        z_common_norm = F.normalize(z_common, dim=-1)
        diag_cosine = (z_pred_norm * z_common_norm).sum(dim=-1)
        print(f"\nz_pred geometry:")
        print(f"  norm(z_pred)       mean={z_pred.norm(dim=-1).mean():.4f}  std={z_pred.norm(dim=-1).std():.4f}")
        print(f"  norm(z_common)     mean={z_common.norm(dim=-1).mean():.4f}")
        print(f"  cosine(z_pred, z_common) mean={diag_cosine.mean():.4f}  std={diag_cosine.std():.4f}  min={diag_cosine.min():.4f}  max={diag_cosine.max():.4f}")

        # Probe logit distribution on z_pred (first task) — diagnose silent failures
        with torch.inference_mode():
            sample_logits = probe_model(F.normalize(z_pred[:32].to(device), dim=-1))
        first_task = active_tasks[0]
        sl = sample_logits[first_task].cpu()
        print(f"\nProbe logit distribution ({first_task}, 32 samples):")
        print(f"  logit mean={sl.mean():.4f}  std={sl.std():.4f}  argmax distribution={sl.argmax(dim=-1).tolist()}")

        print(f"\n" + "=" * 60)
        print(f"  probe(z_pred_{target_mode}) — should be > 0 after training converges")
        print("=" * 60)
        accs_pred = eval_probe_on_embeddings(probe_model, z_pred, probe_targets_t, active_tasks, device)
        for task in active_tasks:
            gt_acc = accs_zcommon.get(task)
            pred_acc = accs_pred.get(task)
            gt_str = f"{gt_acc:.1%}" if gt_acc is not None else "N/A"
            pred_str = f"{pred_acc:.1%}" if pred_acc is not None else "N/A"
            delta = f"{pred_acc - gt_acc:+.1%}" if (gt_acc is not None and pred_acc is not None) else ""
            print(f"  {task:<30s}  z_common={gt_str}  z_pred={pred_str}  {delta}")

    print("\n✅ Sanity check complete.")


if __name__ == "__main__":
    main()

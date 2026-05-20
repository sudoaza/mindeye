#!/usr/bin/env python3
"""Create visual EEG→CLIP nearest-neighbor reconstruction grids.

This is a lightweight Phase-5 bridge: it does not run diffusion.  It loads a
trained EEG→CLIP checkpoint, predicts CLIP vectors for validation EEG crops,
and retrieves the nearest stimulus images in CLIP space so the current semantic
encoder can be inspected visually.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", default="data/processed/semantic_epochs/zuna_real_sub01_runs01_05/all_runs_metadata.csv")
    p.add_argument("--epochs-dir", default="data/processed/semantic_epochs/zuna_real_sub01_runs01_05")
    p.add_argument("--clip-embeddings", default="data/processed/clip_embeddings/sub01_runs01_05_clip_vit_base_patch32.pt")
    p.add_argument("--checkpoint", default="outputs/eeg_clip_baseline_sub01_runs01_05_ep005/best.pt")
    p.add_argument("--output-dir", default="outputs/retrieval_grids")
    p.add_argument("--stimuli-root", default="data/raw/nod/stimuli/ImageNet")
    p.add_argument("--seed", type=int, default=13, help="Split seed used during training")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--split-mode", choices=("random", "run"), default=None, help="Override split mode saved in checkpoint")
    p.add_argument("--val-runs", default=None, help="Override comma-separated validation runs saved in checkpoint")
    p.add_argument("--device", default=None, help="cuda, cpu, or omitted for auto")
    p.add_argument("--num-examples", type=int, default=12)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--thumb-size", type=int, default=160)
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
    if not train_idx or not val_idx:
        raise ValueError(f"Invalid run split: train={len(train_idx)} val={len(val_idx)} for val_runs={sorted(val_runs)}")
    return train_idx, val_idx


def _resolve_image(path_str: str, stimuli_root: Path) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    candidate = stimuli_root / path.name
    if candidate.exists():
        return candidate
    # Some tables may store paths relative to the repo root.
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
    # Keep labels short so Telegram thumbnails remain readable.
    draw.text(xy, text[:28], fill=(0, 0, 0))


def main() -> None:
    args = parse_args()

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
    from PIL import Image, ImageDraw

    from mindseye.datasets.semantic_pairs import SemanticPairConfig, ZunaClipPairDataset, split_indices
    from mindseye.models.eeg_encoder import EEGClipEncoder

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ZunaClipPairDataset(
        SemanticPairConfig(
            metadata_csv=args.metadata,
            epochs_dir=args.epochs_dir,
            clip_embeddings_pt=args.clip_embeddings,
        )
    )
    n_channels, n_times = dataset.eeg_shape
    model = EEGClipEncoder(n_channels=n_channels, n_times=n_times, embedding_dim=dataset.embedding_dim).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    setup = checkpoint.get("setup", {})
    split_mode = args.split_mode or setup.get("split_mode", "random")
    val_runs = args.val_runs or (",".join(str(x) for x in setup.get("val_runs") or []) if setup.get("val_runs") else None)
    if split_mode == "run":
        if not val_runs:
            raise ValueError("run split requested but no validation runs were supplied or saved in checkpoint")
        _, val_idx = _split_by_run(dataset, _parse_val_runs(val_runs))
    else:
        _, val_idx = split_indices(len(dataset), val_fraction=args.val_fraction, seed=args.seed)
    val_idx = val_idx[: args.num_examples]
    loader = DataLoader(Subset(dataset, val_idx), batch_size=args.num_examples, shuffle=False)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    target_center = checkpoint.get("target_center")
    if target_center is not None:
        target_center = target_center.cpu()

    batch = next(iter(loader))
    eeg = batch["eeg"].to(device).float()
    with torch.inference_mode():
        subject_id = batch.get("subject_id", None)
        if subject_id is not None:
            subject_id = subject_id.to(device)
        kwargs = {"subject_id": subject_id} if "spatial_temporal" in type(model).__name__.lower() or "spatialtemporal" in type(model).__name__.lower() else {}
        
        pred = F.normalize(model(eeg, **kwargs), dim=-1).cpu()

    table = torch.load(args.clip_embeddings, map_location="cpu")
    bank_raw = table["embedding"].float()
    if target_center is not None:
        bank_raw = bank_raw - target_center
    bank = F.normalize(bank_raw, dim=-1)
    sims = pred @ bank.T
    topk = sims.topk(min(args.top_k, bank.shape[0]), dim=-1)

    stimuli_root = Path(args.stimuli_root)
    thumb = args.thumb_size
    label_h = 34
    margin = 12
    cols = 1 + topk.indices.shape[1]
    rows = len(val_idx)
    width = cols * thumb + (cols + 1) * margin
    height = rows * (thumb + label_h) + (rows + 1) * margin
    grid = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(grid)

    rows_json: list[dict] = []
    for r, dataset_idx in enumerate(val_idx):
        gt_id = str(batch["image_id"][r])
        gt_table_i = table["image_id"].index(gt_id)
        image_indices = [gt_table_i] + topk.indices[r].tolist()
        labels = ["GT"] + [f"top{i+1} {float(topk.values[r, i]):.2f}" for i in range(topk.indices.shape[1])]
        row_info = {"dataset_index": int(dataset_idx), "ground_truth": gt_id, "retrieved": []}

        for c, image_i in enumerate(image_indices):
            x = margin + c * (thumb + margin)
            y = margin + r * (thumb + label_h + margin)
            image_path = _resolve_image(str(table["image_path"][image_i]), stimuli_root)
            img = _fit_image(Image.open(image_path), thumb)
            grid.paste(img, (x, y))
            image_id = str(table["image_id"][image_i])
            _draw_label(draw, (x, y + thumb + 2), f"{labels[c]} {image_id}")
            if c > 0:
                row_info["retrieved"].append(
                    {"rank": c, "image_id": image_id, "score": float(topk.values[r, c - 1]), "path": str(image_path)}
                )
        rows_json.append(row_info)

    grid_path = output_dir / "eeg_clip_retrieval_grid.jpg"
    grid.save(grid_path, quality=92)
    (output_dir / "retrieval_rows.json").write_text(json.dumps(rows_json, indent=2))
    print(json.dumps({"grid": str(grid_path), "rows": rows, "cols": cols, "device": str(device)}, indent=2))


if __name__ == "__main__":
    main()

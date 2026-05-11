"""CLIP image embedding utilities for NOD stimulus images."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from PIL import Image


@dataclass(frozen=True)
class ClipEmbeddingConfig:
    model_name: str = "openai/clip-vit-base-patch32"
    batch_size: int = 32
    device: str | None = None
    normalize: bool = True


def normalize_image_id(image_id: str) -> str:
    """Return the filename stem used by NOD/ImageNet stimulus images."""
    image_id = str(image_id)
    if image_id.lower().endswith((".jpeg", ".jpg", ".png")):
        return Path(image_id).stem
    return image_id


def candidate_image_paths(stimuli_root: str | Path, class_id: str, image_id: str) -> list[Path]:
    """
    Candidate paths for a NOD stimulus image.

    NOD metadata stores rows like `class_id=n03594734`,
    `image_id=n03594734_45507`. OpenNeuro layouts may be flat or grouped by
    synset, so this checks both without assuming one exact layout.
    """
    root = Path(stimuli_root)
    class_id = str(class_id)
    stem = normalize_image_id(image_id)
    return [
        root / f"{stem}.JPEG",
        root / f"{stem}.jpg",
        root / f"{stem}.jpeg",
        root / class_id / f"{stem}.JPEG",
        root / class_id / f"{stem}.jpg",
        root / class_id / f"{stem}.jpeg",
        root / "ImageNet" / f"{stem}.JPEG",
        root / "ImageNet" / class_id / f"{stem}.JPEG",
    ]


def resolve_image_path(stimuli_root: str | Path, class_id: str, image_id: str) -> Path | None:
    """Return the first existing image path for a metadata row, if present."""
    for path in candidate_image_paths(stimuli_root, class_id, image_id):
        if path.exists():
            return path
    return None


def missing_image_includes(
    metadata: pd.DataFrame,
    stimuli_prefix: str = "stimuli/ImageNet",
    *,
    layout: str = "flat",
) -> list[str]:
    """
    Build OpenNeuro include globs for missing NOD images.

    The returned paths are intentionally specific enough for targeted downloads.

    OpenNeuro rejects include paths that do not exist, so use the known dataset
    layout by default.  NOD ds005811 currently stores ImageNet stimuli flat
    under ``stimuli/ImageNet``.  ``layout="both"`` is available for local
    diagnostics only when a caller can tolerate invalid alternates.
    """
    if layout not in {"flat", "synset", "both"}:
        raise ValueError(f"Unsupported layout {layout!r}; expected flat, synset, or both")
    includes: list[str] = []
    seen: set[str] = set()
    for row in metadata.drop_duplicates(["class_id", "image_id"]).itertuples(index=False):
        class_id = str(getattr(row, "class_id"))
        stem = normalize_image_id(str(getattr(row, "image_id")))
        candidates: list[str] = []
        if layout in {"flat", "both"}:
            candidates.append(f"{stimuli_prefix}/{stem}.JPEG")
        if layout in {"synset", "both"}:
            candidates.append(f"{stimuli_prefix}/{class_id}/{stem}.JPEG")
        for inc in candidates:
            if inc not in seen:
                seen.add(inc)
                includes.append(inc)
    return includes


def _coerce_clip_features(features, model) -> torch.Tensor:
    """Return image features as a tensor across Transformers versions."""
    if isinstance(features, torch.Tensor):
        return features
    if hasattr(features, "image_embeds") and isinstance(features.image_embeds, torch.Tensor):
        return features.image_embeds
    if hasattr(features, "pooler_output") and isinstance(features.pooler_output, torch.Tensor):
        pooled = features.pooler_output
        projection = getattr(model, "visual_projection", None)
        if projection is not None and pooled.shape[-1] == getattr(projection, "in_features", None):
            return projection(pooled)
        return pooled
    raise TypeError(f"Unsupported CLIP image feature output type: {type(features)!r}")


def _load_clip(model_name: str, device: str | None):
    from transformers import CLIPModel, CLIPProcessor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, device


def embed_images(
    image_paths: Iterable[Path],
    *,
    config: ClipEmbeddingConfig | None = None,
) -> torch.Tensor:
    """Embed images with CLIP and return a tensor shaped `[n_images, dim]`."""
    config = config or ClipEmbeddingConfig()
    paths = list(image_paths)
    model, processor, device = _load_clip(config.model_name, config.device)
    outputs: list[torch.Tensor] = []

    with torch.inference_mode():
        for start in range(0, len(paths), config.batch_size):
            batch_paths = paths[start : start + config.batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=images, return_tensors="pt").to(device)
            feats = _coerce_clip_features(model.get_image_features(**inputs), model)
            if config.normalize:
                feats = torch.nn.functional.normalize(feats, dim=-1)
            outputs.append(feats.cpu())

    if not outputs:
        return torch.empty((0, 0), dtype=torch.float32)
    return torch.cat(outputs, dim=0)


def build_clip_embedding_table(
    metadata_csv: str | Path,
    stimuli_root: str | Path,
    output_pt: str | Path,
    *,
    config: ClipEmbeddingConfig | None = None,
) -> dict:
    """Generate CLIP embeddings for unique images referenced by crop metadata."""
    metadata = pd.read_csv(metadata_csv)
    required = {"class_id", "image_id"}
    missing_cols = required - set(metadata.columns)
    if missing_cols:
        raise ValueError(f"Metadata missing required columns: {sorted(missing_cols)}")

    unique = metadata.drop_duplicates(["class_id", "image_id"]).reset_index(drop=True).copy()
    paths: list[Path] = []
    missing: list[dict[str, str]] = []
    for row in unique.itertuples(index=False):
        path = resolve_image_path(stimuli_root, getattr(row, "class_id"), getattr(row, "image_id"))
        if path is None:
            missing.append({"class_id": str(getattr(row, "class_id")), "image_id": str(getattr(row, "image_id"))})
        else:
            paths.append(path)

    if missing:
        examples = ", ".join(f"{m['class_id']}/{m['image_id']}" for m in missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} stimulus images under {stimuli_root}. "
            f"Examples: {examples}. "
            "Use scripts/generate_clip_embeddings.py --write-openneuro-include-list to create targeted includes."
        )

    embeddings = embed_images(paths, config=config)
    table = {
        "model_name": (config or ClipEmbeddingConfig()).model_name,
        "image_id": unique["image_id"].astype(str).tolist(),
        "class_id": unique["class_id"].astype(str).tolist(),
        "image_path": [str(p) for p in paths],
        "embedding": embeddings,
    }
    output_pt = Path(output_pt)
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(table, output_pt)
    return {"output_pt": str(output_pt), "num_images": len(paths), "embedding_shape": list(embeddings.shape)}

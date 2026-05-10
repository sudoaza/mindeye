"""Dataset utilities for ZUNA semantic EEG crops paired with CLIP image embeddings."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SemanticPairConfig:
    """Paths needed to construct EEG→CLIP training pairs."""

    metadata_csv: str | Path
    epochs_dir: str | Path
    clip_embeddings_pt: str | Path
    normalize_eeg: bool = True
    preload_npz: bool = True


class ZunaClipPairDataset(Dataset):
    """
    Pair event-aligned ZUNA EEG crops with ground-truth CLIP image embeddings.

    The cropper writes one row per epoch in `all_runs_metadata.csv` and records
    the per-run compressed NPZ file in `npz_file`. The CLIP embedding table is
    produced by `scripts/generate_clip_embeddings.py` and contains image IDs plus
    a tensor shaped `[n_images, embedding_dim]`.
    """

    def __init__(self, config: SemanticPairConfig):
        self.config = config
        self.metadata_csv = Path(config.metadata_csv)
        self.epochs_dir = Path(config.epochs_dir)
        self.clip_embeddings_pt = Path(config.clip_embeddings_pt)

        self.metadata = pd.read_csv(self.metadata_csv).reset_index(drop=True)
        required = {"image_id", "npz_file"}
        missing = required - set(self.metadata.columns)
        if missing:
            raise ValueError(f"Metadata missing required columns: {sorted(missing)}")

        table = torch.load(self.clip_embeddings_pt, map_location="cpu")
        self.embedding_dim = int(table["embedding"].shape[-1])
        self.image_to_embedding = {
            str(image_id): table["embedding"][i].float()
            for i, image_id in enumerate(table["image_id"])
        }
        missing_images = sorted(set(self.metadata["image_id"].astype(str)) - set(self.image_to_embedding))
        if missing_images:
            examples = ", ".join(missing_images[:5])
            raise ValueError(f"Missing CLIP embeddings for {len(missing_images)} image IDs. Examples: {examples}")

        self._epoch_offsets = self._add_epoch_offsets(self.metadata)
        self._npz_cache: dict[str, np.ndarray] = {}
        if config.preload_npz:
            for npz_file in sorted(self.metadata["npz_file"].astype(str).unique()):
                self._npz_cache[npz_file] = self._load_npz(npz_file)

        first = self._get_eeg(0)
        self.eeg_shape = tuple(int(x) for x in first.shape)

    @staticmethod
    def _add_epoch_offsets(metadata: pd.DataFrame) -> list[int]:
        """Return zero-based row offsets within each per-run NPZ file."""
        offsets: list[int] = []
        counts: dict[str, int] = {}
        for npz_file in metadata["npz_file"].astype(str):
            offset = counts.get(npz_file, 0)
            offsets.append(offset)
            counts[npz_file] = offset + 1
        return offsets

    def _load_npz(self, npz_file: str) -> np.ndarray:
        path = self.epochs_dir / npz_file
        if not path.exists():
            raise FileNotFoundError(f"Missing semantic epoch NPZ: {path}")
        return np.load(path)["eeg"].astype("float32")

    def _get_npz(self, npz_file: str) -> np.ndarray:
        if npz_file not in self._npz_cache:
            self._npz_cache[npz_file] = self._load_npz(npz_file)
        return self._npz_cache[npz_file]

    def _get_eeg(self, idx: int) -> torch.Tensor:
        row = self.metadata.iloc[idx]
        npz_file = str(row["npz_file"])
        epoch_idx = self._epoch_offsets[idx]
        eeg = torch.from_numpy(self._get_npz(npz_file)[epoch_idx]).float()
        if self.config.normalize_eeg:
            mean = eeg.mean(dim=-1, keepdim=True)
            std = eeg.std(dim=-1, keepdim=True).clamp_min(1e-6)
            eeg = (eeg - mean) / std
        return eeg

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        row = self.metadata.iloc[idx]
        image_id = str(row["image_id"])
        return {
            "eeg": self._get_eeg(idx),
            "clip": self.image_to_embedding[image_id],
            "image_id": image_id,
            "index": int(idx),
        }


def split_indices(n_items: int, *, val_fraction: float = 0.15, seed: int = 13) -> tuple[list[int], list[int]]:
    """Deterministically split indices for baseline training."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_items, generator=gen).tolist()
    n_val = max(1, int(round(n_items * val_fraction)))
    return perm[n_val:], perm[:n_val]


def gather_clip_targets(dataset: ZunaClipPairDataset, indices: Sequence[int]) -> torch.Tensor:
    """Stack CLIP targets for retrieval evaluation."""
    return torch.stack([dataset[i]["clip"] for i in indices]).float()

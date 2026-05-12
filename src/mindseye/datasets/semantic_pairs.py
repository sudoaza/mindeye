"""Dataset utilities for ZUNA semantic EEG crops paired with CLIP image embeddings."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


TargetMode = Literal["real", "shuffled", "random", "sameclass"]
InputDomain = Literal["zuna", "raw", "resample"]
WindowMode = Literal["crop", "full5s"]
SemanticTarget = Literal["image", "text", "image_text"]


@dataclass(frozen=True)
class SemanticPairConfig:
    """Paths and target-mode needed to construct EEG→CLIP training pairs.

    target_mode controls what CLIP vector is used as the supervision target:
      - "real"      → paired CLIP embedding for the shown image (default)
      - "shuffled"  → CLIP embeddings are shuffled across the dataset (fixed seed)
      - "random"    → a fresh Gaussian vector normalized to the CLIP unit sphere
      - "sameclass" → a different embedding from the same ImageNet synset (if available)

    input_domain selects which NPZ file to read EEG from:
      - "zuna"      → standard ZUNA-denoised crops (default)
      - "raw"       → un-denoised raw crops from a parallel epochs_dir_raw
      - "resample"  → resample-only crops from epochs_dir_resample
    """

    metadata_csv: str | Path
    epochs_dir: str | Path
    clip_embeddings_pt: str | Path
    text_embeddings_pt: str | Path | None = None
    normalize_eeg: bool = True
    preload_npz: bool = True
    target_mode: TargetMode = "real"
    input_domain: InputDomain = "zuna"
    window_mode: WindowMode = "crop"
    semantic_target: SemanticTarget = "image"
    # Alternative epoch directories for raw / resample conditions
    epochs_dir_raw: str | Path | None = None
    epochs_dir_resample: str | Path | None = None
    shuffle_seed: int = 42


class ZunaClipPairDataset(Dataset):
    """
    Pair event-aligned EEG crops with ground-truth (or control) CLIP embeddings.

    Supports 4 target modes and 3 input domains so all 6 baseline-matrix conditions
    can be exercised from the same dataset class without code duplication.
    """

    def __init__(self, config: SemanticPairConfig):
        self.config = config
        self.metadata_csv = Path(config.metadata_csv)
        self.clip_embeddings_pt = Path(config.clip_embeddings_pt)

        # Select the epoch dir based on input domain
        if config.input_domain == "raw":
            if config.epochs_dir_raw is None:
                raise ValueError("epochs_dir_raw must be set when input_domain='raw'")
            self.epochs_dir = Path(config.epochs_dir_raw)
        elif config.input_domain == "resample":
            if config.epochs_dir_resample is None:
                raise ValueError("epochs_dir_resample must be set when input_domain='resample'")
            self.epochs_dir = Path(config.epochs_dir_resample)
        else:
            self.epochs_dir = Path(config.epochs_dir)

        self.metadata = pd.read_csv(self.metadata_csv).reset_index(drop=True)
        required = {"image_id", "npz_file"}
        missing = required - set(self.metadata.columns)
        if missing:
            raise ValueError(f"Metadata missing required columns: {sorted(missing)}")

        table = torch.load(self.clip_embeddings_pt, map_location="cpu")
        self.embedding_dim = int(table["embedding"].shape[-1])
        self.image_to_embedding: dict[str, torch.Tensor] = {
            str(image_id): table["embedding"][i].float()
            for i, image_id in enumerate(table["image_id"])
        }
        # Filter metadata to only include items that have CLIP embeddings
        initial_n = len(self.metadata)
        self.metadata["image_id_str"] = self.metadata["image_id"].astype(str)
        mask = self.metadata["image_id_str"].isin(self.image_to_embedding.keys())
        self.metadata = self.metadata[mask].reset_index(drop=True)
        dropped = initial_n - len(self.metadata)
        if dropped > 0:
            print(f"  [Dataset] Dropped {dropped} samples due to missing CLIP image embeddings "
                  f"({len(self.metadata)} remaining)")

        # Load text embeddings if needed
        self.class_to_text_embedding: dict[str, torch.Tensor] = {}
        if config.semantic_target in ("text", "image_text"):
            if config.text_embeddings_pt is None:
                raise ValueError("text_embeddings_pt required for semantic_target='text' or 'image_text'")
            text_table = torch.load(config.text_embeddings_pt, map_location="cpu")
            self.class_to_text_embedding = text_table["class_to_embedding"]
            # Verify all metadata classes have text embeddings
            if "class" not in self.metadata.columns:
                 raise ValueError("Metadata missing 'class' column required for text targets")
            missing_text = sorted(set(self.metadata["class"].unique()) - set(self.class_to_text_embedding.keys()))
            if missing_text:
                raise ValueError(f"Missing text embeddings for {len(missing_text)} classes: {missing_text[:5]}")

        self._epoch_offsets = self._add_epoch_offsets(self.metadata)
        self._npz_cache: dict[str, np.ndarray] = {}
        if config.preload_npz:
            for npz_file in sorted(self.metadata["npz_file"].astype(str).unique()):
                self._npz_cache[npz_file] = self._load_npz(npz_file)

        first = self._get_eeg(0)
        self.eeg_shape = tuple(int(x) for x in first.shape)

        # Build shuffled / random target index once at init time
        n = len(self.metadata)
        rng = np.random.default_rng(config.shuffle_seed)
        if config.target_mode == "shuffled":
            self._target_perm = rng.permutation(n).tolist()
        elif config.target_mode == "random":
            # Sample unit-sphere Gaussian vectors, one per sample
            vecs = rng.standard_normal((n, self.embedding_dim)).astype("float32")
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            self._random_targets: list[torch.Tensor] = [
                torch.from_numpy(vecs[i] / norms[i]).float() for i in range(n)
            ]
        elif config.target_mode == "sameclass":
            self._build_sameclass_index(rng)

        # Log target alignment if using multi-modal targets
        if config.semantic_target == "image_text":
            self._log_target_alignment()

    def _log_target_alignment(self) -> None:
        """Log the cosine similarity between paired image and text targets."""
        cosines = []
        for idx in range(len(self.metadata)):
            row = self.metadata.iloc[idx]
            img_emb = F.normalize(self.image_to_embedding[str(row["image_id"])], dim=-1)
            txt_emb = F.normalize(self.class_to_text_embedding[str(row["class"])], dim=-1)
            cos = (img_emb * txt_emb).sum().item()
            cosines.append(cos)
        cosines = np.array(cosines)
        print(f"  [Dataset] Target alignment (image <-> text): "
              f"mean={cosines.mean():.3f}, std={cosines.std():.3f}")

    # ------------------------------------------------------------------ helpers

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

    def _build_sameclass_index(self, rng: np.random.Generator) -> None:
        """Map each sample to a random *different* sample sharing the same synset.

        Falls back to the real target when the synset has only one sample."""
        if "synset" not in self.metadata.columns:
            # Try to extract synset from image_id (e.g. "n01440764_1234" → "n01440764")
            self.metadata["synset"] = self.metadata["image_id"].astype(str).str.split("_").str[0]

        synset_to_indices: dict[str, list[int]] = {}
        for idx, syn in enumerate(self.metadata["synset"]):
            synset_to_indices.setdefault(syn, []).append(idx)

        self._sameclass_targets: list[int] = []
        for idx in range(len(self.metadata)):
            syn = self.metadata.iloc[idx]["synset"]
            pool = [i for i in synset_to_indices[syn] if i != idx]
            self._sameclass_targets.append(int(rng.choice(pool)) if pool else idx)

    # ------------------------------------------------------------------ Dataset API

    def _get_clip(self, idx: int) -> torch.Tensor:
        """Return the target embedding for sample *idx* based on target_mode and semantic_target."""
        mode = self.config.target_mode
        if mode == "shuffled":
            idx = self._target_perm[idx]
        elif mode == "random":
            return self._random_targets[idx]
        elif mode == "sameclass":
            idx = self._sameclass_targets[idx]
        
        row = self.metadata.iloc[idx]
        
        img_id = str(row["image_id"])
        img_emb = F.normalize(self.image_to_embedding[img_id], dim=-1)
        
        if self.config.semantic_target == "image":
            return img_emb
        
        label = str(row["class"])
        txt_emb = F.normalize(self.class_to_text_embedding[label], dim=-1)
        
        if self.config.semantic_target == "text":
            return txt_emb
        
        # Combined image_text: normalize(normalize(image) + normalize(text))
        combined = torch.nn.functional.normalize(img_emb + txt_emb, dim=-1)
        return combined

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        return {
            "eeg": self._get_eeg(idx),
            "clip": self._get_clip(idx),
            "image_id": str(self.metadata.iloc[idx]["image_id"]),
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

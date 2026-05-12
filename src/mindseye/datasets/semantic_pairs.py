"""Dataset utilities for ZUNA semantic EEG crops paired with CLIP image embeddings."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

TargetMode = Literal["real", "shuffled", "random", "sameclass"]
InputDomain = Literal["zuna", "raw", "resample"]
WindowMode = Literal["crop", "full5s", "full5s_backaligned"]
TargetSpace = Literal["image", "semantic", "common"]


@dataclass(frozen=True)
class SemanticPairConfig:
    metadata_csv: str | Path
    epochs_dir: str | Path
    common_embeddings_pt: str | Path
    normalize_eeg: bool = True
    preload_npz: bool = True
    target_mode: TargetMode = "real"
    input_domain: InputDomain = "zuna"
    window_mode: WindowMode = "crop"
    target_space: TargetSpace = "common"
    epochs_dir_raw: str | Path | None = None
    epochs_dir_resample: str | Path | None = None
    shuffle_seed: int = 42
    add_event_marker: bool = False


class ZunaClipPairDataset(Dataset):
    def __init__(self, config: SemanticPairConfig):
        self.config = config
        self.metadata_csv = Path(config.metadata_csv)
        self.common_embeddings_pt = Path(config.common_embeddings_pt)

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

        table = torch.load(self.common_embeddings_pt, map_location="cpu")
        
        self.image_id_to_common = table["image_id_to_common"]
        self.image_id_to_image = table["image_id_to_image"]
        self.image_id_to_semantic = table["image_id_to_semantic"]
        
        # Pick the embedding dimension from the first item
        first_key = next(iter(self.image_id_to_common.keys()))
        self.embedding_dim = self.image_id_to_common[first_key].shape[-1]
        
        initial_n = len(self.metadata)
        self.metadata["image_id_str"] = self.metadata["image_id"].astype(str)
        mask = self.metadata["image_id_str"].isin(self.image_id_to_common.keys())
        self.metadata = self.metadata[mask].reset_index(drop=True)
        dropped = initial_n - len(self.metadata)
        if dropped > 0:
            print(f"  [Dataset] Dropped {dropped} samples due to missing CLIP image embeddings "
                  f"({len(self.metadata)} remaining)")

        self._epoch_offsets = self._add_epoch_offsets(self.metadata)
        self._npz_cache: dict[str, np.ndarray] = {}
        if config.preload_npz:
            for npz_file in sorted(self.metadata["npz_file"].astype(str).unique()):
                self._npz_cache[npz_file] = self._load_npz(npz_file)

        first = self._get_eeg(0)
        self.eeg_shape = tuple(int(x) for x in first.shape)

        if config.window_mode == "full5s_backaligned":
            if self.eeg_shape[-1] not in (1280, 1281):
                raise ValueError(f"full5s_backaligned requires sample length 1280 or 1281, got {self.eeg_shape[-1]}")
            if "event_offset_s" not in self.metadata.columns:
                raise ValueError("event_offset_s missing from metadata")
            if "anchor_sample" not in self.metadata.columns:
                raise ValueError("anchor_sample missing from metadata")
            self.eeg_shape = (self.eeg_shape[0], 1280)

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

    @staticmethod
    def _add_epoch_offsets(metadata: pd.DataFrame) -> list[int]:
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
        
        if self.config.window_mode == "full5s_backaligned":
            eeg = eeg[:, :1280]
        
        if self.config.normalize_eeg:
            mean = eeg.mean(dim=-1, keepdim=True)
            std = eeg.std(dim=-1, keepdim=True).clamp_min(1e-6)
            eeg = (eeg - mean) / std
            
        if getattr(self.config, "add_event_marker", False):
            if "anchor_sample" not in row:
                raise ValueError("anchor_sample missing from metadata")
            anchor = float(row["anchor_sample"])
            t = torch.arange(eeg.shape[1], dtype=torch.float32)
            sigma_samples = 16.0
            marker = torch.exp(-0.5 * ((t - anchor) / sigma_samples) ** 2).unsqueeze(0)
            eeg = torch.cat([eeg, marker], dim=0)

        return eeg

    def _build_sameclass_index(self, rng: np.random.Generator) -> None:
        if "synset" not in self.metadata.columns:
            self.metadata["synset"] = self.metadata["image_id"].astype(str).str.split("_").str[0]

        synset_to_indices: dict[str, list[int]] = {}
        for idx, syn in enumerate(self.metadata["synset"]):
            synset_to_indices.setdefault(syn, []).append(idx)

        self._sameclass_targets: list[int] = []
        for idx in range(len(self.metadata)):
            syn = self.metadata.iloc[idx]["synset"]
            pool = [i for i in synset_to_indices[syn] if i != idx]
            self._sameclass_targets.append(int(rng.choice(pool)) if pool else idx)

    def _get_targets(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mode = self.config.target_mode
        if mode == "shuffled":
            idx = self._target_perm[idx]
        elif mode == "random":
            rand_t = self._random_targets[idx]
            return rand_t, rand_t, rand_t
        elif mode == "sameclass":
            idx = self._sameclass_targets[idx]
        
        row = self.metadata.iloc[idx]
        img_id = str(row["image_id"])
        
        t_common = F.normalize(self.image_id_to_common[img_id], dim=-1)
        t_image = F.normalize(self.image_id_to_image[img_id], dim=-1)
        t_semantic = F.normalize(self.image_id_to_semantic[img_id], dim=-1)
        
        return t_common, t_image, t_semantic

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        t_common, t_image, t_semantic = self._get_targets(idx)
        return {
            "eeg": self._get_eeg(idx),
            "target_common": t_common,
            "target_image": t_image,
            "target_semantic": t_semantic,
            "image_id": str(self.metadata.iloc[idx]["image_id"]),
            "index": int(idx),
        }


def split_indices(n_items: int, *, val_fraction: float = 0.15, seed: int = 13) -> tuple[list[int], list[int]]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_items, generator=gen).tolist()
    n_val = max(1, int(round(n_items * val_fraction)))
    return perm[n_val:], perm[:n_val]

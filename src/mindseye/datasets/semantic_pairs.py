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
from mindseye.models.common_probe import CommonProbeModel, ATTRIBUTE_SCHEMAS, IGNORE_INDEX

TargetMode = Literal["real", "shuffled", "random", "sameclass"]
InputDomain = Literal["zuna", "raw", "resample"]
WindowMode = Literal["crop", "full5s", "full5s_backaligned"]
TargetSpace = Literal["image", "semantic", "common", "label"]


@dataclass(frozen=True)
class SemanticPairConfig:
    metadata_csv: str | Path
    epochs_dir: str | Path
    common_embeddings_pt: str | Path
    vlm_attributes_json: str | Path | None = None
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
    augment_eeg: bool = False
    aug_channel_dropout: float = 0.10
    aug_noise_std: float = 0.03
    aug_amp_scale: float = 0.10
    aug_time_mask: int = 24
    aug_time_jitter: int = 8


class ZunaClipPairDataset(Dataset):
    def __init__(self, config: SemanticPairConfig):
        self.config = config
        self.common_embeddings_pt = Path(config.common_embeddings_pt)

        # Allow comma-separated strings for metadata_csv and epochs_dir paths
        def split_paths(path_str_or_paths) -> list[str]:
            if not path_str_or_paths:
                return []
            if isinstance(path_str_or_paths, (list, tuple)):
                return [str(p) for p in path_str_or_paths]
            return [p.strip() for p in str(path_str_or_paths).split(",")]

        metadata_csv_list = split_paths(config.metadata_csv)
        
        # Select base epochs_dir list based on input domain
        if config.input_domain == "raw":
            if config.epochs_dir_raw is None:
                raise ValueError("epochs_dir_raw must be set when input_domain='raw'")
            epochs_dir_list = split_paths(config.epochs_dir_raw)
        elif config.input_domain == "resample":
            if config.epochs_dir_resample is None:
                raise ValueError("epochs_dir_resample must be set when input_domain='resample'")
            epochs_dir_list = split_paths(config.epochs_dir_resample)
        else:
            epochs_dir_list = split_paths(config.epochs_dir)

        if len(metadata_csv_list) != len(epochs_dir_list):
            if len(epochs_dir_list) == 1:
                epochs_dir_list = epochs_dir_list * len(metadata_csv_list)
            elif len(metadata_csv_list) == 1:
                metadata_csv_list = metadata_csv_list * len(epochs_dir_list)
            else:
                raise ValueError(
                    f"Mismatch in number of metadata CSVs ({len(metadata_csv_list)}) "
                    f"and epochs directories ({len(epochs_dir_list)})"
                )

        dfs = []
        valid_ep_dirs = []
        for csv_path, ep_dir in zip(metadata_csv_list, epochs_dir_list):
            csv_path = Path(csv_path)
            ep_dir = Path(ep_dir)
            if not csv_path.exists() or not ep_dir.exists():
                print(f"[WARN] Skipping missing subject dataset: csv={csv_path} (exists={csv_path.exists()}), ep_dir={ep_dir} (exists={ep_dir.exists()})")
                continue
            
            df = pd.read_csv(csv_path)
            if config.input_domain == "raw":
                df["npz_file"] = df["npz_file"].str.replace("_zuna_semantic.npz", "_raw_semantic.npz")
            elif config.input_domain == "resample":
                df["npz_file"] = df["npz_file"].str.replace("_zuna_semantic.npz", "_resample_semantic.npz")
            
            required = {"image_id", "npz_file"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"Metadata {csv_path} missing required columns: {sorted(missing)}")
                
            df["epochs_dir_resolved"] = str(ep_dir)
            dfs.append(df)
            valid_ep_dirs.append(ep_dir)

        if not dfs:
            raise FileNotFoundError(
                f"No valid subject datasets found among the requested paths.\n"
                f"Metadata paths checked: {metadata_csv_list}\n"
                f"Epochs paths checked: {epochs_dir_list}"
            )

        self.metadata = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
        # Use first valid epochs_dir as default for backwards compatibility
        self.epochs_dir = valid_ep_dirs[0] if valid_ep_dirs else Path(".")

        if config.target_space != "common":
            import warnings
            warnings.warn(f"Non-canonical target_space '{config.target_space}' specified. Only 'common' is canonical.")

        table = torch.load(self.common_embeddings_pt, map_location="cpu")
        
        self.vlm_attributes = {}
        if config.vlm_attributes_json is not None:
            import json
            with open(config.vlm_attributes_json, "r") as f:
                self.vlm_attributes = json.load(f)
        
        target_key = f"image_id_to_{config.target_space}"
        if target_key not in table:
            raise ValueError(f"Target space '{config.target_space}' not found in {self.common_embeddings_pt}")
            
        self.image_id_to_target = table[target_key]
        
        if "class" in self.metadata.columns:
            unique_classes = sorted(self.metadata["class"].dropna().unique().tolist())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
            self.idx_to_class = unique_classes
        else:
            self.class_to_idx = {}
            self.idx_to_class = []
        
        # Pick the embedding dimension from the first item
        first_key = next(iter(self.image_id_to_target.keys()))
        self.embedding_dim = self.image_id_to_target[first_key].shape[-1]
        
        initial_n = len(self.metadata)
        self.metadata["image_id_str"] = self.metadata["image_id"].astype(str)
        mask = self.metadata["image_id_str"].isin(self.image_id_to_target.keys())
        self.metadata = self.metadata[mask].reset_index(drop=True)
        dropped = initial_n - len(self.metadata)
        if dropped > 0:
            print(f"  [Dataset] Dropped {dropped} samples due to missing target embeddings "
                  f"({len(self.metadata)} remaining)")

        if "subject" in self.metadata.columns:
            self.unique_subjects = sorted(self.metadata["subject"].astype(str).unique().tolist())
            self.subject_to_id = {sub: i for i, sub in enumerate(self.unique_subjects)}
        else:
            self.unique_subjects = ["unknown"]
            self.subject_to_id = {"unknown": 0}

        self._epoch_offsets = self._add_epoch_offsets(self.metadata)
        self._npz_cache: dict[tuple[str, str], np.ndarray] = {}
        if config.preload_npz:
            unique_npz_pairs = self.metadata[["npz_file", "epochs_dir_resolved"]].drop_duplicates()
            for _, row in unique_npz_pairs.iterrows():
                npz_f = str(row["npz_file"])
                ep_d = str(row["epochs_dir_resolved"])
                self._npz_cache[(npz_f, ep_d)] = self._load_npz(npz_f, Path(ep_d))

        first = self._get_eeg(0)
        self.eeg_shape = tuple(int(x) for x in first.shape)

        # Load channel names from first NPZ if available
        self.ch_names = None
        try:
            first_row = self.metadata.iloc[0]
            first_npz = first_row["npz_file"]
            first_ep_dir = Path(first_row["epochs_dir_resolved"])
            with np.load(first_ep_dir / first_npz) as f:
                if "ch_names" in f:
                    self.ch_names = list(f["ch_names"])
        except Exception:
            pass

        if self.ch_names is not None:
            if config.add_event_marker:
                self.ch_names = self.ch_names + ["EVENT_MARKER"]

        if config.window_mode == "full5s_backaligned":
            if self.eeg_shape[-1] not in (1280, 1281):
                raise ValueError(f"full5s_backaligned requires sample length 1280 or 1281, got {self.eeg_shape[-1]}")
            if "event_offset_s" not in self.metadata.columns:
                raise ValueError("event_offset_s missing from metadata")
            if "anchor_sample" not in self.metadata.columns:
                raise ValueError("anchor_sample missing from metadata")
            self.eeg_shape = (self.eeg_shape[0], 1280)
            
        elif config.window_mode == "tight1s":
            if self.eeg_shape[-1] not in (301, 302, 307, 308):
                raise ValueError(f"tight1s requires sample length 301, 302, 307 or 308, got {self.eeg_shape[-1]}")
            # Standardize length based on the frequency (301 for 250Hz, 307 for 256Hz)
            target_len = 301 if self.eeg_shape[-1] in (301, 302) else 307
            self.eeg_shape = (self.eeg_shape[0], target_len)

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
        counts: dict[tuple[str, str], int] = {}
        for _, row in metadata.iterrows():
            key = (str(row["npz_file"]), str(row["epochs_dir_resolved"]))
            offset = counts.get(key, 0)
            offsets.append(offset)
            counts[key] = offset + 1
        return offsets

    def _load_npz(self, npz_file: str, epochs_dir: Path) -> np.ndarray:
        path = epochs_dir / npz_file
        if not path.exists():
            raise FileNotFoundError(f"Missing semantic epoch NPZ: {path}")
        return np.load(path)["eeg"].astype("float32")

    def _get_npz(self, npz_file: str, epochs_dir: Path) -> np.ndarray:
        key = (npz_file, str(epochs_dir))
        if key not in self._npz_cache:
            self._npz_cache[key] = self._load_npz(npz_file, epochs_dir)
        return self._npz_cache[key]

    def _get_eeg(self, idx: int) -> torch.Tensor:
        row = self.metadata.iloc[idx]
        npz_file = str(row["npz_file"])
        epochs_dir = Path(row["epochs_dir_resolved"])
        epoch_idx = self._epoch_offsets[idx]
        eeg = torch.from_numpy(self._get_npz(npz_file, epochs_dir)[epoch_idx]).float()
        
        if self.config.window_mode == "full5s_backaligned":
            eeg = eeg[:, :1280]
        elif self.config.window_mode == "tight1s":
            target_len = 301 if eeg.shape[-1] in (301, 302) else 307
            eeg = eeg[:, :target_len]
        
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

        if self.config.augment_eeg:
            eeg = self._augment_eeg(eeg)

        return eeg

    def _non_marker_view(self, eeg: torch.Tensor) -> torch.Tensor:
        """Return channels that may be augmented; never augment appended marker."""
        if getattr(self.config, "add_event_marker", False):
            return eeg[:-1]
        return eeg

    def _augment_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        """Lightweight stochastic EEG augmentations for train-time robustness.

        The optional event marker channel is intentionally excluded from all
        transforms so its timing/amplitude remains an absolute reference.
        """
        x = eeg.clone()
        data = self._non_marker_view(x)
        if data.numel() == 0:
            return x

        # Channel dropout: zero a small random subset of EEG channels.
        p_drop = float(self.config.aug_channel_dropout)
        if p_drop > 0:
            keep = torch.rand(data.shape[0], device=data.device) >= p_drop
            if not bool(keep.any()):
                keep[torch.randint(0, data.shape[0], (1,), device=data.device)] = True
            data *= keep[:, None].to(data.dtype)

        # Per-crop amplitude scaling.
        amp = float(self.config.aug_amp_scale)
        if amp > 0:
            scale = 1.0 + (torch.rand((), device=data.device, dtype=data.dtype) * 2.0 - 1.0) * amp
            data *= scale

        # Additive Gaussian sensor noise.
        noise = float(self.config.aug_noise_std)
        if noise > 0:
            data += torch.randn_like(data) * noise

        # Time masking over EEG channels only.
        mask_width = int(self.config.aug_time_mask)
        if mask_width > 0 and data.shape[-1] > 1:
            width = min(mask_width, data.shape[-1])
            start = int(torch.randint(0, data.shape[-1] - width + 1, (1,)).item())
            data[:, start:start + width] = 0

        # Small temporal jitter over EEG channels only; marker remains fixed.
        jitter = int(self.config.aug_time_jitter)
        if jitter > 0:
            shift = int(torch.randint(-jitter, jitter + 1, (1,)).item())
            if shift:
                data[:] = torch.roll(data, shifts=shift, dims=-1)

        return x

    def audit_target_banks(self) -> dict[str, float]:
        """Summarize whether target bank properties are sane."""
        image_ids = sorted(set(self.metadata["image_id"].astype(str).tolist()))
        target = torch.stack([F.normalize(self.image_id_to_target[i].float(), dim=-1) for i in image_ids])

        # Optimize memory usage using matrix multiplication
        cos = torch.mm(target, target.t())
        # Exclude diagonal
        cos_off = cos[~torch.eye(len(image_ids), dtype=torch.bool, device=cos.device)]
        
        return {
            "target_bank_n": float(len(image_ids)),
            "target_off_diag_mean": float(cos_off.mean().item()),
            "target_off_diag_std": float(cos_off.std().item()),
        }

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

    def _get_targets(self, idx: int) -> torch.Tensor:
        mode = self.config.target_mode
        if mode == "shuffled":
            idx = self._target_perm[idx]
        elif mode == "random":
            return self._random_targets[idx]
        elif mode == "sameclass":
            idx = self._sameclass_targets[idx]
        
        row = self.metadata.iloc[idx]
        img_id = str(row["image_id"])
        
        return F.normalize(self.image_id_to_target[img_id].float(), dim=-1)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int | dict]:
        target_idx = idx
        mode = self.config.target_mode
        if mode == "shuffled":
            target_idx = self._target_perm[idx]
        elif mode == "sameclass":
            target_idx = self._sameclass_targets[idx]

        target_row = self.metadata.iloc[target_idx]
        target_img_id = str(target_row["image_id"])
        
        target = self._get_targets(idx)
        
        subject_str = str(target_row["subject"]) if "subject" in target_row else "unknown"
        subject_id = self.subject_to_id.get(subject_str, 0)
        
        probe_targets = {}
        if mode == "random":
            probe_targets["class_label"] = IGNORE_INDEX
            for attr in ATTRIBUTE_SCHEMAS.keys():
                probe_targets[attr] = IGNORE_INDEX
        else:
            class_val = target_row.get("class", None)
            if pd.isna(class_val) or class_val not in self.class_to_idx:
                probe_targets["class_label"] = IGNORE_INDEX
            else:
                probe_targets["class_label"] = self.class_to_idx[class_val]
                
            img_attrs = self.vlm_attributes.get(target_img_id, {}) if hasattr(self, "vlm_attributes") else {}
            for attr in ATTRIBUTE_SCHEMAS.keys():
                val = img_attrs.get(attr, "unclear")
                probe_targets[attr] = CommonProbeModel.encode_label(attr, val)
                
        ret: dict[str, torch.Tensor | str | int | dict] = {
            "eeg": self._get_eeg(idx),
            "target": target,
            "target_common": target,
            "probe_targets": probe_targets,
            "image_id": target_img_id,
            "index": int(idx),
            "subject_id": int(subject_id),
        }
        
        return ret


def split_indices(n_items: int, *, val_fraction: float = 0.15, seed: int = 13) -> tuple[list[int], list[int]]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_items, generator=gen).tolist()
    n_val = max(1, int(round(n_items * val_fraction)))
    return perm[n_val:], perm[:n_val]

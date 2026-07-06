#!/usr/bin/env python3
import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from pathlib import Path
import json
from PIL import Image
import torchvision.transforms.functional as TF

# Ensure import paths work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from mindseye.adapters.qformer import ZunaToVisionQFormer
from mindseye.models.eeg_encoder import (
    clip_contrastive_loss,
    soft_dino_contrastive_loss,
    retrieval_topk,
    retrieval_topk_full_bank,
)
from mindseye.generation.luminance import luminance_grid_loss

def parse_runs_spec(spec: str) -> list[int]:
    runs = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-")
            runs.update(range(int(start), int(end) + 1))
        else:
            runs.add(int(part))
    return sorted(list(runs))

def variance_floor_loss(pred: torch.Tensor, target_std: torch.Tensor) -> torch.Tensor:
    pred_std = torch.sqrt(pred.var(dim=0) + 1e-6)
    return torch.mean(F.relu(target_std - pred_std))


def batch_std_hinge_loss(pred_u: torch.Tensor, gamma) -> torch.Tensor:
    """VICReg-style anti-collapse on the *normalized* prediction batch.

    ``variance_floor_loss`` compares each vector's per-dim magnitude against a raw
    target std, but under ``force_unit_output`` every prediction is already a unit
    vector, so that floor is trivially met even when all predictions point the same
    direction (hub collapse). This term instead penalizes low *cross-sample* std of
    the normalized predictions per dimension, directly forcing the batch to spread
    over the sphere — the same quantity the ``collapse_pct`` metric measures.

    The hinge is *scale-normalized* (``1 - std/gamma``) so a fully collapsed batch
    yields ~1.0 per dim (making the loss weight meaningful against the O(1..10)
    InfoNCE term) and turns fully **off** once the per-dim std reaches ``gamma``.

    gamma should be the target distribution's own per-dim std of the normalized
    targets (a [D] tensor). Anchoring to the target spread — rather than the
    theoretical unit-vector value 1/sqrt(D) ~= 0.036 which the DINO targets never
    reach (their per-dim std ~= 0.027) — lets the term switch off exactly when preds
    match the target's natural spread, so it stops fighting cosine alignment after
    collapse is broken.
    """
    std_per_dim = torch.sqrt(pred_u.var(dim=0) + 1e-6)
    if not torch.is_tensor(gamma):
        gamma = torch.as_tensor(gamma, device=pred_u.device, dtype=pred_u.dtype)
    return torch.mean(F.relu(1.0 - std_per_dim / gamma.clamp_min(1e-6)))


class NegativeBank:
    """FIFO queue of detached, L2-normalized target embeddings + their int image
    ids, used as extra InfoNCE negatives (MoCo-style).

    Stores negatives without paying the activation-memory cost of a huge batch.
    Only meaningful when the enqueued targets are the true image embeddings that
    align with their image ids (i.e. the ``real`` target mode); enable it there.
    """

    def __init__(self, size: int, dim: int, device):
        self.size = int(size)
        self.device = device
        self._targets = torch.zeros(self.size, dim, device=device)
        self._ids = torch.full((self.size,), -1, dtype=torch.long, device=device)
        self._ptr = 0
        self._count = 0

    def __len__(self) -> int:
        return self._count

    @torch.no_grad()
    def enqueue(self, targets: torch.Tensor, image_ids: torch.Tensor) -> None:
        targets = targets.to(self.device)
        image_ids = image_ids.to(self.device)
        b = targets.shape[0]
        if b == 0:
            return
        if b >= self.size:
            self._targets.copy_(targets[-self.size:])
            self._ids.copy_(image_ids[-self.size:])
            self._ptr = 0
            self._count = self.size
            return
        idx = (self._ptr + torch.arange(b, device=self.device)) % self.size
        self._targets[idx] = targets
        self._ids[idx] = image_ids
        self._ptr = int((self._ptr + b) % self.size)
        self._count = min(self._count + b, self.size)

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the currently-filled (targets, image_ids)."""
        if self._count < self.size:
            return self._targets[: self._count], self._ids[: self._count]
        return self._targets, self._ids


def crop_zuna_latent(
    latent: torch.Tensor,
    n_channels: int = 62,
    tc: int = 40,
    d: int = 32,
    tc_start: int = 20,
    tc_end: int = 36,
) -> torch.Tensor:
    """
    Slice a ZUNA post_mmd latent to a temporal window around the stimulus onset.
    """
    if tc_start == 20 and tc_end == 36 and d == 32 and tc == 40 and n_channels == 62:
        assert latent.shape == (2480, 32), f"Expected input latent shape (2480, 32), got {latent.shape}"
        x = latent.view(n_channels, tc, d)
        cropped = x[:, tc_start:tc_end, :].reshape(n_channels * (tc_end - tc_start), d)
        assert cropped.shape == (992, 32), f"Expected cropped latent shape (992, 32), got {cropped.shape}"
        return cropped
    else:
        x = latent.view(n_channels, tc, d)
        return x[:, tc_start:tc_end, :].reshape(-1, d)


class ZunaLatentTargetDataset(Dataset):
    """
    Dataset that loads cached ZUNA latents and maps them to target embeddings (CLIP/DINO).
    Supports baseline control target modes: real, shuffled, and random.
    """
    def __init__(
        self,
        latents_pt_path: str,
        targets_pt_path: str,
        target_space: str,
        layer_name: str,
        target_mode: str = "real",
        shuffle_seed: int = 42,
        subject_list: list = None,
        use_temporal_window: bool = True,
        n_channels: int = 62,
        tc: int = 40,
        latent_tc_start: int = 20,
        latent_tc_end: int = 36,
        stimuli_dir: str = None,
        stim_size: int = 256,
    ):
        self.use_temporal_window = use_temporal_window
        self.n_channels = n_channels
        self.tc = tc
        self.latent_tc_start = latent_tc_start
        self.latent_tc_end = latent_tc_end
        # Optional stimulus image loading for the luminance-grounding reconstruction loss.
        # Only used when stimuli_dir is set; keeps the retrieval path unaffected.
        self.stimuli_dir = Path(stimuli_dir) if stimuli_dir else None
        self.stim_size = stim_size
        # Resolve cache dir and split paths
        if os.path.isdir(latents_pt_path):
            cache_dir = latents_pt_path
        else:
            cache_dir = os.path.dirname(latents_pt_path)
            
        metadata_path = os.path.join(cache_dir, "metadata.pt")
        layer_path = os.path.join(cache_dir, f"latents_{layer_name}.pt")
        
        # Fallback to combined latents.pt if metadata.pt doesn't exist
        if not os.path.exists(metadata_path):
            metadata_path = os.path.join(cache_dir, "latents.pt")
            
        print(f"Loading metadata from {metadata_path}...")
        self.records = torch.load(metadata_path, map_location="cpu")
        print(f"Loaded {len(self.records)} metadata records.")
        
        # Load layer dictionary
        if os.path.exists(layer_path):
            print(f"Loading layer '{layer_name}' latents from {layer_path}...")
            self.layer_dict = torch.load(layer_path, map_location="cpu")
            self.use_split_files = True
        else:
            print(f"Layer file {layer_path} not found. Assuming combined latents.pt structure.")
            self.use_split_files = False
        
        # Load targets dict
        print(f"Loading target embeddings from {targets_pt_path}...")
        targets_data = torch.load(targets_pt_path, map_location="cpu")
        
        # Map target spaces dynamically
        self.pca_dims = None
        target_space_key = target_space
        if "PCA" in target_space:
            self.pca_dims = int(target_space.split("-")[2])
            target_space_key = "rae_unit"
        elif target_space in ("DINO-CLS-768", "DINO-CLS"):
            target_space_key = "dino_cls"
        elif target_space in ("DINO-Unit-768", "DINO-Unit"):
            target_space_key = "rae_unit"
        elif target_space in ("CLIP-Common-512", "CLIP-Common"):
            target_space_key = "common"

        # Resolve target key
        if target_space_key in targets_data:
            self.image_id_to_target = targets_data[target_space_key]
        elif f"image_id_to_{target_space_key}" in targets_data:
            self.image_id_to_target = targets_data[f"image_id_to_{target_space_key}"]
        else:
            possible_keys = [k for k in targets_data.keys() if target_space_key in k]
            if possible_keys:
                self.image_id_to_target = targets_data[possible_keys[0]]
            else:
                raise ValueError(f"Target space '{target_space_key}' not found in keys: {list(targets_data.keys())}")
                
        # Filter records that have valid target embeddings
        self.valid_records = []
        for r in self.records:
            if r["image_id"] in self.image_id_to_target:
                # If using split files, ensure the sample exists in the layer dict
                if self.use_split_files and r["sample_id"] not in self.layer_dict:
                    continue
                if subject_list is not None:
                    if f"sub-{r['subject_id']:02d}" in subject_list:
                        self.valid_records.append(r)
                else:
                    self.valid_records.append(r)
                    
        print(f"Found {len(self.valid_records)} valid records with target embeddings.")
        if len(self.valid_records) == 0:
            raise ValueError("No valid records found after filtering.")

        # Finding M2: FiLM must be indexed by a cohort-relative id, not the raw
        # subject number, so cohorts that don't start at sub-01 (or have gaps)
        # don't index past the embedding table. Map sorted unique subject_ids -> 0..K-1.
        unique_subject_ids = sorted({int(r["subject_id"]) for r in self.valid_records})
        self.subject_id_to_index = {sid: i for i, sid in enumerate(unique_subject_ids)}
        self.num_subjects = len(unique_subject_ids)
        print(f"Cohort subjects (raw ids): {unique_subject_ids} -> FiLM indices 0..{self.num_subjects - 1}")

        # Stable string image_id -> contiguous int id, used to mask same-image
        # false negatives in the contrastive loss (NOD repeats stimuli across
        # subjects/runs). Sorted for determinism across processes.
        unique_image_ids = sorted({str(r["image_id"]) for r in self.valid_records})
        self.image_id_to_int = {img_id: i for i, img_id in enumerate(unique_image_ids)}
            
        # Expose dimensions
        self.layer_name = layer_name
        if self.use_split_files:
            first_latent = self.layer_dict[self.valid_records[0]["sample_id"]].float()
        else:
            first_latent = self.valid_records[0][layer_name].float()
        self.latent_dim = first_latent.shape[-1]
        first_target = self.image_id_to_target[self.valid_records[0]["image_id"]]

        if self.pca_dims is not None:
            self.target_dim = self.pca_dims
        else:
            self.target_dim = first_target.shape[-1]

        # Compute latent_seq_len after optional temporal windowing
        if self.use_temporal_window:
            # Finding M1: the fixed tc[start:end) window is only correct when the
            # stimulus onset sits where we assume it does. The cache stores onset_tc
            # per trial; verify the window brackets it and that onset_tc is uniform,
            # so a different crop config fails loudly instead of silently windowing
            # the wrong latents.
            onset_tcs = sorted({int(r.get("onset_tc", 24)) for r in self.valid_records})
            if len(onset_tcs) != 1:
                raise ValueError(
                    f"Non-uniform onset_tc across records {onset_tcs}; the fixed "
                    f"[{self.latent_tc_start}:{self.latent_tc_end}) latent window cannot be "
                    f"correct for all trials. Re-cache with a consistent crop or window per-trial."
                )
            onset_tc = onset_tcs[0]
            if not (self.latent_tc_start <= onset_tc < self.latent_tc_end):
                raise ValueError(
                    f"Latent window [{self.latent_tc_start}:{self.latent_tc_end}) does not bracket "
                    f"the stimulus onset (onset_tc={onset_tc}). This windows pre/post-onset latents "
                    f"incorrectly. Adjust --latent-tc-start/--latent-tc-end for this crop."
                )
            first_windowed = crop_zuna_latent(
                first_latent, self.n_channels, self.tc, self.latent_dim,
                self.latent_tc_start, self.latent_tc_end
            )
            self.latent_seq_len = first_windowed.shape[0]
        else:
            self.latent_seq_len = first_latent.shape[0]

        print(f"Latent dim: {self.latent_dim} | Latent seq: {self.latent_seq_len} | Target dim: {self.target_dim}")
        if self.use_temporal_window:
            onset_tc = self.valid_records[0].get("onset_tc", 24)
            pre_s = (self.latent_tc_start - onset_tc) * 0.125
            post_s = (self.latent_tc_end - onset_tc) * 0.125
            print(f"latent_crop_window_s = [{pre_s:+.1f}, {post_s:+.1f}]")
            print(f"latent_tc_start = {self.latent_tc_start}")
            print(f"latent_tc_end = {self.latent_tc_end}")
            print(f"tokens_before = {self.n_channels * self.tc}")
            print(f"tokens_after = {self.latent_seq_len}")
        
        self.target_mode = target_mode
        n = len(self.valid_records)
        rng = np.random.default_rng(shuffle_seed)
        
        if target_mode == "shuffled":
            self.target_perm = rng.permutation(n).tolist()
        elif target_mode == "random":
            # Generate unit-norm random gaussian targets
            vecs = rng.standard_normal((n, self.target_dim)).astype("float32")
            norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-8)
            self.random_targets = [
                torch.from_numpy(vecs[i] / norms[i]).float() for i in range(n)
            ]

    def __len__(self):
        return len(self.valid_records)

    def _load_stimulus(self, image_id: str) -> torch.Tensor:
        """Load a stimulus image as a [3, stim_size, stim_size] tensor in [0, 1].

        Returns a mid-gray tensor if the image file is missing so training does not
        crash on a single bad id (defensive; missing files are rare).
        """
        for ext in (".JPEG", ".jpg", ".png"):
            p = self.stimuli_dir / f"{image_id}{ext}"
            if p.exists():
                img = Image.open(p).convert("RGB").resize((self.stim_size, self.stim_size))
                return TF.to_tensor(img)
        return torch.full((3, self.stim_size, self.stim_size), 0.5)

    def __getitem__(self, idx):
        record = self.valid_records[idx]
        s_id = record["sample_id"]

        if self.use_split_files:
            latent = self.layer_dict[s_id].float()
        else:
            latent = record[self.layer_name].float()

        # Apply temporal window (spatial reshape then time slice)
        if self.use_temporal_window:
            latent = crop_zuna_latent(
                latent, self.n_channels, self.tc, self.latent_dim,
                self.latent_tc_start, self.latent_tc_end
            )

        # The "true" target is always the real image embedding — used for eval
        true_target = self.image_id_to_target[record["image_id"]].float()

        # The training target may be shuffled/random — used only for loss
        if self.target_mode == "real":
            train_target = true_target
        elif self.target_mode == "shuffled":
            perm_idx = self.target_perm[idx]
            perm_record = self.valid_records[perm_idx]
            train_target = self.image_id_to_target[perm_record["image_id"]].float()
        elif self.target_mode == "random":
            train_target = self.random_targets[idx]
        else:
            raise ValueError(f"Unknown target_mode: {self.target_mode}")

        return {
            "latent": latent,
            "target": train_target,       # fake target for loss (real = same as eval_target)
            "eval_target": true_target,    # always the true image embedding for retrieval eval
            "subject_id": torch.tensor(
                self.subject_id_to_index[int(record["subject_id"])], dtype=torch.long
            ),
            "run_id": record["run_id"],
            "image_id": record["image_id"],
            "image_int_id": torch.tensor(
                self.image_id_to_int[str(record["image_id"])], dtype=torch.long
            ),
            "class_id": record["class_id"],
            "sample_id": record["sample_id"],
            # Stimulus image for luminance grounding (mid-gray placeholder when disabled).
            "stimulus": (
                self._load_stimulus(record["image_id"])
                if self.stimuli_dir is not None
                else torch.zeros(1)
            ),
        }

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_epoch(model, loader, optimizer, temperature, target_std, device,
                rae_backend=None, recon_luma_weight=0.0, recon_grid_px=3,
                nce_weight=1.0, cos_weight=0.2, var_weight=0.05,
                spread_weight=1.0, spread_gamma=None,
                soft_dino_weight=0.0, soft_dino_teacher_temp=0.07,
                soft_dino_rkd_weight=0.0, soft_dino_hard_weight=0.0,
                negative_bank=None):
    model.train()
    total_loss = 0
    num_batches = 0
    use_recon = rae_backend is not None and recon_luma_weight > 0.0

    for batch in loader:
        latents = batch["latent"].to(device)
        targets = batch["target"].to(device)
        subject_ids = batch["subject_id"].to(device)
        image_int_ids = batch["image_int_id"].to(device)

        optimizer.zero_grad()
        if use_recon:
            preds, pred_grid = model(latents, subject_id=subject_ids, return_grid=True)
        else:
            preds = model(latents, subject_id=subject_ids)

        # InfoNCE + Cosine + Variance Floor Loss
        pred_u = F.normalize(preds, dim=-1)
        target_u = F.normalize(targets, dim=-1)

        # Extra negatives from the MoCo-style queue (detached, no grad), plus
        # same-image false-negative masking within the batch and the queue.
        queue_targets = None
        queue_image_ids = None
        if negative_bank is not None and len(negative_bank) > 0:
            queue_targets, queue_image_ids = negative_bank.get()

        loss_nce = clip_contrastive_loss(
            pred_u, target_u, temperature=temperature,
            image_ids=image_int_ids,
            queue_targets=queue_targets,
            queue_image_ids=queue_image_ids,
        )
        if soft_dino_weight > 0.0:
            # Relational loss on DINO geometry: reward predicting embeddings in the
            # right *neighborhood* of visual space instead of pinpoint identity.
            loss_soft = soft_dino_contrastive_loss(
                pred_u, target_u,
                temperature=temperature,
                teacher_temperature=soft_dino_teacher_temp,
                rkd_weight=soft_dino_rkd_weight,
                hard_weight=soft_dino_hard_weight,
                image_ids=image_int_ids,
            )
        else:
            loss_soft = torch.zeros((), device=device)
        loss_cos = 0.5 * (1.0 - F.cosine_similarity(pred_u, target_u, dim=-1).mean())
        loss_var = variance_floor_loss(preds, target_std.to(device))
        # Cross-sample spread on the normalized preds: the effective anti-hub-collapse
        # term (variance_floor is blind to directional collapse under force_unit_output).
        loss_spread = batch_std_hinge_loss(pred_u, spread_gamma.to(device))

        loss = (nce_weight * loss_nce + cos_weight * loss_cos
                + var_weight * loss_var + spread_weight * loss_spread
                + soft_dino_weight * loss_soft)

        # Enqueue this batch's (detached) real targets for future negatives.
        if negative_bank is not None:
            negative_bank.enqueue(target_u.detach(), image_int_ids.detach())

        # Luminance grounding: decode the predicted RAE token grid to an image and
        # match its global + 3x3 region luminance to the stimulus image. This is the
        # basic "visual test" that grounds the translator on scene illumination.
        if use_recon:
            stimulus = batch["stimulus"].to(device)  # [B, 3, H, W] in [0, 1]
            gen_img = rae_backend.decode_differentiable(pred_grid)  # [B, 3, H', W']
            if gen_img.shape[-2:] != stimulus.shape[-2:]:
                stimulus = F.interpolate(
                    stimulus, size=gen_img.shape[-2:], mode="bilinear", align_corners=False
                )
            loss_luma = luminance_grid_loss(gen_img.float(), stimulus.float(), grid=recon_grid_px)
            loss = loss + recon_luma_weight * loss_luma
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
    return total_loss / max(num_batches, 1)

@torch.no_grad()
def evaluate_model(model, loader, device, bank=None, image_id_to_bank_idx=None):
    model.eval()
    all_preds = []
    all_targets = []
    all_image_ids = []

    for batch in loader:
        latents = batch["latent"].to(device)
        # Bug 2 fix: always eval against the true image target, never the shuffled/random one
        targets = batch["eval_target"].to(device)
        subject_ids = batch["subject_id"].to(device)

        preds = model(latents, subject_id=subject_ids)

        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
        all_image_ids.extend(list(batch["image_id"]))

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute normalized evaluation metrics
    pred_eval = F.normalize(all_preds, dim=-1)
    target_eval = F.normalize(all_targets, dim=-1)
    
    metrics = retrieval_topk(pred_eval, target_eval)
    
    # Extract metric values
    mrr_norm = metrics["mrr"]
    top1_norm = metrics["top1"]
    top5_norm = metrics["top5"]
    top10_norm = metrics["top10"]
    cosine_norm = F.cosine_similarity(pred_eval, target_eval, dim=-1).mean().item()
    
    # Norm statistics
    pred_norms = all_preds.norm(dim=-1)
    val_pred_norm_mean = pred_norms.mean().item()
    val_pred_norm_std = pred_norms.std().item()
    
    # Pred std ratio (raw)
    pred_std = all_preds.std(dim=0).mean().item()
    target_std = all_targets.std(dim=0).mean().item()
    val_pred_std_ratio = pred_std / max(target_std, 1e-8)
    
    # collapse_pct
    pred_dims_std = all_preds.std(dim=0)
    target_dims_std = all_targets.std(dim=0)
    ratio = pred_dims_std / (target_dims_std + 1e-8)
    collapse_pct = float((ratio < 0.2).float().mean().item()) * 100.0
    
    eval_metrics = {
        "val_mrr_norm": mrr_norm,
        "val_top1_norm": top1_norm,
        "val_top5_norm": top5_norm,
        "val_top10_norm": top10_norm,
        "val_cosine_norm": cosine_norm,
        "val_pred_std_ratio": val_pred_std_ratio,
        "val_pred_norm_mean": val_pred_norm_mean,
        "val_pred_norm_std": val_pred_norm_std,
        "collapse_pct": collapse_pct,
        # backward compatibility keys for training loops
        "mrr": mrr_norm,
        "top1": top1_norm,
        "top10": top10_norm,
        "cosine": cosine_norm,
        "collapse_score": val_pred_std_ratio,
        "pred_std": pred_std,
        "target_std": target_std
    }

    # Finding H1: full-bank retrieval is the honest metric. Rank each prediction
    # against the entire unique-image bank (not just the val set). within-val
    # metrics above are kept but clearly labelled as diagnostic/inflated.
    if bank is not None and image_id_to_bank_idx is not None:
        positive_index = torch.tensor(
            [image_id_to_bank_idx[img_id] for img_id in all_image_ids], dtype=torch.long
        )
        full = retrieval_topk_full_bank(all_preds, bank, positive_index)
        eval_metrics.update({
            "val_mrr_full": full["mrr"],
            "val_top1_full": full["top1"],
            "val_top5_full": full["top5"],
            "val_top10_full": full["top10"],
            "val_median_rank_full": full["median_rank"],
            "bank_size": full["bank_size"],
        })
    return eval_metrics

@torch.no_grad()
def save_eval_metadata(model, loader, device, out_path, bank=None, image_id_to_bank_idx=None):
    model.eval()
    all_preds = []
    all_targets = []
    sample_ids = []
    image_ids = []

    for batch in loader:
        latents = batch["latent"].to(device)
        # Bug 2 fix: always rank against true image targets
        targets = batch["eval_target"].to(device)
        subject_ids = batch["subject_id"].to(device)

        preds = model(latents, subject_id=subject_ids)

        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
        sample_ids.extend(batch["sample_id"])
        image_ids.extend(batch["image_id"])
        
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Within-val rank (diagnostic / inflated): rank against the other val samples.
    pred_n = F.normalize(all_preds, dim=-1)
    tgt_n = F.normalize(all_targets, dim=-1)
    
    logits = pred_n @ tgt_n.T  # [N, N]
    n = pred_n.shape[0]
    truth = torch.arange(n)
    
    sorted_indices = logits.argsort(dim=-1, descending=True)
    rank_of_truth = (sorted_indices == truth[:, None]).nonzero(as_tuple=False)[:, 1].float()
    top10_hit = (rank_of_truth < 10).float()
    
    data = {
        "sample_id": sample_ids,
        "image_id": image_ids,
        "pred": all_preds,
        "target": all_targets,
        # within-val (diagnostic, inflated) rank kept for backward compatibility
        "rank": rank_of_truth,
        "top10_hit": top10_hit,
    }

    # Finding H1: also store the honest full-bank rank/hit so the grid gate can
    # consume the metric the docs mandate.
    if bank is not None and image_id_to_bank_idx is not None:
        positive_index = torch.tensor(
            [image_id_to_bank_idx[img_id] for img_id in image_ids], dtype=torch.long
        )
        bank_n = F.normalize(bank, dim=-1)
        full_logits = pred_n @ bank_n.T  # [N, M]
        pos_sim = full_logits.gather(1, positive_index[:, None]).squeeze(1)
        rank_full = (full_logits > pos_sim[:, None]).sum(dim=1).float()
        data["rank_full"] = rank_full
        data["top10_hit_full"] = (rank_full < 10).float()
        data["bank_size"] = int(bank.shape[0])

    torch.save(data, out_path)
    print(f"Saved evaluation metadata to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Train QFormer adapter mapping ZUNA latents to vision space.")
    # Dataset and paths
    parser.add_argument("--latents-pt", type=str, required=True, help="Path to cached latents.pt")
    parser.add_argument("--targets-pt", type=str, required=True, help="Path to targets.pt")
    parser.add_argument("--target-space", type=str, default="common", help="Target space name inside targets.pt")
    parser.add_argument("--target-mode", choices=("real", "shuffled", "random"), default="real", help="Mapping mode (controls)")
    parser.add_argument("--layer-name", type=str, default="post_mmd", help="Source ZUNA layer")
    
    # Explicit splits
    parser.add_argument("--train-runs", type=str, default=None, help="Train run list/range, e.g. 1-24")
    parser.add_argument("--val-runs", type=str, default=None, help="Val run list/range, e.g. 25-28")
    parser.add_argument("--test-runs", type=str, default=None, help="Test run list/range, e.g. 29-32")
    parser.add_argument("--subjects", type=str, default=None, help="Comma-separated subject IDs (e.g. 'sub-01') to filter")
    parser.add_argument("--require-image-disjoint", action="store_true",
                        help="Hard-fail if train shares any image_id with val/test (finding M3).")

    # Temporal windowing
    parser.add_argument("--temporal-window", action="store_true", default=True,
                        help="Slice post_mmd tokens to event window")
    parser.add_argument("--no-temporal-window", action="store_false", dest="temporal_window")
    parser.add_argument("--latent-tc-start", type=int, default=20, help="Latent time slice start index")
    parser.add_argument("--latent-tc-end", type=int, default=36, help="Latent time slice end index (exclusive)")
    
    # QFormer architecture
    parser.add_argument("--num-query-tokens", type=int, default=32, help="Number of query tokens")
    parser.add_argument("--pooling-mode", choices=("cls", "attention", "mean"), default="cls", help="Query pooling mode")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="QFormer hidden dimension")
    parser.add_argument("--nhead", type=int, default=8, help="QFormer attention heads")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of QFormer layers")
    parser.add_argument("--dropout", type=float, default=0.15, help="QFormer dropout")
    parser.add_argument("--num-subjects", type=int, default=1, help="Number of subjects for FiLM embeddings")
    
    # Output stabilization head flags
    parser.add_argument("--output-layernorm", action="store_true", default=True, help="Enable LayerNorm in final head")
    parser.add_argument("--no-output-layernorm", action="store_false", dest="output_layernorm")
    parser.add_argument("--force-unit-output", action="store_true", default=True, help="Enable L2 normalization in final head")
    parser.add_argument("--no-force-unit-output", action="store_false", dest="force_unit_output")

    # Reconstruction / luminance grounding (opt-in; overrides HANDOVER non-negotiable #3
    # by explicit decision — see docs/HANDOVER.md). When --recon-luma-weight > 0 the QFormer
    # predicts an RAE token grid, decodes it through the frozen RAE, and matches the decoded
    # image's global + region luminance to the stimulus image.
    parser.add_argument("--recon-luma-weight", type=float, default=0.0,
                        help="Weight of the stimulus-vs-generated luminance-grid loss. 0 disables reconstruction (pure retrieval).")
    parser.add_argument("--recon-grid-px", type=int, default=3, help="Spatial luminance grid size (3 -> 3x3 regions + global)")
    parser.add_argument("--recon-token-dim", type=int, default=768, help="RAE token dim for the reconstruction grid head")
    parser.add_argument("--recon-grid-size", type=int, default=16, help="RAE token grid size G (G x G tokens)")
    parser.add_argument("--stimuli-dir", type=str, default="data/raw/nod/stimuli/ImageNet",
                        help="Directory of stimulus images (used only when --recon-luma-weight > 0)")
    parser.add_argument("--rae-model-id", type=str, default="nyu-visionx/RAE-dinov2-wReg-base-ViTXL-n08",
                        help="RAE decoder model id for differentiable decode")

    # Optimization
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size (larger = more in-batch InfoNCE negatives; A100 has ample headroom)")
    parser.add_argument("--temperature", type=float, default=0.05, help="Contrastive InfoNCE temperature")

    # Loss weighting (fine-tuning knobs). The cosine term pulls each prediction
    # toward its own target (hubness pressure) and competes with InfoNCE's
    # separation; its default is lowered from the old hard-coded 1.0 to 0.2 to
    # soften that pull. Pass --cos-weight 1.0 to recover the previous loss.
    parser.add_argument("--nce-weight", type=float, default=1.0, help="Weight on the InfoNCE contrastive term")
    parser.add_argument("--cos-weight", type=float, default=0.2, help="Weight on the per-sample cosine pull (was 1.0)")
    parser.add_argument("--var-weight", type=float, default=0.05, help="Weight on the variance-floor anti-collapse term")
    parser.add_argument("--spread-weight", type=float, default=1.0,
                        help="Weight on the VICReg-style cross-sample spread term on normalized preds "
                             "(effective anti-hub-collapse; 0 recovers the old blind-floor-only behavior)")
    parser.add_argument("--soft-dino-weight", type=float, default=0.0,
                        help="Weight on the relational soft-DINO contrastive loss (soft-target InfoNCE "
                             "using the DINO teacher similarity matrix instead of hard image-id labels; "
                             "0 = off, pure hard InfoNCE)")
    parser.add_argument("--soft-dino-teacher-temp", type=float, default=0.07,
                        help="Teacher softmax temperature for soft-DINO labels; lower=sharper (closer to "
                             "hard InfoNCE), higher=smoother visual neighborhoods")
    parser.add_argument("--soft-dino-rkd-weight", type=float, default=0.0,
                        help="Weight on the RKD relational term inside soft-DINO loss (match pred vs DINO "
                             "similarity-matrix geometry directly)")
    parser.add_argument("--soft-dino-hard-weight", type=float, default=0.0,
                        help="Blend of hard-label InfoNCE inside soft-DINO loss for a soft/hard curriculum "
                             "(0 = pure soft)")

    # Extra negatives: MoCo-style FIFO queue of detached real target embeddings
    # appended to the InfoNCE denominator. 0 disables (current behavior). Only
    # active in the real target mode (queue ids must align with targets).
    parser.add_argument("--negative-bank-size", type=int, default=0,
                        help="Size K of the negative queue of past target embeddings (0 = off). Real mode only.")
    
    # Output and execution
    parser.add_argument("--device", type=str, default="cuda", help="Torch device")
    parser.add_argument("--out-dir", type=str, default="outputs/qformer_aligned_grid", help="Directory to save checkpoints and logs")
    parser.add_argument("--slug", type=str, default=None, help="Optional experiment slug")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # The full-image-bank retrieval (34k images) is a near-chance-by-construction
    # image-id metric and is expensive to run every epoch. Judge signal by vector
    # distance (val_cosine_norm) instead; only run the bank retrieval once at the
    # end for reference. "each" preserves the old per-epoch behavior.
    parser.add_argument("--full-bank-eval", choices=("none", "final", "each"), default="final",
                        help="When to rank preds against the full image bank (default: final only)")
    parser.add_argument("--select-metric", choices=("cosine", "mrr_full", "mrr_norm"), default="cosine",
                        help="Checkpoint/early-stop selection metric (default: val_cosine_norm)")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Parse subjects list
    subject_list = None
    if args.subjects:
        subject_list = [s.strip() for s in args.subjects.split(",") if s.strip()]
        
    # Map target spaces dynamically
    is_pca = "PCA" in args.target_space
    pca_dims = None
    target_space_key = args.target_space
    if is_pca:
        pca_dims = int(args.target_space.split("-")[2])
        target_space_key = "rae_unit"
    elif args.target_space in ("DINO-CLS-768", "DINO-CLS"):
        target_space_key = "dino_cls"
    elif args.target_space in ("DINO-Unit-768", "DINO-Unit"):
        target_space_key = "rae_unit"
    elif args.target_space in ("CLIP-Common-512", "CLIP-Common"):
        target_space_key = "common"

    # Load dataset
    use_recon = args.recon_luma_weight > 0.0
    full_dataset = ZunaLatentTargetDataset(
        latents_pt_path=args.latents_pt,
        targets_pt_path=args.targets_pt,
        target_space=args.target_space,
        layer_name=args.layer_name,
        target_mode=args.target_mode,
        shuffle_seed=args.seed,
        subject_list=subject_list,
        use_temporal_window=args.temporal_window,
        latent_tc_start=args.latent_tc_start,
        latent_tc_end=args.latent_tc_end,
        stimuli_dir=args.stimuli_dir if use_recon else None,
    )
    
    # Explicit splits determination
    all_runs = sorted(list({int(r["run_id"]) for r in full_dataset.records}))
    max_run = max(all_runs) if all_runs else 32
    
    if args.train_runs:
        train_run_ids = parse_runs_spec(args.train_runs)
    else:
        train_run_ids = list(range(1, 25)) if max_run <= 32 else list(range(1, 33))
        
    if args.val_runs:
        val_run_ids = parse_runs_spec(args.val_runs)
    else:
        val_run_ids = list(range(25, 29)) if max_run <= 32 else list(range(33, 37))
        
    if args.test_runs:
        test_run_ids = parse_runs_spec(args.test_runs)
    else:
        test_run_ids = list(range(29, 33)) if max_run <= 32 else list(range(37, 41))

    train_indices = []
    val_indices = []
    test_indices = []
    
    for idx in range(len(full_dataset)):
        record = full_dataset.valid_records[idx]
        run_id = int(record["run_id"])
        if run_id in train_run_ids:
            train_indices.append(idx)
        elif run_id in val_run_ids:
            val_indices.append(idx)
        elif run_id in test_run_ids:
            test_indices.append(idx)
            
    print(f"Available dataset runs: {all_runs}")
    print(f"Split: Train={len(train_indices)} samples | Val={len(val_indices)} samples | Test={len(test_indices)} samples")
    print(f"Train runs: {train_run_ids}")
    print(f"Val runs: {val_run_ids}")
    print(f"Test runs: {test_run_ids}")
    
    if len(train_indices) == 0 or len(val_indices) == 0:
        raise ValueError(f"Invalid split: train={len(train_indices)} val={len(val_indices)}.")

    # Setup directories early to write PCA params
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{ts}_{args.target_mode}_{args.layer_name}_{args.target_space.replace('/', '_')}"
    if args.slug:
        experiment_name += f"_{args.slug}"
        
    run_dir = Path(args.out_dir) / experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoints and metrics to {run_dir}")

    # Fit PCA if target space is a PCA target
    if is_pca:
        train_image_ids = set()
        for idx in train_indices:
            rec = full_dataset.valid_records[idx]
            train_image_ids.add(rec["image_id"])
            
        train_image_ids = sorted(list(train_image_ids))
        train_targets = torch.stack([full_dataset.image_id_to_target[img_id] for img_id in train_image_ids])
        
        print(f"Fitting PCA with {full_dataset.pca_dims} components on {len(train_image_ids)} training images...")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=full_dataset.pca_dims)
        pca.fit(train_targets.numpy())
        
        # Transform all targets
        all_image_ids = list(full_dataset.image_id_to_target.keys())
        all_targets = torch.stack([full_dataset.image_id_to_target[img_id] for img_id in all_image_ids])
        all_transformed = pca.transform(all_targets.numpy())
        
        # Normalize transformed targets
        all_transformed_tensor = torch.from_numpy(all_transformed).float()
        all_transformed_tensor = F.normalize(all_transformed_tensor, dim=-1)
        
        # Update targets dict
        full_dataset.image_id_to_target = {
            img_id: all_transformed_tensor[i] for i, img_id in enumerate(all_image_ids)
        }
        
        # Save PCA params
        pca_params = {
            "mean": torch.from_numpy(pca.mean_).float(),
            "components": torch.from_numpy(pca.components_).float(),
            "explained_variance": torch.from_numpy(pca.explained_variance_).float(),
        }
        torch.save(pca_params, run_dir / "pca_params.pt")
        print(f"Saved PCA parameters to {run_dir / 'pca_params.pt'}")

    # Build the full unique-image target bank (finding H1). After any PCA transform
    # above, image_id_to_target holds the final target vectors. Dedup by image_id so
    # each image appears once; predictions are ranked against this whole bank.
    bank_image_ids = list(full_dataset.image_id_to_target.keys())
    image_id_to_bank_idx = {img_id: i for i, img_id in enumerate(bank_image_ids)}
    full_bank = torch.stack(
        [full_dataset.image_id_to_target[img_id].float() for img_id in bank_image_ids]
    )
    print(f"Full retrieval bank: {full_bank.shape[0]} unique images, dim {full_bank.shape[1]}")

    # Finding M3: quantify (and optionally forbid) train/val/test image overlap.
    def _img_set(indices):
        return {full_dataset.valid_records[i]["image_id"] for i in indices}

    train_imgs = _img_set(train_indices)
    val_imgs = _img_set(val_indices)
    test_imgs = _img_set(test_indices)
    tv_overlap = len(train_imgs & val_imgs)
    tt_overlap = len(train_imgs & test_imgs)
    print(
        f"[split] image overlap — train∩val={tv_overlap} "
        f"({100.0 * tv_overlap / max(len(val_imgs), 1):.1f}% of val images), "
        f"train∩test={tt_overlap} "
        f"({100.0 * tt_overlap / max(len(test_imgs), 1):.1f}% of test images)"
    )
    if args.require_image_disjoint and (tv_overlap > 0 or tt_overlap > 0):
        raise ValueError(
            f"--require-image-disjoint set but train shares {tv_overlap} image(s) with val and "
            f"{tt_overlap} with test. Full-bank retrieval would be optimistic. Use an "
            f"image-disjoint split or drop the flag to accept the leakage."
        )

    # Compute target_std from train split targets
    train_targets_list = []
    for idx in train_indices:
        train_targets_list.append(full_dataset[idx]["target"])
    train_targets_all = torch.stack(train_targets_list)
    target_std = train_targets_all.std(dim=0).clamp_min(1e-4)
    print(f"Calculated training target_std (mean={target_std.mean().item():.6f}, min={target_std.min().item():.6f}, max={target_std.max().item():.6f})")
    # Per-dim std of the *normalized* train targets: the anchor the spread hinge
    # aims for, so the anti-collapse term switches off once preds match the
    # target's own spread on the sphere (rather than the unreachable 1/sqrt(D)).
    spread_gamma = F.normalize(train_targets_all, dim=-1).std(dim=0).clamp_min(1e-6)
    print(f"Spread hinge gamma (normalized-target std): mean={spread_gamma.mean().item():.6f}")

    # Subset datasets and dataloaders
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    # Save training config with split info
    config_dict = vars(args)
    config_dict.update({
        "available_runs": all_runs,
        "train_runs": train_run_ids,
        "val_runs_list": val_run_ids,
        "test_runs_list": test_run_ids
    })
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
        
    # Finding M2: size the FiLM table from the actual cohort, not a possibly-wrong
    # --num-subjects. The dataset remaps raw subject ids to contiguous 0..K-1 indices.
    effective_num_subjects = full_dataset.num_subjects
    if args.num_subjects != effective_num_subjects:
        print(
            f"[warn] --num-subjects={args.num_subjects} but cohort has "
            f"{effective_num_subjects} distinct subject(s); using {effective_num_subjects} for FiLM."
        )

    # Instantiate stabilized QFormer adapter
    model = ZunaToVisionQFormer(
        d_in=full_dataset.latent_dim,
        d_out=full_dataset.target_dim,
        hidden_dim=args.hidden_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        num_query_tokens=args.num_query_tokens,
        pooling_mode=args.pooling_mode,
        dropout=args.dropout,
        num_subjects=effective_num_subjects,
        output_layernorm=args.output_layernorm,
        force_unit_output=args.force_unit_output,
        normalize_output=False, # disable old L2 normalizer
        recon_grid=use_recon,
        recon_grid_size=args.recon_grid_size,
        recon_token_dim=args.recon_token_dim,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized ZunaToVisionQFormer with {num_params:,} trainable parameters.")

    # Frozen RAE decoder for the luminance-grounding loss (only when reconstruction is on).
    rae_backend = None
    if use_recon:
        from mindseye.generation.rae_backend import RaeDecoderBackend
        print(f"Reconstruction enabled (luma weight={args.recon_luma_weight}). Loading frozen RAE decoder...")
        rae_backend = RaeDecoderBackend(model_id=args.rae_model_id, device=str(device), apply_patch=True)
        rae_backend.load()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Negative queue (MoCo-style extra negatives). Only valid in the real target
    # mode, where the enqueued target aligns with its image id; in shuffled/random
    # the training target belongs to a different image, so masking would be wrong.
    negative_bank = None
    if args.negative_bank_size > 0:
        if args.target_mode == "real":
            negative_bank = NegativeBank(args.negative_bank_size, full_dataset.target_dim, device)
            print(f"Negative queue enabled: size={args.negative_bank_size}, dim={full_dataset.target_dim}")
        else:
            print(
                f"[warn] --negative-bank-size={args.negative_bank_size} ignored in "
                f"target_mode={args.target_mode} (queue is real-mode only)."
            )

    print(
        f"Loss weights: nce={args.nce_weight} cos={args.cos_weight} var={args.var_weight} "
        f"spread={args.spread_weight} soft_dino={args.soft_dino_weight} "
        f"(teacher_temp={args.soft_dino_teacher_temp} rkd={args.soft_dino_rkd_weight} "
        f"hard={args.soft_dino_hard_weight}) | temperature={args.temperature}"
    )

    # Training loop
    best_mrr = -1.0
    history = []
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, args.temperature, target_std, device,
            rae_backend=rae_backend, recon_luma_weight=args.recon_luma_weight,
            recon_grid_px=args.recon_grid_px,
            nce_weight=args.nce_weight, cos_weight=args.cos_weight, var_weight=args.var_weight,
            spread_weight=args.spread_weight, spread_gamma=spread_gamma,
            soft_dino_weight=args.soft_dino_weight,
            soft_dino_teacher_temp=args.soft_dino_teacher_temp,
            soft_dino_rkd_weight=args.soft_dino_rkd_weight,
            soft_dino_hard_weight=args.soft_dino_hard_weight,
            negative_bank=negative_bank,
        )
        scheduler.step()
        eval_bank = full_bank if args.full_bank_eval == "each" else None
        eval_bank_idx = image_id_to_bank_idx if args.full_bank_eval == "each" else None
        val_metrics = evaluate_model(
            model, val_loader, device,
            bank=eval_bank, image_id_to_bank_idx=eval_bank_idx,
        )
        
        val_mrr = val_metrics["val_mrr_norm"]
        val_top10 = val_metrics["val_top10_norm"]
        val_cosine = val_metrics["val_cosine_norm"]
        std_ratio = val_metrics["val_pred_std_ratio"]
        collapse_pct = val_metrics["collapse_pct"]
        val_mrr_full = val_metrics.get("val_mrr_full", float("nan"))
        val_top10_full = val_metrics.get("val_top10_full", float("nan"))

        # Selection metric. Default is vector distance (val_cosine_norm): the
        # full-bank image-id retrieval is near-chance-by-construction and no
        # longer the per-epoch gate. cosine measures how close preds land to the
        # true target direction, which is the honest signal indicator here.
        if args.select_metric == "cosine":
            select_metric = val_cosine
        elif args.select_metric == "mrr_full" and "val_mrr_full" in val_metrics:
            select_metric = val_metrics["val_mrr_full"]
        else:
            if args.select_metric == "mrr_full":
                print("[warn] select-metric=mrr_full but full-bank eval disabled; using val_mrr_norm")
            select_metric = val_mrr
        
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | "
              f"Val Cosine (Norm): {val_cosine:.4f} | Val MRR (Norm): {val_mrr:.4f} | "
              f"Val Top-10 (Norm): {val_top10:.4f} | "
              f"MRR(full): {val_mrr_full:.4f} | Top-10(full): {val_top10_full:.4f} | "
              f"StdRatio: {std_ratio:.3f} | Collapse: {collapse_pct:.1f}%")
              
        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics
        }
        history.append(history_row)
        
        # Save checkpoint if the selection metric (full-bank MRR) improves
        is_best = select_metric > best_mrr
        if is_best:
            best_mrr = select_metric
            patience_counter = 0
            checkpoint_path = run_dir / "checkpoint_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mrr": val_mrr,
                "val_mrr_full": val_mrr_full,
                "select_metric": select_metric,
                "config": config_dict
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⏹ Early stopping at epoch {epoch} (patience={args.patience}, best MRR={best_mrr:.4f})")
                break
            
    # Save final model state
    torch.save(model.state_dict(), run_dir / "model_final.pt")
    
    # Save history to CSV
    df_history = pd.DataFrame(history)
    df_history.to_csv(run_dir / "history.csv", index=False)
    
    # Save final metrics summary. The "best" row reflects the saved checkpoint,
    # selected on the configured selection metric.
    final_metrics = history[-1]
    sel_col = {"cosine": "val_cosine_norm", "mrr_full": "val_mrr_full", "mrr_norm": "val_mrr_norm"}[args.select_metric]
    if sel_col in df_history.columns and df_history[sel_col].notna().any():
        best_epoch_idx = df_history[sel_col].idxmax()
    else:
        print(f"[warn] selection metric {sel_col} absent — picking best row on val_cosine_norm")
        best_epoch_idx = df_history["val_cosine_norm"].idxmax()
    best_metrics = history[best_epoch_idx]

    summary = {
        "final": final_metrics,
        "best": best_metrics
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\n✓ Training complete! Best {sel_col}={best_mrr:.4f} at epoch {best_metrics['epoch']}.")

    # Bug 1 fix: reload best checkpoint before saving eval predictions
    best_ckpt = torch.load(run_dir / "checkpoint_best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()
    print(f"Reloaded best checkpoint (epoch {best_ckpt['epoch']}) for eval prediction saving.")

    # Full-bank retrieval is reported once here (unless disabled) rather than
    # every epoch, so the expensive 34k-image ranking doesn't dominate the grid.
    final_bank = None if args.full_bank_eval == "none" else full_bank
    final_bank_idx = None if args.full_bank_eval == "none" else image_id_to_bank_idx
    save_eval_metadata(
        model, val_loader, device, run_dir / "val_eval_preds.pt",
        bank=final_bank, image_id_to_bank_idx=final_bank_idx,
    )
    if len(test_dataset) > 0:
        save_eval_metadata(
            model, test_loader, device, run_dir / "test_eval_preds.pt",
            bank=final_bank, image_id_to_bank_idx=final_bank_idx,
        )

    print(f"Results saved in: {run_dir}")

if __name__ == "__main__":
    main()

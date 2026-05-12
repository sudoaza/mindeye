"""Baseline EEG encoder models for mapping ZUNA crops to CLIP space."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class EEGClipEncoder(nn.Module):
    """
    Small temporal-convolution EEG encoder.

    Input shape is `[batch, channels, time]`; output is a CLIP-sized vector.
    This is intentionally modest so it can serve as a Phase 4 baseline before
    trying larger subject-aware or transformer architectures.
    """

    def __init__(
        self,
        *,
        n_channels: int = 62,
        n_times: int = 321,
        embedding_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        normalize_output: bool = True,
    ):
        super().__init__()
        self.normalize_output = normalize_output
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        x = self.net(eeg)
        x = self.head(x)
        if self.normalize_output:
            x = F.normalize(x, dim=-1)
        return x


def cosine_mse_loss(pred: torch.Tensor, target: torch.Tensor, *, mse_weight: float = 0.25) -> torch.Tensor:
    """Blend cosine embedding loss with a small MSE term for stable baseline training."""
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    cosine = 1.0 - (pred * target).sum(dim=-1).mean()
    mse = F.mse_loss(pred, target)
    return cosine + mse_weight * mse


def clip_contrastive_loss(pred: torch.Tensor, target: torch.Tensor, *, temperature: float = 0.07) -> torch.Tensor:
    """Symmetric CLIP-style InfoNCE loss for paired EEG and image embeddings.

    Direct cosine/MSE regression can collapse toward a generic CLIP-space hub: all
    examples are only pulled toward their own target, with no explicit pressure to
    separate the other images in the batch.  This loss treats the diagonal as the
    positive pairs and all off-diagonal items in the batch as negatives, matching
    the usual CLIP training objective.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have matching shape, got {pred.shape} and {target.shape}")
    if pred.shape[0] < 2:
        raise ValueError("contrastive loss needs at least two items per batch")
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = pred @ target.T / temperature
    labels = torch.arange(pred.shape[0], device=pred.device)
    eeg_to_img = F.cross_entropy(logits, labels)
    img_to_eeg = F.cross_entropy(logits.T, labels)
    return 0.5 * (eeg_to_img + img_to_eeg)


def retrieval_topk(
    pred: torch.Tensor,
    targets: torch.Tensor,
    *,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Compute retrieval metrics against a full validation target bank.

    Returns top-k accuracy, MRR, median rank, off-diagonal cosine mean, and
    collapse score (pred_std / target_std) — all required by the baseline matrix.
    """
    pred_n = F.normalize(pred, dim=-1)
    tgt_n = F.normalize(targets, dim=-1)
    logits = pred_n @ tgt_n.T  # [N, N]
    n = pred.shape[0]
    truth = torch.arange(n, device=pred.device)

    # Sort descending once; cheapest approach for a single pass
    sorted_indices = logits.argsort(dim=-1, descending=True)  # [N, N]
    # rank of the correct target for each query (0-based)
    rank_of_truth = (sorted_indices == truth[:, None]).nonzero(as_tuple=False)[:, 1].float()

    out: dict[str, float] = {}
    for k in ks:
        out[f"top{k}"] = (rank_of_truth < k).float().mean().item()

    out["mrr"] = (1.0 / (rank_of_truth + 1.0)).mean().item()
    out["median_rank"] = float(rank_of_truth.median().item() + 1)  # 1-indexed

    # Off-diagonal cosine: mean similarity to all *wrong* targets
    diag_mask = torch.eye(n, dtype=torch.bool, device=pred.device)
    off_diag = logits[~diag_mask]  # [(N*(N-1))] elements
    out["off_diag_cosine"] = float(off_diag.mean().item())

    # Collapse score: pred_std / target_std (1.0 = same spread as targets)
    pred_std = float(pred.std(dim=0).mean().item())
    tgt_std = float(targets.std(dim=0).mean().item())
    out["pred_std"] = pred_std
    out["target_std"] = tgt_std
    out["collapse_score"] = pred_std / max(tgt_std, 1e-8)

    return out

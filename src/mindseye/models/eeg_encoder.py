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


def retrieval_topk(pred: torch.Tensor, targets: torch.Tensor, *, ks: tuple[int, ...] = (1, 5)) -> dict[str, float]:
    """Compute image-retrieval top-k accuracy against an in-batch/validation target bank."""
    pred = F.normalize(pred, dim=-1)
    targets = F.normalize(targets, dim=-1)
    logits = pred @ targets.T
    truth = torch.arange(pred.shape[0], device=pred.device)
    out: dict[str, float] = {}
    for k in ks:
        k_eff = min(k, targets.shape[0])
        topk = logits.topk(k_eff, dim=-1).indices
        out[f"top{k}"] = (topk == truth[:, None]).any(dim=-1).float().mean().item()
    return out

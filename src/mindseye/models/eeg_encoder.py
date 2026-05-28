"""Baseline EEG encoder models for mapping ZUNA crops to CLIP space."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import math


def _group_norm(num_channels: int, preferred_groups: int = 8) -> nn.GroupNorm:
    """GroupNorm helper that gracefully handles small/non-divisible widths."""
    groups = min(preferred_groups, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class EEGClipEncoder(nn.Module):
    """
    Standard temporal-convolution EEG encoder.
    """

    def __init__(
        self,
        *,
        n_channels: int = 62,
        n_times: int = 321,
        embedding_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        stem_dropout1d: float = 0.15,
        normalize_output: bool = True,
        num_subjects: int = 1,
    ):
        super().__init__()
        self.num_subjects = num_subjects
        self.normalize_output = normalize_output
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 128, kernel_size=7, padding=3),
            _group_norm(128),
            nn.GELU(),
            nn.Dropout1d(stem_dropout1d),
            nn.MaxPool1d(2),
            nn.Conv1d(128, hidden_dim, kernel_size=5, padding=2),
            _group_norm(hidden_dim),
            nn.GELU(),
            nn.Dropout1d(stem_dropout1d),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            _group_norm(hidden_dim),
            nn.GELU(),
            nn.Dropout1d(stem_dropout1d),
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

    def forward(self, eeg: torch.Tensor, return_features: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.net(eeg)
        features = torch.flatten(features, 1)  # [B, hidden_dim]
        x = self.head(features)
        if self.normalize_output:
            x = F.normalize(x, dim=-1)
        if return_features:
            return x, features
        return x


class AttentionPooler(nn.Module):
    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        b = x.shape[0]
        q = self.query.expand(b, -1, -1)
        out, _ = self.attn(q, x, x)
        return out.squeeze(1)


class TemporalAttnEncoder(nn.Module):
    """
    Lightweight Transformer-based encoder for longer EEG windows (e.g. 5s).
    Supports subject-specific FiLM conditioning and per-subject projection heads.
    """

    def __init__(
        self,
        *,
        n_channels: int = 62,
        embedding_dim: int = 512,
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.2,
        stem_dropout1d: float = 0.15,
        normalize_output: bool = True,
        num_subjects: int = 1,
        no_film: bool = False,
        no_subject_heads: bool = False,
        head_reg_weight: float = 0.0,
    ):
        super().__init__()
        self.num_subjects = num_subjects
        self.normalize_output = normalize_output
        self.no_film = no_film
        self.no_subject_heads = no_subject_heads
        self.head_reg_weight = head_reg_weight
        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, 128, kernel_size=7, stride=4, padding=3),
            _group_norm(128),
            nn.GELU(),
            nn.Dropout1d(stem_dropout1d),
            nn.Conv1d(128, hidden_dim, kernel_size=5, stride=2, padding=2),
            _group_norm(hidden_dim),
            nn.GELU(),
            nn.Dropout1d(stem_dropout1d),
        )

        self.max_tokens = 256
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_tokens, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pooler = AttentionPooler(hidden_dim, heads=n_heads)
        self.head = nn.Linear(hidden_dim, embedding_dim)

        if num_subjects > 1:
            # FiLM: zero-init so initial transform is identity
            self.subject_embed = nn.Embedding(num_subjects, hidden_dim * 2)
            nn.init.zeros_(self.subject_embed.weight)
            # Subject heads: warm-init from shared head (copy weights, not random)
            self.subject_heads = nn.ModuleList([
                nn.Linear(hidden_dim, embedding_dim) for _ in range(num_subjects)
            ])
            for h in self.subject_heads:
                h.load_state_dict(self.head.state_dict())
        else:
            self.subject_embed = None
            self.subject_heads = None

    def compute_head_reg(self) -> torch.Tensor:
        """L2 regularization: mean ||W_subject - W_shared||^2 over all subject heads."""
        if self.subject_heads is None or self.no_subject_heads:
            return torch.tensor(0.0)
        ref_w = self.head.weight.detach()
        ref_b = self.head.bias.detach()
        reg = sum(
            ((h.weight - ref_w) ** 2).mean() + ((h.bias - ref_b) ** 2).mean()
            for h in self.subject_heads
        )
        return reg / self.num_subjects

    def forward(
        self,
        eeg: torch.Tensor,
        subject_id: torch.Tensor | None = None,
        return_features: bool = False,
        return_head_reg: bool = False,
    ) -> torch.Tensor | tuple:
        # eeg: [B, C, T]
        x = self.stem(eeg)  # [B, D, T']
        x = x.transpose(1, 2)  # [B, T', D]

        # FiLM conditioning (skipped when no_film=True)
        use_film = (
            not self.no_film
            and getattr(self, "subject_embed", None) is not None
            and subject_id is not None
        )
        if use_film:
            film_params = self.subject_embed(subject_id)  # [B, D*2]
            gamma, beta = film_params.chunk(2, dim=-1)     # [B, D] each
            x = x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        t = x.shape[1]
        if t > self.max_tokens:
            raise ValueError(f"Token length {t} exceeds max_tokens={self.max_tokens}")
        x = x + self.pos_embed[:, :t, :]

        x = self.transformer(x)
        features = self.pooler(x)

        # Subject-specific projection heads (skipped when no_subject_heads=True)
        use_subj_heads = (
            not self.no_subject_heads
            and getattr(self, "subject_heads", None) is not None
            and subject_id is not None
        )
        if use_subj_heads:
            b = features.shape[0]
            out = torch.zeros(b, self.head.out_features, device=features.device, dtype=features.dtype)
            for sub_idx in range(self.num_subjects):
                mask = (subject_id == sub_idx)
                if mask.any():
                    out[mask] = self.subject_heads[sub_idx](features[mask])
            x = out
        else:
            x = self.head(features)

        if self.normalize_output:
            x = F.normalize(x, dim=-1)

        head_reg = self.compute_head_reg() if return_head_reg else None

        if return_features and return_head_reg:
            return x, features, head_reg
        if return_features:
            return x, features
        if return_head_reg:
            return x, head_reg
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


class DualHeadTemporalAttnEncoder(nn.Module):
    """
    Dual-head Temporal Attention Encoder.
    Outputs L2-normalized unit embeddings (z_pred_unit) and raw embedding norm (pred_norm).
    Supports subject-specific FiLM conditioning and per-subject projection heads.
    """

    def __init__(
        self,
        *,
        n_channels: int = 62,
        embedding_dim: int = 512,
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.2,
        stem_dropout1d: float = 0.15,
        num_subjects: int = 1,
        no_film: bool = False,
        no_subject_heads: bool = False,
        head_reg_weight: float = 0.0,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.num_subjects = num_subjects
        self.no_film = no_film
        self.no_subject_heads = no_subject_heads
        self.head_reg_weight = head_reg_weight
        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, 128, kernel_size=7, stride=4, padding=3),
            _group_norm(128),
            nn.GELU(),
            nn.Dropout1d(stem_dropout1d),
            nn.Conv1d(128, hidden_dim, kernel_size=5, stride=2, padding=2),
            _group_norm(hidden_dim),
            nn.GELU(),
            nn.Dropout1d(stem_dropout1d),
        )

        self.max_tokens = 256
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_tokens, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pooler = AttentionPooler(hidden_dim, heads=n_heads)

        # Dual Heads
        self.unit_head = nn.Linear(hidden_dim, embedding_dim)
        self.norm_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        # backward compat alias
        self.subject_heads = None

        if num_subjects > 1:
            # FiLM: zero-init so initial transform is identity
            self.subject_embed = nn.Embedding(num_subjects, hidden_dim * 2)
            nn.init.zeros_(self.subject_embed.weight)
            # Subject unit heads: warm-init from shared unit_head
            self.subject_unit_heads = nn.ModuleList([
                nn.Linear(hidden_dim, embedding_dim) for _ in range(num_subjects)
            ])
            for u_h in self.subject_unit_heads:
                u_h.load_state_dict(self.unit_head.state_dict())
            # Subject norm heads: warm-init from shared norm_head
            self.subject_norm_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, 64),
                    nn.GELU(),
                    nn.Linear(64, 1)
                ) for _ in range(num_subjects)
            ])
            for n_h in self.subject_norm_heads:
                n_h.load_state_dict(self.norm_head.state_dict())
        else:
            self.subject_embed = None
            self.subject_unit_heads = None
            self.subject_norm_heads = None

    def compute_head_reg(self) -> torch.Tensor:
        """L2 regularization: mean ||W_subject - W_shared||^2 over all subject unit heads."""
        if self.subject_unit_heads is None or self.no_subject_heads:
            return torch.tensor(0.0)
        ref_w = self.unit_head.weight.detach()
        ref_b = self.unit_head.bias.detach()
        reg = sum(
            ((h.weight - ref_w) ** 2).mean() + ((h.bias - ref_b) ** 2).mean()
            for h in self.subject_unit_heads
        )
        return reg / self.num_subjects

    def forward(
        self,
        eeg: torch.Tensor,
        subject_id: torch.Tensor | None = None,
        return_features: bool = False,
        return_norm: bool = False,
        return_head_reg: bool = False,
    ) -> torch.Tensor | tuple:
        # eeg: [B, C, T]
        x = self.stem(eeg)  # [B, D, T']
        x = x.transpose(1, 2)  # [B, T', D]

        # FiLM conditioning (skipped when no_film=True)
        use_film = (
            not self.no_film
            and getattr(self, "subject_embed", None) is not None
            and subject_id is not None
        )
        if use_film:
            film_params = self.subject_embed(subject_id)  # [B, D*2]
            gamma, beta = film_params.chunk(2, dim=-1)     # [B, D] each
            x = x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        t = x.shape[1]
        if t > self.max_tokens:
            raise ValueError(f"Token length {t} exceeds max_tokens={self.max_tokens}")
        x = x + self.pos_embed[:, :t, :]

        x = self.transformer(x)
        features = self.pooler(x)

        # Subject-specific projection heads (skipped when no_subject_heads=True)
        use_subj_heads = (
            not self.no_subject_heads
            and getattr(self, "subject_unit_heads", None) is not None
            and subject_id is not None
        )
        if use_subj_heads:
            b = features.shape[0]
            z_pred_unit = torch.zeros(b, self.unit_head.out_features, device=features.device, dtype=features.dtype)
            pred_norm = torch.zeros(b, 1, device=features.device, dtype=features.dtype)
            for sub_idx in range(self.num_subjects):
                mask = (subject_id == sub_idx)
                if mask.any():
                    z_pred_unit[mask] = self.subject_unit_heads[sub_idx](features[mask])
                    pred_norm[mask] = self.subject_norm_heads[sub_idx](features[mask])
        else:
            z_pred_unit = self.unit_head(features)
            pred_norm = self.norm_head(features)

        # L2-normalized unit vector
        z_pred_unit = F.normalize(z_pred_unit, dim=-1)

        # Predicted norm (must be positive, use softplus)
        pred_norm = F.softplus(pred_norm) + 1e-6

        head_reg = self.compute_head_reg() if return_head_reg else None

        if return_norm and return_features and return_head_reg:
            return z_pred_unit, pred_norm, features, head_reg
        if return_norm and return_features:
            return z_pred_unit, pred_norm, features
        if return_norm and return_head_reg:
            return z_pred_unit, pred_norm, head_reg
        if return_norm:
            return z_pred_unit, pred_norm
        if return_features and return_head_reg:
            return z_pred_unit, features, head_reg
        if return_features:
            return z_pred_unit, features
        if return_head_reg:
            return z_pred_unit, head_reg
        return z_pred_unit


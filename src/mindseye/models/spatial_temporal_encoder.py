"""Spatial-Temporal EEG encoder for mapping ZUNA crops to CLIP space.

Treats each EEG channel as a spatial token. Per-channel temporal features are
extracted via depthwise-separable convolutions (grouped so each channel is
processed independently), then cross-channel transformer attention learns
electrode interactions before attention-pooling to a fixed-size embedding
projected into CLIP space.

Inspired by ATM (NeurIPS 2024) and NICE encoder patterns, adapted for the
MindEye ZUNA pipeline.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SpatialAttentionPooler(nn.Module):
    """Attention pooling across channel tokens to produce a single representation."""

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        b = x.shape[0]
        q = self.query.expand(b, -1, -1)
        out, _ = self.attn(q, x, x)
        return out.squeeze(1)  # [B, D]


# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------

class SpatialTemporalEncoder(nn.Module):
    """Spatial-Temporal EEG encoder.

    Two-phase architecture:
        Phase 1 — Temporal: Extract per-channel features using grouped 1D
                  convolutions. Each of the C channels is processed
                  independently (depthwise) to produce a D-dimensional feature
                  vector summarising its temporal content.
        Phase 2 — Spatial: The C channel-tokens attend to each other via a
                  transformer, then attention-pool to a single embedding that
                  is projected into CLIP space.

    This preserves spatial identity (which electrode is which) through the
    temporal extraction stage, and only mixes across channels during the
    explicit spatial attention phase.

    Args:
        n_channels: Number of EEG channels (electrodes).
        embedding_dim: Output dimension (match CLIP space, typically 512).
        hidden_dim: Internal feature dimension for transformer tokens.
        n_layers: Number of spatial transformer layers.
        n_heads: Number of attention heads.
        dropout: General dropout rate (transformer + head).
        stem_dropout: Dropout in the temporal stem convolutions.
        stem_width: Number of intermediate features per channel in the stem.
        normalize_output: L2-normalize the output embedding.
    """

    def __init__(
        self,
        *,
        n_channels: int = 63,
        embedding_dim: int = 512,
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.2,
        stem_dropout: float = 0.15,
        stem_width: int = 8,
        normalize_output: bool = True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        self.normalize_output = normalize_output

        w = stem_width  # features per channel before projection

        # ── Phase 1: Per-channel temporal feature extraction ──────────────
        # Grouped convolutions keep channels independent.
        # Each channel goes from 1 → w features across three temporal scales,
        # with strided convolutions downsampling the time dimension.
        self.temporal_stem = nn.Sequential(
            # Fine temporal (kernel=7, stride=2) -> T/2
            nn.Conv1d(n_channels, n_channels * w, kernel_size=7, stride=2, padding=3,
                      groups=n_channels, bias=False),
            nn.BatchNorm1d(n_channels * w),
            nn.GELU(),

            # Medium temporal (kernel=15, stride=2) -> T/4
            nn.Conv1d(n_channels * w, n_channels * w, kernel_size=15, stride=2, padding=7,
                      groups=n_channels * w, bias=False),
            nn.BatchNorm1d(n_channels * w),
            nn.GELU(),
            nn.Dropout1d(stem_dropout),

            # Wide temporal (kernel=31, stride=2) -> T/8
            nn.Conv1d(n_channels * w, n_channels * w, kernel_size=31, stride=2, padding=15,
                      groups=n_channels * w, bias=False),
            nn.BatchNorm1d(n_channels * w),
            nn.GELU(),
            nn.Dropout1d(stem_dropout),
        )

        # We will flatten the time dimension dynamically in forward(),
        # then project the flattened features to hidden_dim.
        # Assuming T ≈ 307, T/8 ≈ 39. So input to linear is w * T'
        # We use a LazyLinear to avoid hardcoding the exact time length,
        # or we can use an AdaptiveAvgPool1d(16) to ensure a fixed size.
        # Let's use AdaptiveAvgPool1d(16) before flattening to guarantee size.
        self.temporal_pool = nn.AdaptiveAvgPool1d(16)

        # Project per-channel features (w * 16 dims) → hidden_dim
        self.channel_proj = nn.Sequential(
            nn.Linear(w * 16, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(stem_dropout),
        )

        # ── Learnable channel identity embeddings ─────────────────────────
        self.channel_embed = nn.Parameter(
            torch.randn(1, n_channels, hidden_dim) * 0.02
        )

        # ── Phase 2: Spatial transformer ──────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm for training stability
        )
        self.spatial_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )
        self.post_norm = nn.LayerNorm(hidden_dim)

        # ── Readout ───────────────────────────────────────────────────────
        self.pooler = SpatialAttentionPooler(hidden_dim, heads=n_heads)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg: ``[B, C, T]`` — batch of EEG tensors (channels × time).

        Returns:
            ``[B, embedding_dim]`` — L2-normalised CLIP-space embedding.
        """
        b, c, t = eeg.shape

        # Phase 1: per-channel temporal features
        h = self.temporal_stem(eeg)  # [B, C*w, T']
        h = self.temporal_pool(h)    # [B, C*w, 16]
        w_features = h.shape[1] // c
        
        # Reshape to keep channels separate: [B, C, w_features, 16]
        h = h.view(b, c, w_features, 16)
        
        # Flatten time and feature dimensions: [B, C, w_features * 16]
        h = h.flatten(2)
        
        # Project to hidden_dim
        h = self.channel_proj(h)    # [B, C, D]

        # Add channel identity
        h = h + self.channel_embed[:, :c, :]

        # Phase 2: spatial attention across channels
        h = self.spatial_transformer(h)  # [B, C, D]
        h = self.post_norm(h)

        # Readout
        h = self.pooler(h)   # [B, D]
        out = self.head(h)   # [B, embedding_dim]

        if self.normalize_output:
            out = F.normalize(out, dim=-1)

        return out


# ---------------------------------------------------------------------------
# Factory presets
# ---------------------------------------------------------------------------

def build_spatial_temporal_encoder(
    preset: str = "medium",
    *,
    n_channels: int = 63,
    embedding_dim: int = 512,
    **overrides,
) -> SpatialTemporalEncoder:
    """Build a SpatialTemporalEncoder with a named preset.

    Presets:
        small:  ~0.3M params — fast iteration, good for <5K samples
        medium: ~1.5M params — balanced
        large:  ~4M params   — for larger datasets (multi-subject)
    """
    configs = {
        "small": dict(
            hidden_dim=128,
            n_layers=2,
            n_heads=4,
            dropout=0.35,
            stem_dropout=0.15,
            stem_width=8,
        ),
        "medium": dict(
            hidden_dim=256,
            n_layers=4,
            n_heads=8,
            dropout=0.25,
            stem_dropout=0.15,
            stem_width=8,
        ),
        "large": dict(
            hidden_dim=384,
            n_layers=6,
            n_heads=8,
            dropout=0.2,
            stem_dropout=0.10,
            stem_width=16,
        ),
    }

    if preset not in configs:
        raise ValueError(f"Unknown preset '{preset}', choose from {list(configs.keys())}")

    cfg = {**configs[preset], **overrides}
    return SpatialTemporalEncoder(
        n_channels=n_channels,
        embedding_dim=embedding_dim,
        **cfg,
    )

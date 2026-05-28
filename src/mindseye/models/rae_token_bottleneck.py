"""RAE Token Bottleneck Autoencoder.

Compresses full RAE spatial tokens [B, 768, 16, 16] to a compact spatial code
[B, C, 4, 4] and expands back. The bottleneck is trained purely on image data
(no EEG) so the compressor/expander can be frozen when training EEG→code.

Supported architectures:
    spatial_768x4x4  — trivial 4x spatial pooling (reconstruction baseline)
    conv_256x4x4     — strided Conv2d encoder/decoder, primary EEG target (4096 values)
    conv_128x4x4     — leaner variant to probe compression tolerance (2048 values)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def _spatial_cosine_loss(expanded: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean (1 - cosine_similarity) computed per spatial position.

    Args:
        expanded: [B, C, H, W] reconstructed tokens
        target:   [B, C, H, W] original tokens
    Returns:
        Scalar loss.
    """
    B, C, H, W = expanded.shape
    # Reshape to [B*H*W, C] to compute cosine per position
    e = expanded.permute(0, 2, 3, 1).reshape(-1, C)
    t = target.permute(0, 2, 3, 1).reshape(-1, C)
    cos_sim = F.cosine_similarity(e, t, dim=-1)  # [B*H*W]
    return (1.0 - cos_sim).mean()


def _std_reg_loss(code: torch.Tensor, target_std: float = 0.5) -> torch.Tensor:
    """Penalize collapsed standard deviation in the bottleneck code.

    Encourages the code to maintain diversity across the batch.
    Loss is max(0, target_std - per_channel_std).mean() across channels.

    Args:
        code: [B, C, H, W] bottleneck code
        target_std: minimum acceptable std per channel across batch
    Returns:
        Scalar loss.
    """
    B, C, H, W = code.shape
    # Compute per-channel std across batch*spatial dimensions
    flat = code.reshape(B, C, -1)  # [B, C, H*W]
    per_channel_std = flat.std(dim=[0, 2])  # [C]
    deficit = F.relu(target_std - per_channel_std)
    return deficit.mean()


# ---------------------------------------------------------------------------
# Architecture implementations
# ---------------------------------------------------------------------------

class _SpatialPoolBottleneck(nn.Module):
    """Trivial baseline: 4x average-pool compress, 4x bilinear upsample expand.

    Code shape: [B, 768, 4, 4]  (12,288 values)
    No learned parameters — pure reconstruction upper-bound baseline.
    """

    def compress(self, tokens: torch.Tensor) -> torch.Tensor:
        """[B, 768, 16, 16] → [B, 768, 4, 4]"""
        return F.avg_pool2d(tokens, kernel_size=4, stride=4)

    def expand(self, code: torch.Tensor) -> torch.Tensor:
        """[B, 768, 4, 4] → [B, 768, 16, 16]"""
        return F.interpolate(code, size=(16, 16), mode="bilinear", align_corners=False)

    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        code = self.compress(tokens)
        expanded = self.expand(code)
        mse = F.mse_loss(expanded, tokens)
        cos = _spatial_cosine_loss(expanded, tokens)
        std = _std_reg_loss(code)
        total = mse + 0.5 * cos + 0.05 * std
        return {"code": code, "expanded": expanded, "mse": mse, "cos": cos, "std": std, "loss": total}


class _ConvBottleneck(nn.Module):
    """Convolutional bottleneck: two strided encoder stages + two transpose decoder stages.

    Maps [B, 768, 16, 16] → [B, code_channels, 4, 4] → [B, 768, 16, 16].
    Uses GroupNorm for batch-size independence.

    Args:
        code_channels: number of channels in the bottleneck code (128 or 256)
        hidden_channels: intermediate channel size
    """

    def __init__(self, code_channels: int = 256, hidden_channels: int = 512):
        super().__init__()
        # Encoder: 16x16 → 8x8 → 4x4
        self.encoder = nn.Sequential(
            # Stage 1: 768 → hidden, stride 2 → 8x8
            nn.Conv2d(768, hidden_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(32, hidden_channels // 8), hidden_channels),
            nn.GELU(),
            # Stage 2: hidden → code, stride 2 → 4x4
            nn.Conv2d(hidden_channels, code_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(32, code_channels // 8), code_channels),
            nn.GELU(),
        )
        # Decoder: 4x4 → 8x8 → 16x16
        self.decoder = nn.Sequential(
            # Stage 1: code → hidden, 4x4 → 8x8
            nn.ConvTranspose2d(code_channels, hidden_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(32, hidden_channels // 8), hidden_channels),
            nn.GELU(),
            # Stage 2: hidden → 768, 8x8 → 16x16
            nn.ConvTranspose2d(hidden_channels, 768, kernel_size=4, stride=2, padding=1, bias=False),
        )
        self.code_channels = code_channels

    def compress(self, tokens: torch.Tensor) -> torch.Tensor:
        """[B, 768, 16, 16] → [B, code_channels, 4, 4]"""
        return self.encoder(tokens.float())

    def expand(self, code: torch.Tensor) -> torch.Tensor:
        """[B, code_channels, 4, 4] → [B, 768, 16, 16]"""
        return self.decoder(code.float())

    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        code = self.compress(tokens)
        expanded = self.expand(code)
        mse = F.mse_loss(expanded, tokens.float())
        cos = _spatial_cosine_loss(expanded, tokens.float())
        std = _std_reg_loss(code)
        total = mse + 0.5 * cos + 0.05 * std
        return {"code": code, "expanded": expanded, "mse": mse, "cos": cos, "std": std, "loss": total}


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

ARCHITECTURES = {
    "spatial_768x4x4": lambda: _SpatialPoolBottleneck(),
    "conv_256x4x4": lambda: _ConvBottleneck(code_channels=256, hidden_channels=512),
    "conv_128x4x4": lambda: _ConvBottleneck(code_channels=128, hidden_channels=384),
}

CODE_SHAPES = {
    "spatial_768x4x4": (768, 4, 4),
    "conv_256x4x4": (256, 4, 4),
    "conv_128x4x4": (128, 4, 4),
}


def build_bottleneck(arch: str) -> nn.Module:
    """Instantiate a RAE token bottleneck by architecture name.

    Args:
        arch: one of 'spatial_768x4x4', 'conv_256x4x4', 'conv_128x4x4'
    Returns:
        nn.Module with .compress(), .expand(), and .forward() methods.
    Raises:
        ValueError if arch is not recognized.
    """
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown bottleneck arch '{arch}'. Choose from: {list(ARCHITECTURES)}")
    return ARCHITECTURES[arch]()


def code_shape(arch: str) -> tuple:
    """Return the (C, H, W) shape of the compressed code for an architecture."""
    if arch not in CODE_SHAPES:
        raise ValueError(f"Unknown arch '{arch}'")
    return CODE_SHAPES[arch]

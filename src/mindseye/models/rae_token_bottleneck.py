"""RAE Token Bottleneck Autoencoder.

Compresses full RAE spatial tokens [B, 768, 16, 16] to a compact spatial code
[B, C, H_code, W_code] and expands back. The bottleneck is trained purely on
image data (no EEG) so the bridge can be frozen when training EEG→code.

Supported architectures:
    spatial_768x4x4  — parameter-free compressor (AvgPool); learned expander back to [768,16,16]
    conv_256x4x4     — strided Conv2d encoder/decoder → [256,4,4]  (primary EEG target, 4096 values)
    conv_128x4x4     — leaner variant → [128,4,4]  (2048 values, compression tolerance test)
    conv_256x8x8     — larger code → [256,8,8]  (16384 values, fallback if 4x4 fails visually)

Loss design
-----------
total = MSE × 1.0 + CosLoss × 0.5 + StdLoss × 0.05

CosLoss: channel-vector cosine at each spatial site.
    cos = F.cosine_similarity(recon, target, dim=1)  # [B, H, W]
    cos_loss = (1 - cos).mean()

StdLoss: relative std ratio (recon std vs target std per channel).
    Penalizes channels where std_ratio = recon_std / (target_std + eps) < 0.2.
    Avoids hardcoded absolute thresholds that depend on the token distribution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def spatial_cosine_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Channel-vector cosine loss at each spatial site.

    For each (batch, H, W) position, computes the cosine similarity between the
    reconstructed and target channel vectors, then averages over batch and spatial dims.

    Args:
        recon:  [B, C, H, W] reconstructed tokens
        target: [B, C, H, W] original tokens
    Returns:
        Scalar loss: mean (1 - cosine_similarity) across all spatial sites.
    """
    # F.cosine_similarity with dim=1 → channel-wise cosine at each [B, H, W] site
    cos = F.cosine_similarity(recon, target, dim=1)  # [B, H, W]
    return (1.0 - cos).mean()


def relative_std_loss(code: torch.Tensor, target: torch.Tensor, collapse_ratio: float = 0.2) -> torch.Tensor:
    """Relative std regularization: penalize channels whose recon std is much smaller
    than the original token std, indicating collapse.

    std_ratio = std(code_channel) / (std(target_channel) + eps)
    Loss = mean(max(0, collapse_ratio - std_ratio)) across channels.

    Args:
        code:   [B, C_code, H_code, W_code] bottleneck code
        target: [B, C_tok, H_tok, W_tok] original tokens (provides reference distribution)
        collapse_ratio: threshold below which a channel is considered collapsed (default 0.2)
    Returns:
        Scalar loss.
    """
    eps = 1e-6
    B = code.shape[0]
    C_code = code.shape[1]
    C_tok = target.shape[1]

    # Per-channel std across batch*spatial for the code
    code_flat = code.reshape(B, C_code, -1)            # [B, C_code, H_code*W_code]
    code_std = code_flat.std(dim=[0, 2])               # [C_code]

    # Reference std from original tokens (use same channel count via mean if sizes differ)
    tok_flat = target.float().reshape(B, C_tok, -1)    # [B, C_tok, H*W]
    tok_std = tok_flat.std(dim=[0, 2])                 # [C_tok]
    # Use the mean of token stds as a global reference scalar
    ref_std = tok_std.mean()

    ratio = code_std / (ref_std + eps)                 # [C_code]
    deficit = F.relu(collapse_ratio - ratio)           # penalize only collapsed channels
    return deficit.mean()


# ---------------------------------------------------------------------------
# Architecture implementations
# ---------------------------------------------------------------------------

class _SpatialPoolBottleneck(nn.Module):
    """Parameter-free adaptive avg-pool compressor + learned bilinear-conv expander.

    Compressor:  F.adaptive_avg_pool2d to (code_size, code_size) — no learned parameters.
    Expander:    bilinear upsample back to 16×16 + 1×1 conv refinement (learned).

    Supports any target code_size. The 1×1 conv is initialized to identity so the
    model starts as a pure interpolation baseline and can learn corrections.

    Code shape: [B, 768, code_size, code_size]
    Values:     768 × code_size²

    Examples:
        code_size=4 → 12,288 values
        code_size=3 →  6,912 values
        code_size=2 →  3,072 values
    """

    def __init__(self, code_size: int = 4):
        super().__init__()
        self.code_size = code_size
        # Learned 1×1 refinement after upsample (~589k params regardless of code_size)
        self.refine = nn.Conv2d(768, 768, kernel_size=1, bias=True)
        nn.init.eye_(self.refine.weight.reshape(768, 768))
        nn.init.zeros_(self.refine.bias)

    def compress(self, tokens: torch.Tensor) -> torch.Tensor:
        """[B, 768, 16, 16] → [B, 768, code_size, code_size]  (parameter-free)"""
        return F.adaptive_avg_pool2d(tokens, (self.code_size, self.code_size))

    def expand(self, code: torch.Tensor) -> torch.Tensor:
        """[B, 768, code_size, code_size] → [B, 768, 16, 16]  (learned refinement)"""
        up = F.interpolate(code, size=(16, 16), mode="bilinear", align_corners=False)
        return self.refine(up)

    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens = tokens.float()
        code = self.compress(tokens)
        expanded = self.expand(code)
        mse = F.mse_loss(expanded, tokens)
        cos = spatial_cosine_loss(expanded, tokens)
        std = relative_std_loss(code, tokens)
        total = mse + 0.5 * cos + 0.05 * std
        return {"code": code, "expanded": expanded, "mse": mse, "cos": cos, "std": std, "loss": total}


class _ConvBottleneck(nn.Module):
    """Convolutional bottleneck: strided encoder + transposed decoder.

    Maps [B, 768, 16, 16] → [B, code_channels, H_code, W_code] → [B, 768, 16, 16].

    Stride pattern:
      16×16 → 8×8 (stride-2) → H_code×W_code (stride-2)
    so: H_code = W_code = 4 for two stride-2 stages,
        H_code = W_code = 8 for one stride-2 stage.

    Uses GroupNorm for batch-size independence (works with B=1).

    Args:
        code_channels:  number of channels in the bottleneck code
        hidden_channels: intermediate channel size
        code_size:      spatial size of the code (4 or 8)
    """

    def __init__(self, code_channels: int = 256, hidden_channels: int = 512, code_size: int = 4):
        super().__init__()
        self.code_channels = code_channels
        self.code_size = code_size

        if code_size == 4:
            # Two stride-2 encoder stages: 16 → 8 → 4
            self.encoder = nn.Sequential(
                nn.Conv2d(768, hidden_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(min(32, hidden_channels // 8), hidden_channels),
                nn.GELU(),
                nn.Conv2d(hidden_channels, code_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(min(32, code_channels // 8), code_channels),
                nn.GELU(),
            )
            # Two stride-2 decoder stages: 4 → 8 → 16
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(code_channels, hidden_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.GroupNorm(min(32, hidden_channels // 8), hidden_channels),
                nn.GELU(),
                nn.ConvTranspose2d(hidden_channels, 768, kernel_size=4, stride=2, padding=1, bias=False),
            )
        elif code_size == 8:
            # One stride-2 encoder stage: 16 → 8
            self.encoder = nn.Sequential(
                nn.Conv2d(768, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(min(32, hidden_channels // 8), hidden_channels),
                nn.GELU(),
                nn.Conv2d(hidden_channels, code_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(min(32, code_channels // 8), code_channels),
                nn.GELU(),
            )
            # One stride-2 decoder stage: 8 → 16
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(code_channels, hidden_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.GroupNorm(min(32, hidden_channels // 8), hidden_channels),
                nn.GELU(),
                nn.ConvTranspose2d(hidden_channels, 768, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            raise ValueError(f"code_size must be 4 or 8, got {code_size}")

    def compress(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens.float())

    def expand(self, code: torch.Tensor) -> torch.Tensor:
        return self.decoder(code.float())

    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens = tokens.float()
        code = self.compress(tokens)
        expanded = self.expand(code)
        mse = F.mse_loss(expanded, tokens)
        cos = spatial_cosine_loss(expanded, tokens)
        std = relative_std_loss(code, tokens)
        total = mse + 0.5 * cos + 0.05 * std
        return {"code": code, "expanded": expanded, "mse": mse, "cos": cos, "std": std, "loss": total}


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

ARCHITECTURES: Dict[str, object] = {
    "spatial_768x4x4": lambda: _SpatialPoolBottleneck(code_size=4),
    "spatial_768x3x3": lambda: _SpatialPoolBottleneck(code_size=3),
    "spatial_768x2x2": lambda: _SpatialPoolBottleneck(code_size=2),
    "conv_256x4x4":    lambda: _ConvBottleneck(code_channels=256, hidden_channels=512, code_size=4),
    "conv_128x4x4":    lambda: _ConvBottleneck(code_channels=128, hidden_channels=384, code_size=4),
    "conv_256x8x8":    lambda: _ConvBottleneck(code_channels=256, hidden_channels=512, code_size=8),
}

CODE_SHAPES: Dict[str, tuple] = {
    "spatial_768x4x4": (768, 4, 4),
    "spatial_768x3x3": (768, 3, 3),
    "spatial_768x2x2": (768, 2, 2),
    "conv_256x4x4":    (256, 4, 4),
    "conv_128x4x4":    (128, 4, 4),
    "conv_256x8x8":    (256, 8, 8),
}


def build_bottleneck(arch: str) -> nn.Module:
    """Instantiate a RAE token bottleneck by architecture name.

    Args:
        arch: one of 'spatial_768x4x4', 'conv_256x4x4', 'conv_128x4x4', 'conv_256x8x8'
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

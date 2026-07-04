"""Luminance-grid "visual tests" for grounding the ZUNA->QFormer->RAE translator.

These are deliberately basic, objective, pixel-derived signals (no VLM, no learned
label): the mean luminance of the whole image plus the mean luminance of a coarse
spatial grid (top-left, top, top-right, left, center, right, bottom-left, bottom,
bottom-right). Comparing the luminance grid of a *generated* image against the
*stimulus* image gives the model a simple lighting-structure target — general
illumination (global) and illumination direction (which region is bright).

Everything here is differentiable and batched so it can be used as a training loss
on a decoded image, or computed once to build a stimulus luminance-grid bank.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Rec. 601 luma weights (perceptual grayscale). Applied to RGB in [0, 1].
_LUMA_WEIGHTS = (0.299, 0.587, 0.114)

# Region order for the 3x3 grid descriptor (row-major), matching how
# adaptive_avg_pool2d(3) flattens its output.
REGION_NAMES: tuple[str, ...] = (
    "top_left", "top", "top_right",
    "left", "center", "right",
    "bottom_left", "bottom", "bottom_right",
)

# Full descriptor = global mean + 9 region means.
LUMINANCE_GRID_KEYS: tuple[str, ...] = ("global",) + REGION_NAMES
LUMINANCE_GRID_DIM: int = len(LUMINANCE_GRID_KEYS)  # 10


def rgb_to_luma(images: torch.Tensor) -> torch.Tensor:
    """Convert an RGB image tensor [B, 3, H, W] in [0, 1] to luma [B, 1, H, W]."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected images [B, 3, H, W], got {tuple(images.shape)}")
    w = torch.tensor(_LUMA_WEIGHTS, dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    return (images * w).sum(dim=1, keepdim=True)


def luminance_grid(images: torch.Tensor, grid: int = 3) -> torch.Tensor:
    """Compute the luminance-grid descriptor for a batch of RGB images.

    Args:
        images: [B, 3, H, W] float tensor in [0, 1].
        grid:   spatial grid size (3 -> 3x3 regions).

    Returns:
        [B, 1 + grid*grid] tensor: column 0 is the global mean luminance, the rest
        are per-region mean luminance in row-major order (REGION_NAMES for grid=3).
    """
    luma = rgb_to_luma(images)  # [B, 1, H, W]
    global_mean = luma.mean(dim=[1, 2, 3], keepdim=False).unsqueeze(1)  # [B, 1]
    regions = F.adaptive_avg_pool2d(luma, grid)  # [B, 1, grid, grid]
    regions = regions.reshape(regions.shape[0], -1)  # [B, grid*grid]
    return torch.cat([global_mean, regions], dim=1)  # [B, 1 + grid*grid]


def luminance_grid_loss(
    generated: torch.Tensor,
    stimulus: torch.Tensor,
    grid: int = 3,
    global_weight: float = 1.0,
) -> torch.Tensor:
    """MSE between the luminance grids of a generated image and its stimulus.

    Args:
        generated: decoded image [B, 3, H, W] in [0, 1] (differentiable).
        stimulus:  stimulus image [B, 3, H, W] in [0, 1].
        grid:      spatial grid size.
        global_weight: relative weight of the global-illumination term vs regions.

    Returns:
        scalar loss.
    """
    gen_grid = luminance_grid(generated, grid=grid)
    stim_grid = luminance_grid(stimulus, grid=grid)
    global_term = F.mse_loss(gen_grid[:, :1], stim_grid[:, :1])
    region_term = F.mse_loss(gen_grid[:, 1:], stim_grid[:, 1:])
    return global_weight * global_term + region_term

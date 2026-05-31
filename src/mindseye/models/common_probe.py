"""Multi-task probe model to evaluate and predict semantic attributes and labels from z_common/z_pred_common embeddings."""

import torch
import torch.nn as nn
from typing import Dict

# Mapping of all attributes to their expected class choices.
ATTRIBUTE_SCHEMAS = {
    "is_animate": ["no", "yes"],
    "human_visible": ["no", "yes"],
    "face_visible": ["no", "yes"],
    "animal_visible": ["no", "yes"],
    "indoor_outdoor": ["indoor", "outdoor", "mixed"],
    "natural_artificial": ["natural", "artificial", "mixed"],
    "scene_dominance": ["isolated_object", "object_with_background", "full_scene"],
    "real_world_size": ["tiny", "small", "medium", "large", "huge"],
    "dominant_color": ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "gray"],
    "main_subject_position_x": ["left", "center", "right", "full_frame"],
    "subject_scale": ["close_up", "medium_shot", "wide_shot"],
    "soft_texture": ["no", "yes"],
    "spiky_or_pointed": ["no", "yes"],
    "furry": ["no", "yes"],
    "metallic": ["no", "yes"],
    "tool_like": ["no", "yes"],
    "vehicle_like": ["no", "yes"],
    "food_like": ["no", "yes"],
    # Phase 11A Visual Calibration axes
    "warm_vs_cool": ["warm", "cool", "neutral"],
    "bright_vs_dark": ["bright", "dark", "neutral"],
    "round_or_curved": ["no", "yes"],
    "angular_or_geometric": ["no", "yes"],
    "symmetrical": ["no", "yes"],
    "single_object": ["no", "yes"],
    "glossy": ["no", "yes"],
    "rough": ["no", "yes"],
    "smooth": ["no", "yes"],
    "transparent": ["no", "yes"],
    "organic_texture": ["no", "yes"],
}

# Canonical attribute tiers (keep in sync with generate_vlm_attributes.py / analyze_vlm_attributes.py).
# Tier 1: original natural-image semantics (18 attrs) — always requested in full runs.
TIER1_ATTRIBUTE_NAMES: tuple[str, ...] = (
    "is_animate",
    "human_visible",
    "face_visible",
    "animal_visible",
    "indoor_outdoor",
    "natural_artificial",
    "scene_dominance",
    "real_world_size",
    "dominant_color",
    "main_subject_position_x",
    "subject_scale",
    "soft_texture",
    "spiky_or_pointed",
    "furry",
    "metallic",
    "tool_like",
    "vehicle_like",
    "food_like",
)

# Phase 11A calibration / material-shape axes (11 attrs) — were in ATTRIBUTE_SCHEMAS but
# missing from the original Qwen prompt; backfill with --tier calibration.
CALIBRATION_ATTRIBUTE_NAMES: tuple[str, ...] = (
    "warm_vs_cool",
    "bright_vs_dark",
    "round_or_curved",
    "angular_or_geometric",
    "symmetrical",
    "single_object",
    "glossy",
    "rough",
    "smooth",
    "transparent",
    "organic_texture",
)

ALL_VLM_ATTRIBUTE_NAMES: tuple[str, ...] = tuple(ATTRIBUTE_SCHEMAS.keys())

# Probe training adds ImageNet class_label (+1) → up to 30 heads before gating.
PROBE_CLASS_LABEL_TASK = "class_label"

IGNORE_INDEX = -100


def vlm_json_schema_lines(attr_names: tuple[str, ...]) -> list[str]:
    """Build JSON schema fragment lines for the VLM system prompt."""
    lines: list[str] = []
    for name in attr_names:
        choices = ATTRIBUTE_SCHEMAS[name]
        if set(choices) <= {"no", "yes"}:
            opt = '"yes" | "no" | "unclear"'
        elif name == "dominant_color":
            opt = " | ".join(f'"{c}"' for c in choices) + ' | "unclear"'
        else:
            opt = " | ".join(f'"{c}"' for c in choices) + ' | "unclear"'
        lines.append(f'  "{name}": {opt}')
    return lines


def unclear_fallback(attr_names: tuple[str, ...]) -> dict[str, str]:
    return {name: "unclear" for name in attr_names}

class CommonProbeModel(nn.Module):
    """Multi-task probe model targeting class labels and VLM attributes from common space."""
    def __init__(self, embedding_dim: int, task_specs: Dict[str, int]):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.task_specs = task_specs
        
        self.trunk = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.GELU(),
        )
        
        self.heads = nn.ModuleDict({
            name: nn.Linear(512, num_classes)
            for name, num_classes in task_specs.items()
        })
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Tensor of shape [B, embedding_dim]
        Returns:
            Dict mapping task name to prediction logits.
        """
        feat = self.trunk(x)
        return {name: head(feat) for name, head in self.heads.items()}
        
    @staticmethod
    def encode_label(attr_name: str, label_str: str) -> int:
        """Encode a string label into an integer class index.
        Returns IGNORE_INDEX (-100) if unclear or unknown.
        """
        if label_str == "unclear" or label_str is None:
            return IGNORE_INDEX
            
        choices = ATTRIBUTE_SCHEMAS.get(attr_name, [])
        try:
            return choices.index(label_str)
        except ValueError:
            return IGNORE_INDEX

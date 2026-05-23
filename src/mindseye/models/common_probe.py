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

IGNORE_INDEX = -100

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

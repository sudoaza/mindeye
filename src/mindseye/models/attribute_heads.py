"""Auxiliary heads for predicting semantic attributes from EEG embeddings."""

import torch
import torch.nn as nn
from typing import Dict, Tuple

# Mapping of attributes to their expected class choices.
# We will encode these classes to integers (0, 1, 2, ...).
# "unclear" is mapped to the ignore_index (-100).
ATTRIBUTE_SCHEMAS = {
    "is_animate": ["no", "yes"],
    "human_visible": ["no", "yes"],
    "face_visible": ["no", "yes"],
    "animal_visible": ["no", "yes"],
    "indoor_outdoor": ["indoor", "outdoor", "mixed"],
    "natural_artificial": ["natural", "artificial", "mixed"],
    "scene_dominance": ["isolated_object", "object_with_background", "full_scene"],
    "real_world_size": ["tiny", "small", "medium", "large", "huge"],
}

IGNORE_INDEX = -100

class AttrHead(nn.Module):
    """A simple classification head for a single attribute."""
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(in_features // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MultiTaskAttributeHeads(nn.Module):
    """Container for multiple attribute classification heads."""
    def __init__(self, in_features: int, attributes: list[str]):
        super().__init__()
        self.heads = nn.ModuleDict()
        self.attributes = attributes
        for attr in attributes:
            if attr not in ATTRIBUTE_SCHEMAS:
                raise ValueError(f"Unknown attribute: {attr}")
            num_classes = len(ATTRIBUTE_SCHEMAS[attr])
            self.heads[attr] = AttrHead(in_features, num_classes)
            
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary mapping attribute name to logits.
        """
        return {attr: head(x) for attr, head in self.heads.items()}
        
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

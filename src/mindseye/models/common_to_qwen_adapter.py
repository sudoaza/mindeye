import torch
import torch.nn as nn

class CommonToQwenAdapter(nn.Module):
    """
    Maps continuous z_common embeddings (e.g., from CLIP/SigLIP) to learned soft conditioning tokens
    (prompt_embeds) for injection into a frozen Qwen-Image model.
    """
    def __init__(
        self,
        common_dim: int = 512,
        adapter_dim: int = 1024,
        num_tokens: int = 16,
        qwen_hidden_dim: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.qwen_hidden_dim = qwen_hidden_dim
        
        self.ln1 = nn.LayerNorm(common_dim)
        self.fc1 = nn.Linear(common_dim, adapter_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(adapter_dim, num_tokens * qwen_hidden_dim)
        self.ln2 = nn.LayerNorm(qwen_hidden_dim)
        
    def forward(self, z_common: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_common: [B, common_dim] continuous latent (e.g. from MindEye encoder or CLIP)
        Returns:
            prompt_embeds: [B, num_tokens, qwen_hidden_dim] soft tokens for Qwen-Image
        """
        x = self.ln1(z_common)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        # Reshape to [B, num_tokens, qwen_hidden_dim]
        x = x.view(x.shape[0], self.num_tokens, self.qwen_hidden_dim)
        
        # Final norm across the qwen_hidden_dim
        x = self.ln2(x)
        return x

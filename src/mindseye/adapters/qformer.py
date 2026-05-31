import torch
import torch.nn as nn
import torch.nn.functional as F

class QFormerBlock(nn.Module):
    """
    A single Q-Former block consisting of:
    1. Self-attention over query tokens.
    2. Cross-attention from query tokens (Q) to key-value tokens (K, V) from ZUNA.
    3. Feed-forward network (FFN).
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = nn.GELU()

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # query: [B, num_queries, d_model]
        # key_value: [B, N, d_model]
        
        # 1. Self-attention among queries
        q2, _ = self.self_attn(query, query, query)
        query = query + self.dropout1(q2)
        query = self.norm1(query)
        
        # 2. Cross-attention (queries attend to key_value ZUNA latents)
        q2, _ = self.cross_attn(query, key_value, key_value, key_padding_mask=key_padding_mask)
        query = query + self.dropout2(q2)
        query = self.norm2(query)
        
        # 3. Feed Forward Network
        q2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(q2)
        query = self.norm3(query)
        
        return query


class AttentionPooler(nn.Module):
    """
    Attention-based pooling to map token sequences [B, S, D] to a single vector [B, D].
    """
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        q = self.query.expand(b, -1, -1)
        out, _ = self.attn(q, x, x)
        return out.squeeze(1)  # [B, D]


class ZunaToVisionQFormer(nn.Module):
    """
    QFormer adapter that maps ZUNA latents of shape [B, N, D_in] to vision embeddings [B, D_out].
    Supports:
      - Learnable query tokens.
      - Prependable learned CLS token or attention-pooling readout.
      - Multi-subject FiLM scale/shift modulators.
    """
    def __init__(
        self,
        *,
        d_in: int = 1024,
        d_out: int = 512,
        hidden_dim: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        num_query_tokens: int = 32,
        pooling_mode: str = "cls",  # "cls", "attention", or "mean"
        dropout: float = 0.15,
        num_subjects: int = 1,
        normalize_output: bool = True,
        output_layernorm: bool = True,
        force_unit_output: bool = True,
    ):
        super().__init__()
        self.pooling_mode = pooling_mode
        self.normalize_output = normalize_output
        self.output_layernorm = output_layernorm
        self.force_unit_output = force_unit_output
        self.hidden_dim = hidden_dim
        self.num_subjects = num_subjects
        
        # Input projection: maps ZUNA latents [B, N, d_in] -> [B, N, hidden_dim]
        self.input_proj = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Learnable queries
        self.query_tokens = nn.Parameter(torch.randn(num_query_tokens, hidden_dim) * 0.02)
        
        # Optional prepended CLS token
        if pooling_mode == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
            
        # Subject conditioning (FiLM) if multi-subject is active
        if num_subjects > 1:
            self.subject_embed = nn.Embedding(num_subjects, hidden_dim * 2)
            nn.init.zeros_(self.subject_embed.weight)
        else:
            self.subject_embed = None
            
        # Q-Former Blocks
        self.blocks = nn.ModuleList([
            QFormerBlock(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Readout Pooler if attention mode is chosen
        if pooling_mode == "attention":
            self.pooler = AttentionPooler(dim=hidden_dim, heads=nhead)
            
        # Output projection head to vision space
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_out)
        )
        if self.output_layernorm:
            self.final_norm = nn.LayerNorm(d_out)
            
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, zuna_latents: torch.Tensor, subject_id: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            zuna_latents: [B, N, d_in]
            subject_id:   [B] containing subject index (optional)
        Returns:
            [B, d_out] normalized target embedding
        """
        b = zuna_latents.shape[0]
        
        # Project ZUNA latents to internal representation size
        kv = self.input_proj(zuna_latents)  # [B, N, hidden_dim]
        
        # Setup query tokens
        queries = self.query_tokens.unsqueeze(0).expand(b, -1, -1)  # [B, num_query_tokens, hidden_dim]
        
        if self.pooling_mode == "cls":
            cls_tok = self.cls_token.expand(b, -1, -1)  # [B, 1, hidden_dim]
            queries = torch.cat([cls_tok, queries], dim=1)  # [B, num_query_tokens + 1, hidden_dim]
            
        # Apply subject FiLM if enabled
        if self.subject_embed is not None and subject_id is not None:
            film_params = self.subject_embed(subject_id)  # [B, hidden_dim * 2]
            gamma, beta = film_params.chunk(2, dim=-1)  # [B, hidden_dim] each
            
            # Modulate query tokens (queries)
            # queries: [B, S_q, hidden_dim]
            queries = queries * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
            
        # Pass queries and ZUNA latents through QFormer blocks
        for block in self.blocks:
            queries = block(queries, kv)
            
        # Pool query tokens to single vector
        if self.pooling_mode == "cls":
            features = queries[:, 0]  # [B, hidden_dim]
        elif self.pooling_mode == "attention":
            features = self.pooler(queries)  # [B, hidden_dim]
        else:  # "mean"
            features = queries.mean(dim=1)  # [B, hidden_dim]
            
        # Project to target dimension
        out = self.proj_head(features)  # [B, d_out]
        
        if self.output_layernorm:
            out = self.final_norm(out)
            
        if self.force_unit_output or self.normalize_output:
            out = F.normalize(out, dim=-1)
            
        return out

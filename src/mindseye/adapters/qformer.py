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
        d_in: int = 32,
        d_out: int = 512,
        hidden_dim: int = 1024,
        nhead: int = 8,
        num_layers: int = 4,
        num_query_tokens: int = 32,
        pooling_mode: str = "cls",  # "cls", "attention", or "mean"
        dropout: float = 0.15,
        num_subjects: int = 1,
        normalize_output: bool = True,
        output_layernorm: bool = True,
        force_unit_output: bool = True,
        recon_grid: bool = False,
        recon_grid_size: int = 16,
        recon_token_dim: int = 768,
        num_categories: int = 0,
    ):
        super().__init__()
        self.d_in = d_in
        self.pooling_mode = pooling_mode
        self.normalize_output = normalize_output
        self.output_layernorm = output_layernorm
        self.force_unit_output = force_unit_output
        self.hidden_dim = hidden_dim
        self.num_subjects = num_subjects
        # Optional auxiliary coarse-category classification head. EEG carries coarse
        # semantic content (animal/vehicle/food...) far more reliably than per-image
        # identity, so a jointly-trained category head grounds the shared features on
        # the structure the signal actually has — the same role the old CLIP pipeline's
        # frozen semantic probe played. It reads the *pooled features* (pre-projection)
        # so it shapes the representation the retrieval head also consumes. 0 disables.
        self.num_categories = num_categories
        # Reconstruction path: predict the full RAE token grid [B, recon_token_dim, G, G]
        # so the prediction can be decoded to an image (luminance grounding loss). This
        # closes the retrieval->reconstruction gap documented in docs/HANDOVER.md §3.
        # It is opt-in and additive: the pooled retrieval vector is still produced.
        self.recon_grid = recon_grid
        self.recon_grid_size = recon_grid_size
        self.recon_token_dim = recon_token_dim

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

        # Auxiliary category classifier off the pooled hidden features.
        if self.num_categories > 0:
            self.category_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_categories),
            )

        # Reconstruction token-grid head: map every query token to the RAE token grid.
        # We produce G*G spatial slots, each a recon_token_dim vector, from the pooled
        # query features broadcast over learned spatial position queries. This is the
        # minimal decodable bridge — a [B, recon_token_dim, G, G] grid for AutoencoderRAE.
        if self.recon_grid:
            g = self.recon_grid_size
            self.recon_pos = nn.Parameter(torch.randn(1, g * g, hidden_dim) * 0.02)
            self.recon_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
            self.recon_norm = nn.LayerNorm(hidden_dim)
            self.recon_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.recon_token_dim),
            )

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

    def forward(self, zuna_latents: torch.Tensor, subject_id: torch.Tensor | None = None, return_grid: bool = False, return_category: bool = False):
        """
        Args:
            zuna_latents: [B, N, d_in]
            subject_id:   [B] containing subject index (optional)
            return_grid:  if True (requires recon_grid=True), also return the decodable
                          RAE token grid [B, recon_token_dim, G, G].
            return_category: if True (requires num_categories>0), also return the
                          auxiliary category logits [B, num_categories].
        Returns:
            [B, d_out] pooled retrieval embedding, or a tuple with the requested
            extra outputs appended in order (grid, then category).
        """
        # Bug 3 fix: assert input is [B, N, D]
        assert zuna_latents.ndim == 3, (
            f"Expected zuna_latents to be 3-D [B, N, D], got shape {tuple(zuna_latents.shape)}"
        )
        assert zuna_latents.shape[-1] == self.d_in, (
            f"Expected latent dim={self.d_in}, got {zuna_latents.shape[-1]}"
        )
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

        extras = []
        if return_grid:
            if not self.recon_grid:
                raise RuntimeError("return_grid=True requires the model to be built with recon_grid=True")
            g = self.recon_grid_size
            spatial_q = self.recon_pos.expand(b, -1, -1)  # [B, G*G, hidden_dim]
            grid_feat, _ = self.recon_attn(spatial_q, queries, queries)  # [B, G*G, hidden_dim]
            grid_feat = self.recon_norm(grid_feat)
            grid = self.recon_proj(grid_feat)  # [B, G*G, recon_token_dim]
            grid = grid.transpose(1, 2).reshape(b, self.recon_token_dim, g, g)  # [B, D, G, G]
            extras.append(grid)

        if return_category:
            if self.num_categories <= 0:
                raise RuntimeError("return_category=True requires the model to be built with num_categories>0")
            extras.append(self.category_head(features))  # [B, num_categories]

        if extras:
            return (out, *extras)
        return out

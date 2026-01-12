import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from unish.utils.data_utils import rot6d_to_rotmat
from unish.utils.constants import SMPL_MEAN_PARAMS


class TimeStepRoPE1D(nn.Module):
    """1D RoPE for timestep embedding, similar to pi3's RoPE2D but for 1D time sequence"""
    
    def __init__(self, freq=100.0):
        super().__init__()
        self.base = freq
        self.cache = {}
        self.max_train_len = 120
    
    def get_cos_sin(self, D, seq_len, device, dtype):
        if (D, seq_len, device, dtype) in self.cache:
            return self.cache[D, seq_len, device, dtype]
        
        if seq_len <= self.max_train_len:
            assert D % 2 == 0
            
            inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            
            freqs = torch.cat((freqs, freqs), dim=-1)
            cos = freqs.cos()  # (seq_len, D)
            sin = freqs.sin()  # (seq_len, D)
            self.cache[D, seq_len, device, dtype] = (cos, sin)
            return cos, sin
            
        else:
            cos_train, sin_train = self.get_cos_sin(D, self.max_train_len, device, dtype)
            cos_train_res = cos_train.transpose(0, 1).unsqueeze(0)
            sin_train_res = sin_train.transpose(0, 1).unsqueeze(0)
            
            # [1, D, max_train_len] -> [1, D, seq_len]
            cos_interp = F.interpolate(cos_train_res, size=seq_len, mode='linear', align_corners=True)
            sin_interp = F.interpolate(sin_train_res, size=seq_len, mode='linear', align_corners=True)
            
            # [1, D, seq_len] -> [seq_len, D]
            cos_final = cos_interp.squeeze(0).transpose(0, 1)
            sin_final = sin_interp.squeeze(0).transpose(0, 1)
            
            self.cache[D, seq_len, device, dtype] = (cos_final, sin_final)
            return cos_final, sin_final
    
    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rope1d(self, tokens, pos1d, cos, sin):
        """Apply 1D RoPE to tokens based on 1D positions"""
        cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]  # [batch, 1, seq_len, D]
        sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]  # [batch, 1, seq_len, D]
        return (tokens * cos) + (self.rotate_half(tokens) * sin)
    
    def forward(self, tokens, positions):
        """
        Apply 1D RoPE to tokens based on timestep positions.
        Args:
            tokens: [batch, num_heads, seq_len, head_dim]
            positions: [batch, seq_len] - timestep positions (0, 1, 2, ...)
        Returns:
            tokens with RoPE applied: [batch, num_heads, seq_len, head_dim]
        """
        head_dim = tokens.size(3)
        assert head_dim % 2 == 0, "head_dim should be a multiple of two"
        assert positions.ndim == 2  # [batch, seq_len]
        
        cos, sin = self.get_cos_sin(head_dim, int(positions.max()) + 1, tokens.device, tokens.dtype)
        
        return self.apply_rope1d(tokens, positions.long(), cos, sin)


class TransformerDecoderLayer(nn.Module):    
    def __init__(self, hidden_dim=512, num_heads=8, ff_dim=1024, dropout=0.1, use_rope=True):
        super().__init__()
        
        self.use_rope = use_rope
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        if use_rope:
            self.self_attention = None
            self.cross_attention = None
            
            self.self_q_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.self_k_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.self_v_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.self_out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
            
            self.cross_q_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.cross_k_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.cross_v_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.cross_out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
            
            # RoPE for timestep embedding
            self.timestep_rope = TimeStepRoPE1D(freq=100.0)
        else:
            self.self_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)  # for self attention
        self.norm2 = nn.LayerNorm(hidden_dim)  # for cross attention
        self.norm3 = nn.LayerNorm(hidden_dim)  # for feed forward
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        
        # Scale factor for attention
        self.scale = self.head_dim ** -0.5
        
        # Gradient checkpointing flag
        self.use_gradient_checkpoint = False
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.use_gradient_checkpoint = True
    
    def _rope_attention(self, q_proj, k_proj, v_proj, out_proj, query, key, value, timestep_pos=None):
        """Apply RoPE-based attention using torch.nn.functional.scaled_dot_product_attention"""
        batch_size, seq_len, _ = query.shape
        
        # Project Q, K, V
        q = q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_proj(key).view(batch_size, key.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = v_proj(value).view(batch_size, value.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to Q and K if timestep positions are provided
        if timestep_pos is not None and self.use_rope:
            # For self-attention, both q and k use the same timestep positions
            if query.shape == key.shape:  # self-attention case
                q = self.timestep_rope(q, timestep_pos)
                k = self.timestep_rope(k, timestep_pos)
            else:  # cross-attention case
                # Only apply RoPE to query (cam_token), key/value are spatial features
                q = self.timestep_rope(q, timestep_pos)
        
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            scale=self.scale
        )
        
        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Output projection
        return out_proj(attn_output)
        
    def forward(self, query, key, value, self_attn_mask=None, cross_attn_mask=None, timestep_pos=None):
        """
        Args:
            query: [batch, num_views, hidden_dim]
            key: [batch, num_views, hidden_dim] 
            value: [batch, num_views, hidden_dim]
            timestep_pos: [batch, num_views] - timestep positions for RoPE
        """
        if self.use_gradient_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            
            if self.use_rope:
                # 1. Self Attention + Residual with RoPE (with gradient checkpointing)
                self_attn_output = checkpoint(
                    self._rope_attention,
                    self.self_q_proj, self.self_k_proj, self.self_v_proj, self.self_out_proj,
                    query, query, query, timestep_pos,
                    use_reentrant=False
                )
                query = self.norm1(query + self.dropout(self_attn_output))
                
                # 2. Cross Attention + Residual with RoPE (with gradient checkpointing)
                cross_attn_output = checkpoint(
                    self._rope_attention,
                    self.cross_q_proj, self.cross_k_proj, self.cross_v_proj, self.cross_out_proj,
                    query, key, value, timestep_pos,
                    use_reentrant=False
                )
                query = self.norm2(query + self.dropout(cross_attn_output))
            else:
                # 1. Self Attention + Residual (with gradient checkpointing)
                def self_attn_fn(q, k, v):
                    out, _ = self.self_attention(q, k, v, attn_mask=self_attn_mask)
                    return out
                self_attn_output = checkpoint(self_attn_fn, query, query, query, use_reentrant=False)
                query = self.norm1(query + self.dropout(self_attn_output))
                
                # 2. Cross Attention + Residual (with gradient checkpointing)
                def cross_attn_fn(q, k, v):
                    out, _ = self.cross_attention(q, k, v, attn_mask=cross_attn_mask)
                    return out
                cross_attn_output = checkpoint(cross_attn_fn, query, key, value, use_reentrant=False)
                query = self.norm2(query + self.dropout(cross_attn_output))
            
            # 3. Feed Forward + Residual (with gradient checkpointing)
            ff_output = checkpoint(self.feed_forward, query, use_reentrant=False)
            query = self.norm3(query + ff_output)
        else:
            # Original implementation without gradient checkpointing
            if self.use_rope:
                # 1. Self Attention + Residual with RoPE
                self_attn_output = self._rope_attention(
                    self.self_q_proj, self.self_k_proj, self.self_v_proj, self.self_out_proj,
                    query, query, query, timestep_pos
                )
                query = self.norm1(query + self.dropout(self_attn_output))
                
                # 2. Cross Attention + Residual with RoPE
                cross_attn_output = self._rope_attention(
                    self.cross_q_proj, self.cross_k_proj, self.cross_v_proj, self.cross_out_proj,
                    query, key, value, timestep_pos
                )
                query = self.norm2(query + self.dropout(cross_attn_output))
            else:
                # 1. Self Attention + Residual (original implementation)
                self_attn_output, _ = self.self_attention(query, query, query, attn_mask=self_attn_mask)
                query = self.norm1(query + self.dropout(self_attn_output))
                
                # 2. Cross Attention + Residual (original implementation)
                cross_attn_output, _ = self.cross_attention(query, key, value, attn_mask=cross_attn_mask)
                query = self.norm2(query + self.dropout(cross_attn_output))
            
            # 3. Feed Forward + Residual
            ff_output = self.feed_forward(query)
            query = self.norm3(query + ff_output)
        
        return query


class CrossViewTransformerDecoderLayer(nn.Module):
    """Cross-view Transformer Decoder Layer for V4 - handles concatenated tokens from multiple views"""
    
    def __init__(self, hidden_dim=512, num_heads=8, ff_dim=1024, dropout=0.1, use_rope=True):
        super().__init__()
        
        self.use_rope = use_rope
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        if use_rope:
            self.self_attention = None
            self.cross_attention = None
            
            # Self-attention components
            self.self_q_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.self_k_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.self_v_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.self_out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
            
            # Cross-attention components
            self.cross_q_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.cross_k_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.cross_v_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.cross_out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
            
            # RoPE for timestep embedding
            self.timestep_rope = TimeStepRoPE1D(freq=100.0)
        else:
            self.self_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)  # for self attention
        self.norm2 = nn.LayerNorm(hidden_dim)  # for cross attention
        self.norm3 = nn.LayerNorm(hidden_dim)  # for feed forward
        
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5
        
        self.use_gradient_checkpoint = False
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.use_gradient_checkpoint = True
    
    def _rope_attention(self, q_proj, k_proj, v_proj, out_proj, query, key, value, query_timestep_pos=None, key_timestep_pos=None):
        """Apply RoPE-based attention for cross-view scenarios using torch.nn.functional.scaled_dot_product_attention"""
        batch_size, query_seq_len, _ = query.shape
        _, key_seq_len, _ = key.shape
        
        # Project Q, K, V
        q = q_proj(query).view(batch_size, query_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_proj(key).view(batch_size, key_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v_proj(value).view(batch_size, key_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to Q and K if timestep positions are provided
        if self.use_rope:
            if query_timestep_pos is not None:
                q_scale = q[:, :, 0:1, :]  # [batch, num_heads, 1, head_dim] - scale token
                q_cam = q[:, :, 1:, :]     # [batch, num_heads, num_views, head_dim] - cam tokens
                
                cam_timestep_pos = query_timestep_pos[:, 1:]
                q_cam_rope = self.timestep_rope(q_cam, cam_timestep_pos)
                
                q = torch.cat([q_scale, q_cam_rope], dim=2)
            if key_timestep_pos is not None:
                k = self.timestep_rope(k, key_timestep_pos)
        
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            scale=self.scale
        )
        
        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, query_seq_len, self.hidden_dim)
        
        # Output projection
        return out_proj(attn_output)
        
    def forward(self, query, key, value, query_timestep_pos=None, key_timestep_pos=None):
        """
        Args:
            query: [batch, num_queries, hidden_dim] - cam tokens + scale token
            key: [batch, num_views * num_tokens, hidden_dim] - concatenated feature tokens from all views
            value: [batch, num_views * num_tokens, hidden_dim] - concatenated feature tokens from all views
            query_timestep_pos: [batch, num_queries] - timestep positions for query tokens
            key_timestep_pos: [batch, num_views * num_tokens] - timestep positions for key/value tokens
        """
        if self.use_gradient_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            
            if self.use_rope:
                # 1. Self Attention + Residual with RoPE (with gradient checkpointing)
                self_attn_output = checkpoint(
                    self._rope_attention,
                    self.self_q_proj, self.self_k_proj, self.self_v_proj, self.self_out_proj,
                    query, query, query, query_timestep_pos, query_timestep_pos,
                    use_reentrant=False
                )
                query = self.norm1(query + self.dropout(self_attn_output))
                
                # 2. Cross Attention + Residual with RoPE (with gradient checkpointing)
                cross_attn_output = checkpoint(
                    self._rope_attention,
                    self.cross_q_proj, self.cross_k_proj, self.cross_v_proj, self.cross_out_proj,
                    query, key, value, query_timestep_pos, key_timestep_pos,
                    use_reentrant=False
                )
                query = self.norm2(query + self.dropout(cross_attn_output))
            else:
                # 1. Self Attention + Residual (with gradient checkpointing)
                def self_attn_fn(q, k, v):
                    out, _ = self.self_attention(q, k, v)
                    return out
                self_attn_output = checkpoint(self_attn_fn, query, query, query, use_reentrant=False)
                query = self.norm1(query + self.dropout(self_attn_output))
                
                # 2. Cross Attention + Residual (with gradient checkpointing)
                def cross_attn_fn(q, k, v):
                    out, _ = self.cross_attention(q, k, v)
                    return out
                cross_attn_output = checkpoint(cross_attn_fn, query, key, value, use_reentrant=False)
                query = self.norm2(query + self.dropout(cross_attn_output))
            
            # 3. Feed Forward + Residual (with gradient checkpointing)
            ff_output = checkpoint(self.feed_forward, query, use_reentrant=False)
            query = self.norm3(query + ff_output)
        else:
            # Original implementation without gradient checkpointing
            if self.use_rope:
                # 1. Self Attention + Residual with RoPE
                self_attn_output = self._rope_attention(
                    self.self_q_proj, self.self_k_proj, self.self_v_proj, self.self_out_proj,
                    query, query, query, query_timestep_pos, query_timestep_pos
                )
                query = self.norm1(query + self.dropout(self_attn_output))
                
                # 2. Cross Attention + Residual with RoPE
                cross_attn_output = self._rope_attention(
                    self.cross_q_proj, self.cross_k_proj, self.cross_v_proj, self.cross_out_proj,
                    query, key, value, query_timestep_pos, key_timestep_pos
                )
                query = self.norm2(query + self.dropout(cross_attn_output))
            else:
                # 1. Self Attention + Residual (original implementation)
                self_attn_output, _ = self.self_attention(query, query, query)
                query = self.norm1(query + self.dropout(self_attn_output))
                
                # 2. Cross Attention + Residual (original implementation)
                cross_attn_output, _ = self.cross_attention(query, key, value)
                query = self.norm2(query + self.dropout(cross_attn_output))
            
            # 3. Feed Forward + Residual
            ff_output = self.feed_forward(query)
            query = self.norm3(query + ff_output)
        
        return query


class AlignNet(nn.Module):
    def __init__(self, aggregated_dim=2048, cam_dim=1024, hidden_dim=512, num_heads=8, ff_dim=512, dropout=0.1, use_rope=True, num_decoder_layers=2):
        super().__init__()
        
        self.use_rope = use_rope
        self.hidden_dim = hidden_dim
        self.num_decoder_layers = num_decoder_layers
        
        self.scale_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        self.cam_feature_adapter = nn.Sequential(
            nn.LayerNorm(cam_dim),
            nn.Linear(cam_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.patch_feature_adapter = nn.Sequential(
            nn.LayerNorm(aggregated_dim),
            nn.Linear(aggregated_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.register_feature_adapter = nn.Sequential(
            nn.LayerNorm(aggregated_dim),
            nn.Linear(aggregated_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.decoder_layers = nn.ModuleList([
            CrossViewTransformerDecoderLayer(hidden_dim, num_heads, ff_dim, dropout, use_rope=use_rope)
            for _ in range(num_decoder_layers)
        ])
        
        mean_params = SMPL_MEAN_PARAMS
        init_body_pose = torch.from_numpy(mean_params['pose'].astype(np.float32)).unsqueeze(0)
        init_betas = torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32)).unsqueeze(0)
        self.register_buffer('init_body_pose', init_body_pose)
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_cam', init_cam)

        self.trans_head = nn.Linear(hidden_dim, 3)
        
        self.scale_head = nn.Linear(hidden_dim, 1)

        self.joint_conversion_fn = rot6d_to_rotmat
        
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        for layer in self.decoder_layers:
            if hasattr(layer, 'gradient_checkpointing_enable'):
                layer.gradient_checkpointing_enable()

    def forward(self, hidden_tokens, cam_token, fps=6.0):
        batch_size, num_views, num_tokens, _ = hidden_tokens.shape
        
        register_tokens = hidden_tokens[:, :, :5, :]
        patch_tokens = hidden_tokens[:, :, 5:, :]
        
        if cam_token.dim() == 4:
            cam_token = cam_token.squeeze(2)  # [batch, num_views, 1, 1024] -> [batch, num_views, 1024]
        
        cam_adapted = self.cam_feature_adapter(cam_token)  # [batch, num_views, hidden_dim]
        
        patch_tokens_reshaped = patch_tokens.view(batch_size * num_views, patch_tokens.shape[2], -1)  # [batch*num_views, 777, 2048]
        patch_adapted_tokens = self.patch_feature_adapter(patch_tokens_reshaped)  # [batch*num_views, 777, hidden_dim]
        patch_adapted_tokens = patch_adapted_tokens.view(batch_size, num_views, patch_tokens.shape[2], -1)  # [batch, num_views, 777, hidden_dim]
        
        register_tokens_reshaped = register_tokens.view(batch_size * num_views, 5, -1)  # [batch*num_views, 5, 2048]
        register_adapted_tokens = self.register_feature_adapter(register_tokens_reshaped)  # [batch*num_views, 5, hidden_dim]
        register_adapted_tokens = register_adapted_tokens.view(batch_size, num_views, 5, -1)  # [batch, num_views, 5, hidden_dim]
        
        fused_features_per_view = torch.cat([register_adapted_tokens, patch_adapted_tokens], dim=2)  # [batch, num_views, 782, hidden_dim]
        
        concatenated_features = fused_features_per_view.view(batch_size, num_views * num_tokens, -1)
        
        scale_token_expanded = self.scale_token.expand(batch_size, -1, -1)
        
        query_tokens = torch.cat([scale_token_expanded, cam_adapted], dim=1)
        
        if self.use_rope:
            base_fps = 6.0
            
            time_scale = base_fps / fps
            
            scale_timestep = torch.zeros((batch_size, 1), device=cam_adapted.device, dtype=torch.long)
            
            cam_timestep_float = torch.arange(num_views, device=cam_adapted.device, dtype=torch.float32) * time_scale
            cam_timestep = cam_timestep_float.round().long().unsqueeze(0).expand(batch_size, -1)
            query_timestep_pos = torch.cat([scale_timestep, cam_timestep], dim=1)  # [batch, 1 + num_views]
            
            key_timestep_base_float = torch.arange(num_views, device=cam_adapted.device, dtype=torch.float32) * time_scale
            key_timestep_base = key_timestep_base_float.round().long()
            key_timestep_pos = key_timestep_base.unsqueeze(1).expand(-1, num_tokens).flatten()
            key_timestep_pos = key_timestep_pos.unsqueeze(0).expand(batch_size, -1)  # [batch, num_views * num_tokens]
        else:
            query_timestep_pos = None
            key_timestep_pos = None

        decoder_output = query_tokens
        for i, layer in enumerate(self.decoder_layers):
            residual = decoder_output
            
            decoder_output = layer(
                decoder_output, concatenated_features, concatenated_features, 
                query_timestep_pos=query_timestep_pos, key_timestep_pos=key_timestep_pos
            )
            
            decoder_output = decoder_output + residual
        
        scale_output = decoder_output[:, 0, :]
        cam_outputs = decoder_output[:, 1:, :]
        
        scale_logits = self.scale_head(scale_output)  # [batch, 1]
        scale = F.softplus(scale_logits)
        
        trans_raw = self.trans_head(cam_outputs) # [batch, num_views, 3]
        xy, z = trans_raw.split([2, 1], dim=-1)  # xy: [batch, num_views, 2], z: [batch, num_views, 1]
        z = torch.exp(z)
        trans = torch.cat([xy * z, z], dim=-1)  # [batch, num_views, 3]


        return {
            "scale": scale,       # [batch, 1]
            "trans_cam": trans,        # [batch, num_views, 3]
        }

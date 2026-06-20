# Simulation DiT (SimTransformer) for PV-Simulator.
# 10-block Transformer with self-attention + FFN, timestep modulation.
# No cross-attention (text conditioning is only in the video branch).
#
# Input: encoded point states (B, T, N, d_state) + init_enc (B, T, N, d_state)
#        + init_mask (B, T, N, 1) + point anchors (B, T, N, d_anchor)
#        + conditions (B, T, N, d_cond)
# Output: predicted value in latent space (B, T, N, d_state)

from typing import Optional

import torch
import torch.nn as nn
import torch.cuda.amp as amp

from videox_fun.models.attention_utils import attention
from videox_fun.models.wan_transformer3d import (
    WanRMSNorm,
    WanLayerNorm,
    sinusoidal_embedding_1d,
)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_rope_1d(x: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """Apply standard 1D RoPE to attention tensors shaped (B, S, H, D)."""
    _, seq_len, _, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires even head_dim, got {head_dim}.")

    pos = torch.arange(seq_len, device=x.device, dtype=torch.float32)
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim)
    )
    freqs = torch.outer(pos, inv_freq)
    cos = torch.repeat_interleave(freqs.cos(), 2, dim=-1).view(1, seq_len, 1, head_dim)
    sin = torch.repeat_interleave(freqs.sin(), 2, dim=-1).view(1, seq_len, 1, head_dim)

    x_float = x.to(torch.float32)
    x_rope = x_float * cos + _rotate_half(x_float) * sin
    return x_rope.to(x.dtype)


class SimSelfAttention(nn.Module):
    """Self-attention for simulation tokens.

    Mirrors WanSelfAttention but without RoPE (simulation tokens don't have
    a 3D spatial grid). Uses the shared attention() backend for FlashAttention.

    Args:
        dim: Hidden dimension.
        num_heads: Number of attention heads.
        qk_norm: Whether to apply RMSNorm to Q/K.
        eps: Epsilon for normalization.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_rope = use_rope
        self.rope_theta = rope_theta
        self.register_buffer("rope_indicator", torch.tensor(float(use_rope)), persistent=True)

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, dtype=torch.bfloat16, attn_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: (B, L, C) where L = T*N.
            dtype: Dtype for the attention kernel (FlashAttention needs bf16/fp16).
            attn_mask: Optional additive attention bias (B, 1, 1, L) or (B, 1, L, L).
        Returns:
            (B, L, C)
        """
        b, s, _ = x.shape
        n, d = self.num_heads, self.head_dim
        # Use model's own weight dtype for linear projections.
        # dtype arg only controls the attention kernel call.
        w_dtype = self.q.weight.dtype

        q = self.norm_q(self.q(x.to(w_dtype))).view(b, s, n, d)
        k = self.norm_k(self.k(x.to(w_dtype))).view(b, s, n, d)
        v = self.v(x.to(w_dtype)).view(b, s, n, d)

        if self.use_rope:
            q = _apply_rope_1d(q, theta=self.rope_theta)
            k = _apply_rope_1d(k, theta=self.rope_theta)

        # Force SDPA when attn_mask is provided: the shared wrapper would otherwise
        # convert attn_mask → k_lens for Flash Attention and silently drop the mask
        # if FA isn't installed (falling back to SDPA without the mask).
        attention_type = "SDPA" if attn_mask is not None else None
        x = attention(q.to(dtype), k.to(dtype), v=v.to(dtype),
                      attn_mask=attn_mask, attention_type=attention_type)
        x = x.to(w_dtype).flatten(2)
        x = self.o(x)
        return x


class SimAttentionBlock(nn.Module):
    """Simulation Transformer block: LayerNorm → Self-Attention → LayerNorm → FFN.

    Uses 6-vector timestep modulation matching WanAttentionBlock's pattern:
    e[0]=shift_sa, e[1]=scale_sa, e[2]=gate_sa_out,
    e[3]=shift_ffn, e[4]=scale_ffn, e[5]=gate_ffn_out.

    No cross-attention — text conditioning is only in the video branch.

    Args:
        dim: Hidden dimension.
        ffn_dim: FFN intermediate dimension.
        num_heads: Number of attention heads.
        qk_norm: Whether to apply RMSNorm to Q/K.
        eps: Epsilon for normalization.
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()

        # Self-attention
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = SimSelfAttention(dim, num_heads, qk_norm, eps)

        # FFN
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim),
        )

        # 6-vector timestep modulation (matching WanAttentionBlock)
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward(self, x, e, dtype=torch.bfloat16, attn_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: (B, L, C) where L = T*N.
            e: (B, 6, C) timestep modulation embedding.
            attn_mask: Optional additive attention bias forwarded to self-attention.
        Returns:
            (B, L, C)
        """
        e = (self.modulation + e).chunk(6, dim=1)

        # Self-attention with modulation
        h = self.norm1(x) * (1 + e[1]) + e[0]
        h = h.to(dtype)
        y = self.self_attn(h, dtype, attn_mask=attn_mask)
        x = x + y * e[2]

        # FFN with modulation
        h = self.norm2(x) * (1 + e[4]) + e[3]
        h = h.to(dtype)
        y = self.ffn(h)
        x = x + y * e[5]

        return x


class SimTemporalCorrespondenceBlock(nn.Module):
    """Explicit same-point temporal attention across frames.

    Operates on each point track independently: for point p, attention runs over
    [x_0p, x_1p, ..., x_Tp]. This is a minimal way to inject per-point
    correspondence without introducing any spatial structural loss.
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
        use_temporal_rope: bool = False,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.norm1 = WanLayerNorm(dim, eps)
        self.temporal_attn = SimSelfAttention(
            dim,
            num_heads,
            qk_norm,
            eps,
            use_rope=use_temporal_rope,
            rope_theta=rope_theta,
        )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, x, dtype=torch.bfloat16, valid_token_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: (B, T, N, C)
            valid_token_mask: Optional (B, T, N) bool
        Returns:
            (B, T, N, C)
        """
        B, T, N, C = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, C)  # (B*N, T, C)

        attn_mask = None
        if valid_token_mask is not None:
            temporal_mask = valid_token_mask.permute(0, 2, 1).reshape(B * N, T)
            attn_mask = torch.zeros(B * N, 1, 1, T, device=x.device, dtype=dtype)
            attn_mask.masked_fill_(
                ~temporal_mask.unsqueeze(1).unsqueeze(2),
                torch.finfo(dtype).min,
            )

        h = self.norm1(x).to(dtype)
        x = x + self.temporal_attn(h, dtype=dtype, attn_mask=attn_mask)
        h = self.norm2(x).to(dtype)
        x = x + self.ffn(h)
        x = x.view(B, N, T, C).permute(0, 2, 1, 3).contiguous()  # (B, T, N, C)
        return x


class SimTransformer(nn.Module):
    """Simulation DiT: 10-block Transformer for physics trajectory denoising.

    Input: encoded noisy states x_enc (B, T, N, d_state), initial frame
    encoding init_enc (B, T, N, d_state) zero-padded beyond t=0, inpainting
    mask init_mask (B, T, N, 1), per-point anchors point_anchor (B, T, N, d_anchor)
    repeated across time, and conditions c_sim (B, T, N, d_cond). All concatenated
    and projected to hidden dim.

    Output: predicted value in latent space (B, T, N, d_state).

    Args:
        d_state: Dimension of encoded point states (AE pos+vel concat = 32).
        d_cond: Dimension of condition embedding (60).
        d_sim: Hidden dimension of the transformer.
        ffn_dim: FFN intermediate dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer blocks.
        freq_dim: Dimension of sinusoidal timestep embedding.
        qk_norm: Whether to apply RMSNorm to Q/K.
        eps: Epsilon for normalization.
    """

    def __init__(
        self,
        d_state: int = 32,
        d_cond: int = 60,
        d_anchor: int = 3,
        d_sim: int = 256,
        ffn_dim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 10,
        use_temporal_correspondence: bool = False,
        use_temporal_rope: bool = False,
        rope_theta: float = 10000.0,
        freq_dim: int = 256,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.d_state = d_state
        self.d_cond = d_cond
        self.d_anchor = d_anchor
        self.d_sim = d_sim
        self.num_layers = num_layers
        self.use_temporal_correspondence = use_temporal_correspondence
        self.use_temporal_rope = use_temporal_rope
        self.freq_dim = freq_dim

        # Input projection:
        # [x_enc | init_enc | init_mask | point_anchor | c_sim] → d_sim
        # = 2 * d_state + 1 + d_anchor + d_cond
        self.input_proj = nn.Linear(2 * d_state + 1 + d_anchor + d_cond, d_sim)

        # Timestep embedding (same pattern as WanTransformer3DModel)
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, d_sim),
            nn.SiLU(),
            nn.Linear(d_sim, d_sim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_sim, d_sim * 6),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            SimAttentionBlock(d_sim, ffn_dim, num_heads, qk_norm, eps)
            for _ in range(num_layers)
        ])
        self.temporal_corr_block = (
            SimTemporalCorrespondenceBlock(
                d_sim,
                ffn_dim,
                num_heads,
                qk_norm,
                eps,
                use_temporal_rope=use_temporal_rope,
                rope_theta=rope_theta,
            )
            if use_temporal_correspondence else None
        )

        # Output head
        self.head_norm = WanLayerNorm(d_sim, eps)
        self.head_proj = nn.Linear(d_sim, d_state)

        self._init_weights()

    def _init_weights(self):
        """Zero-init head projection for stable training start."""
        nn.init.zeros_(self.head_proj.weight)
        nn.init.zeros_(self.head_proj.bias)

    def forward(self, x_enc, init_enc, init_mask, point_anchor, c_sim, t, dtype=torch.bfloat16,
                valid_seq_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x_enc: (B, T, N, d_state) — AE-encoded noisy point states.
            init_enc: (B, T, N, d_state) — AE-encoded initial frame, zero-padded beyond t=0.
            init_mask: (B, T, N, 1) — 0 at t=0 (given frame), 1 elsewhere.
            point_anchor: (B, T, N, d_anchor) — per-point anchor features repeated across time.
            c_sim: (B, T, N, d_cond) — condition embeddings.
            t: (B,) — diffusion timestep.
            dtype: Compute dtype for attention.
            valid_seq_mask: Optional (B, T*N) bool — True for valid (non-padded) tokens.
                If provided, padded tokens are masked out in self-attention via an
                additive key bias of -inf. Used in padded batch mode.
        Returns:
            (B, T, N, d_state) — predicted value in latent space.
        """
        B, T, N, _ = x_enc.shape

        # Concatenate state, init conditioning, mask, and conditions → project
        x = torch.cat([x_enc, init_enc, init_mask, point_anchor, c_sim], dim=-1)
        x = self.input_proj(x)                   # (B, T, N, d_sim)

        # Give every token an explicit trajectory-time coordinate before global
        # (T, N) attention.  The diffusion timestep embedding below is shared by
        # every token and therefore cannot distinguish frame t from frame t + 1.
        # This parameter-free encoding remains active when temporal RoPE is off.
        temporal_pos = sinusoidal_embedding_1d(
            self.d_sim,
            torch.arange(T, device=x.device),
        ).to(dtype=x.dtype)
        x = x + temporal_pos.view(1, T, 1, self.d_sim)

        valid_token_mask = None
        if valid_seq_mask is not None:
            valid_token_mask = valid_seq_mask.view(B, T, N)
        if self.temporal_corr_block is not None:
            x = self.temporal_corr_block(x, dtype=dtype, valid_token_mask=valid_token_mask)

        x = x.view(B, T * N, self.d_sim)         # (B, T*N, d_sim)

        # Timestep modulation (computed in fp32 for numerical stability, then cast back)
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float()
            )
            e0 = self.time_projection(e).unflatten(1, (6, self.d_sim))
            # e0: (B, 6, d_sim)
        e0 = e0.to(dtype)   # cast back so modulation arithmetic stays in model dtype

        # Build additive attention bias to mask padding tokens (padded batch mode)
        attn_mask = None
        if valid_seq_mask is not None:
            L = T * N
            # key_bias: (B, 1, 1, L) — 0 for valid tokens, -inf for padding
            key_bias = torch.zeros(B, 1, 1, L, device=x.device, dtype=dtype)
            key_bias.masked_fill_(~valid_seq_mask.unsqueeze(1).unsqueeze(2),
                                  torch.finfo(dtype).min)
            attn_mask = key_bias

        # Transformer blocks
        for block in self.blocks:
            x = block(x, e0, dtype, attn_mask=attn_mask)

        # Output head
        x = x.view(B, T, N, self.d_sim)
        x = self.head_proj(self.head_norm(x.to(self.head_proj.weight.dtype)))  # (B, T, N, d_state)
        return x

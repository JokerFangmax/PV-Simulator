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


def _spatial_sinusoidal_embedding(anchor_positions: torch.Tensor, dim: int) -> torch.Tensor:
    """Encode anchor-frame XYZ positions as a ``(B, N, dim)`` sinusoid."""
    num_frequencies = dim // 6
    if num_frequencies == 0:
        return anchor_positions.new_zeros(*anchor_positions.shape[:2], dim)
    inv_freq = 1.0 / (
        10000 ** (torch.arange(num_frequencies, device=anchor_positions.device, dtype=torch.float32)
                   / num_frequencies)
    )
    angles = anchor_positions.float().unsqueeze(-1) * inv_freq
    embedding = torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2)
    if embedding.shape[-1] < dim:
        embedding = torch.nn.functional.pad(embedding, (0, dim - embedding.shape[-1]))
    return embedding


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
    """Global-attention ablation or factorized spatial-then-temporal DiT block."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        d_cond: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
        use_temporal_rope: bool = False,
        rope_theta: float = 10000.0,
    ):
        super().__init__()

        # ``self_attn`` is spatial attention in factorized mode and flat global
        # attention in the backward-compatible ablation.
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = SimSelfAttention(dim, num_heads, qk_norm, eps)
        self.temporal_norm = WanLayerNorm(dim, eps)
        self.temporal_attn = SimSelfAttention(
            dim, num_heads, qk_norm, eps,
            use_rope=use_temporal_rope,
            rope_theta=rope_theta,
        )
        # Per-token physical conditioning modulates the spatial AdaLN path.
        self.spatial_condition_modulation = nn.Linear(d_cond, 2 * dim)

        # FFN
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim),
        )

        # 6-vector timestep modulation (matching WanAttentionBlock)
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward_global(self, x, e, dtype=torch.bfloat16, attn_mask: Optional[torch.Tensor] = None):
        """Original flattened global-attention path for ablations."""
        e = (self.modulation + e).chunk(6, dim=1)
        h = self.norm1(x) * (1 + e[1]) + e[0]
        h = h.to(dtype)
        y = self.self_attn(h, dtype, attn_mask=attn_mask)
        x = x + y * e[2]
        h = self.norm2(x) * (1 + e[4]) + e[3]
        h = h.to(dtype)
        y = self.ffn(h)
        x = x + y * e[5]
        return x

    def forward(self, x, e, dtype=torch.bfloat16, attn_mask: Optional[torch.Tensor] = None):
        """Compatibility entry point for callers that use flattened attention."""
        return self.forward_global(x, e, dtype=dtype, attn_mask=attn_mask)

    def forward_factorized(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        c_sim: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
        spatial_attn_mask: Optional[torch.Tensor] = None,
        temporal_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply spatial attention over points, then temporal attention per point."""
        B, T, N, C = x.shape
        e = (self.modulation + e).chunk(6, dim=1)
        cond_dtype = self.spatial_condition_modulation.weight.dtype
        cond_shift, cond_scale = self.spatial_condition_modulation(c_sim.to(cond_dtype)).chunk(2, dim=-1)

        # Spatial AdaLN: each frame attends over all of its points.
        h = self.norm1(x) * (1 + e[1].unsqueeze(1) + cond_scale) + e[0].unsqueeze(1) + cond_shift
        h = h.to(dtype).reshape(B * T, N, C)
        y = self.self_attn(h, dtype, attn_mask=spatial_attn_mask)
        x = x + y.view(B, T, N, C).to(x.dtype) * e[2].unsqueeze(1)

        # Temporal AdaLN: each point attends to its own trajectory only.
        h = self.temporal_norm(x) * (1 + e[1].unsqueeze(1)) + e[0].unsqueeze(1)
        h = h.to(dtype).permute(0, 2, 1, 3).reshape(B * N, T, C)
        y = self.temporal_attn(h, dtype, attn_mask=temporal_attn_mask)
        y = y.view(B, N, T, C).permute(0, 2, 1, 3)
        x = x + y.to(x.dtype) * e[2].unsqueeze(1)

        h = self.norm2(x) * (1 + e[4].unsqueeze(1)) + e[3].unsqueeze(1)
        y = self.ffn(h.to(self.ffn[0].weight.dtype)).to(x.dtype)
        return x + y * e[5].unsqueeze(1)


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
        use_factorized_attention: bool = True,
        use_temporal_correspondence: bool = False,
        use_temporal_rope: bool = False,
        use_object_local_attention: bool = False,
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
        self.use_factorized_attention = use_factorized_attention
        self.use_temporal_correspondence = use_temporal_correspondence
        self.use_temporal_rope = use_temporal_rope
        self.use_object_local_attention = use_object_local_attention
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
            SimAttentionBlock(
                d_sim, ffn_dim, num_heads, d_cond, qk_norm, eps,
                use_temporal_rope=use_temporal_rope,
                rope_theta=rope_theta,
            )
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
            if use_temporal_correspondence and not use_factorized_attention else None
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
                valid_seq_mask: Optional[torch.Tensor] = None,
                point_obj_idx: Optional[torch.Tensor] = None):
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
            point_obj_idx: Optional (B, N) int — point-to-object mapping. When
                ``use_object_local_attention`` is enabled, different objects are
                masked from each other at every temporal position.
        Returns:
            (B, T, N, d_state) — predicted value in latent space.
        """
        B, T, N, _ = x_enc.shape
        if point_anchor.ndim == 3:
            if point_anchor.shape[:2] != (B, N):
                raise ValueError(
                    "Static point_anchor must have shape (B, N, d_anchor), got "
                    f"{tuple(point_anchor.shape)} for B={B}, N={N}."
                )
            point_anchor = point_anchor.unsqueeze(1).expand(-1, T, -1, -1)

        # Concatenate state, init conditioning, mask, and conditions → project
        x = torch.cat([x_enc, init_enc, init_mask, point_anchor, c_sim], dim=-1)
        x = self.input_proj(x)                   # (B, T, N, d_sim)

        # Explicit trajectory-time coordinates remain active when temporal RoPE is off.
        temporal_pos = sinusoidal_embedding_1d(
            self.d_sim,
            torch.arange(T, device=x.device),
        ).to(dtype=x.dtype)
        x = x + temporal_pos.view(1, T, 1, self.d_sim)
        if point_anchor.shape[-1] >= 3:
            spatial_pos = _spatial_sinusoidal_embedding(point_anchor[:, 0, :, :3], self.d_sim)
            x = x + spatial_pos.to(dtype=x.dtype).unsqueeze(1)

        valid_token_mask = None
        if valid_seq_mask is not None:
            valid_token_mask = valid_seq_mask.view(B, T, N)
        if self.temporal_corr_block is not None:
            x = self.temporal_corr_block(x, dtype=dtype, valid_token_mask=valid_token_mask)

        # Timestep modulation (computed in fp32 for numerical stability, then cast back)
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float()
            )
            e0 = self.time_projection(e).unflatten(1, (6, self.d_sim))
            # e0: (B, 6, d_sim)
        e0 = e0.to(dtype)   # cast back so modulation arithmetic stays in model dtype

        if point_obj_idx is not None and point_obj_idx.shape != (B, N):
            raise ValueError(
                "point_obj_idx must have shape (B, N), got "
                f"{tuple(point_obj_idx.shape)} for B={B}, N={N}."
            )

        if self.use_factorized_attention:
            # Spatial masks are applied independently to each frame.
            spatial_attn_mask = None
            if self.use_object_local_attention and point_obj_idx is not None:
                same_object = point_obj_idx.to(x.device).unsqueeze(2) == point_obj_idx.to(x.device).unsqueeze(1)
                spatial_attn_mask = torch.zeros(B * T, 1, N, N, device=x.device, dtype=dtype)
                spatial_attn_mask.masked_fill_(
                    ~same_object.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, 1, N, N),
                    torch.finfo(dtype).min,
                )
            if valid_token_mask is not None:
                spatial_key_mask = valid_token_mask.reshape(B * T, 1, 1, N)
                if spatial_attn_mask is None:
                    spatial_attn_mask = torch.zeros(B * T, 1, 1, N, device=x.device, dtype=dtype)
                spatial_attn_mask.masked_fill_(~spatial_key_mask, torch.finfo(dtype).min)

            # Temporal attention has one sequence per point identity.
            temporal_attn_mask = None
            if valid_token_mask is not None:
                temporal_key_mask = valid_token_mask.permute(0, 2, 1).reshape(B * N, 1, 1, T)
                temporal_attn_mask = torch.zeros(B * N, 1, 1, T, device=x.device, dtype=dtype)
                temporal_attn_mask.masked_fill_(~temporal_key_mask, torch.finfo(dtype).min)

            for block in self.blocks:
                x = block.forward_factorized(
                    x, e0, c_sim, dtype,
                    spatial_attn_mask=spatial_attn_mask,
                    temporal_attn_mask=temporal_attn_mask,
                )
        else:
            # Original flattened global-attention ablation.
            x = x.view(B, T * N, self.d_sim)
            attn_mask = None
            if self.use_object_local_attention and point_obj_idx is not None:
                L = T * N
                token_obj_idx = point_obj_idx.to(x.device).unsqueeze(1).expand(-1, T, -1).reshape(B, L)
                same_object = token_obj_idx.unsqueeze(2) == token_obj_idx.unsqueeze(1)
                attn_mask = torch.zeros(B, 1, L, L, device=x.device, dtype=dtype)
                attn_mask.masked_fill_(~same_object.unsqueeze(1), torch.finfo(dtype).min)
            if valid_seq_mask is not None:
                L = T * N
                if attn_mask is None:
                    attn_mask = torch.zeros(B, 1, 1, L, device=x.device, dtype=dtype)
                attn_mask.masked_fill_(~valid_seq_mask.view(B, 1, 1, L), torch.finfo(dtype).min)
            for block in self.blocks:
                x = block.forward_global(x, e0, dtype, attn_mask=attn_mask)
            x = x.view(B, T, N, self.d_sim)

        # Output head
        x = self.head_proj(self.head_norm(x.to(self.head_proj.weight.dtype)))  # (B, T, N, d_state)
        return x

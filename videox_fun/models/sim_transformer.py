# Simulation DiT (SimTransformer) for PV-Simulator.
# 10-block Transformer with self-attention + FFN, timestep modulation.
# No cross-attention (text conditioning is only in the video branch).
#
# Input: encoded point states (B, T, N, d_state) + init_enc (B, T, N, d_state)
#        + init_mask (B, T, N, 1) + conditions (B, T, N, d_cond)
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

    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, eps: float = 1e-6):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

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

        # No RoPE for simulation tokens.
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


class SimTransformer(nn.Module):
    """Simulation DiT: 10-block Transformer for physics trajectory denoising.

    Input: encoded noisy states x_enc (B, T, N, d_state), initial frame
    encoding init_enc (B, T, N, d_state) zero-padded beyond t=0, inpainting
    mask init_mask (B, T, N, 1) with 0=given/1=unknown, and conditions
    c_sim (B, T, N, d_cond). All concatenated and projected to hidden dim.

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
        d_sim: int = 256,
        ffn_dim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 10,
        freq_dim: int = 256,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.d_state = d_state
        self.d_cond = d_cond
        self.d_sim = d_sim
        self.num_layers = num_layers
        self.freq_dim = freq_dim

        # Input projection: [x_enc | init_enc | init_mask | c_sim] → d_sim
        # = 2 * d_state + 1 + d_cond
        self.input_proj = nn.Linear(2 * d_state + 1 + d_cond, d_sim)

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

        # Output head
        self.head_norm = WanLayerNorm(d_sim, eps)
        self.head_proj = nn.Linear(d_sim, d_state)

        self._init_weights()

    def _init_weights(self):
        """Zero-init head projection for stable training start."""
        nn.init.zeros_(self.head_proj.weight)
        nn.init.zeros_(self.head_proj.bias)

    def forward(self, x_enc, init_enc, init_mask, c_sim, t, dtype=torch.bfloat16,
                valid_seq_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x_enc: (B, T, N, d_state) — AE-encoded noisy point states.
            init_enc: (B, T, N, d_state) — AE-encoded initial frame, zero-padded beyond t=0.
            init_mask: (B, T, N, 1) — 0 at t=0 (given frame), 1 elsewhere.
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
        x = torch.cat([x_enc, init_enc, init_mask, c_sim], dim=-1)  # (B, T, N, 2*d_state+1+d_cond)
        x = self.input_proj(x)                   # (B, T, N, d_sim)
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


class SimSTAttentionBlock(nn.Module):
    """Spatial-then-temporal Transformer block for raw xyz diffusion.

    Spatial attention runs across points inside each frame. Temporal attention
    then runs across frames for each point trajectory.
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.norm_spatial = nn.LayerNorm(dim, eps=eps)
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_temporal = nn.LayerNorm(dim, eps=eps)
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    @staticmethod
    def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale) + shift

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        point_mask: Optional[torch.Tensor] = None,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, F, N, D)
            e: (B, 6, D)
            point_mask: optional (B, N) bool, True for valid points.
            frame_mask: optional (B, F) bool, True for valid frames.
        """
        B, F, N, D = x.shape
        shift_s, scale_s, gate_s, shift_t, scale_t, gate_t = (
            self.modulation + e
        ).chunk(6, dim=1)

        # Spatial MHA: (B, F, N, D) -> (B*F, N, D)
        h = self._modulate(
            self.norm_spatial(x),
            shift_s[:, None, :, :],
            scale_s[:, None, :, :],
        )
        h = h.reshape(B * F, N, D)
        spatial_key_padding_mask = None
        if point_mask is not None:
            spatial_key_padding_mask = (
                ~point_mask[:, None, :].expand(B, F, N).reshape(B * F, N)
            )
        y, _ = self.spatial_attn(
            h, h, h,
            key_padding_mask=spatial_key_padding_mask,
            need_weights=False,
        )
        y = y.reshape(B, F, N, D)
        x = x + y * gate_s[:, None, :, :]

        # Temporal MHA: (B, F, N, D) -> (B*N, F, D)
        h = self._modulate(
            self.norm_temporal(x),
            shift_t[:, None, :, :],
            scale_t[:, None, :, :],
        )
        h = h.permute(0, 2, 1, 3).reshape(B * N, F, D)
        temporal_key_padding_mask = None
        if frame_mask is not None:
            temporal_key_padding_mask = (
                ~frame_mask[:, None, :].expand(B, N, F).reshape(B * N, F)
            )
            if point_mask is not None:
                invalid_point_rows = ~point_mask.reshape(B * N)
                # Avoid all-masked rows inside MultiheadAttention. Invalid point
                # outputs are zeroed by the caller through the loss mask.
                temporal_key_padding_mask[invalid_point_rows] = False
        y, _ = self.temporal_attn(
            h, h, h,
            key_padding_mask=temporal_key_padding_mask,
            need_weights=False,
        )
        y = y.reshape(B, N, F, D).permute(0, 2, 1, 3)
        x = x + y * gate_t[:, None, :, :]

        x = x + self.ffn(self.norm_ffn(x))
        return x


class SimSTTransformer(nn.Module):
    """PhysCtrl-style raw xyz denoiser with spatial-temporal attention.

    The model operates directly on normalized xyz coordinates. When
    ``frame_cond`` is true, the caller prepends the clean initial frame to the
    noisy target sequence and this module drops that conditioning frame from
    the returned prediction. When ``pred_offset`` is true, the returned value is
    a residual relative to the initial point cloud.
    """

    def __init__(
        self,
        d_state: int = 3,
        d_cond: int = 34,
        d_sim: int = 256,
        ffn_dim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 10,
        freq_dim: int = 256,
        frame_cond: bool = True,
        pred_offset: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.d_state = d_state
        self.d_cond = d_cond
        self.d_sim = d_sim
        self.num_layers = num_layers
        self.freq_dim = freq_dim
        self.frame_cond = frame_cond
        self.pred_offset = pred_offset

        self.input_proj = nn.Linear(d_state + d_cond, d_sim)
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, d_sim),
            nn.SiLU(),
            nn.Linear(d_sim, d_sim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_sim, d_sim * 6),
        )
        self.blocks = nn.ModuleList([
            SimSTAttentionBlock(d_sim, ffn_dim, num_heads, eps)
            for _ in range(num_layers)
        ])
        self.head_norm = nn.LayerNorm(d_sim, eps=eps)
        self.head_proj = nn.Linear(d_sim, d_state)
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.head_proj.weight)
        nn.init.zeros_(self.head_proj.bias)

    def forward(
        self,
        x_xyz: torch.Tensor,
        c_sim: torch.Tensor,
        t: torch.Tensor,
        dtype=torch.bfloat16,
        point_mask: Optional[torch.Tensor] = None,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_xyz: (B, F, N, 3) normalized noisy xyz plus optional frame_cond.
            c_sim: (B, F, N, d_cond) condition features at the same frame count.
            t: (B,) diffusion timestep.
            point_mask: optional (B, N) bool.
            frame_mask: optional (B, F) bool.
        Returns:
            (B, F, N, 3), or (B, F-1, N, 3) when frame_cond=True.
        """
        B, F, N, _ = x_xyz.shape
        x = torch.cat([x_xyz, c_sim], dim=-1)
        x = self.input_proj(x)

        time_pos = sinusoidal_embedding_1d(
            self.d_sim, torch.arange(F, device=x.device)
        ).to(device=x.device, dtype=x.dtype)
        point_pos = sinusoidal_embedding_1d(
            self.d_sim, torch.arange(N, device=x.device)
        ).to(device=x.device, dtype=x.dtype)
        x = x + time_pos[None, :, None, :] + point_pos[None, None, :, :]

        emb_dtype = self.time_embedding[0].weight.dtype
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=emb_dtype)
        )
        e = self.time_projection(e).unflatten(1, (6, self.d_sim))
        e = e.to(x.dtype)

        for block in self.blocks:
            x = block(x, e, point_mask=point_mask, frame_mask=frame_mask)

        x = self.head_proj(self.head_norm(x.to(self.head_proj.weight.dtype)))
        if self.frame_cond:
            x = x[:, 1:]
        return x

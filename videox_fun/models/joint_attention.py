# Joint Attention module for PV-Simulator MoT architecture.
# Implements gated cross-modal attention between video and simulation branches.
#
# Design: bidirectional cross-attention (not full-sequence concatenation) because
# video sequences can be ~49k tokens while sim sequences are ~850 tokens.
#
# Gating: sigmoid gates initialized to -10 (≈0) for gradual transition from
# independent attention to joint attention during Stage 2 training.

import torch
import torch.nn as nn

from videox_fun.models.attention_utils import attention
from videox_fun.models.wan_transformer3d import WanRMSNorm


class CrossModalAttention(nn.Module):
    """One-directional cross-attention: query branch attends to key/value branch.

    Projects both branches to a shared attention dimension, computes cross-attention,
    then projects back to the query branch dimension.

    Args:
        q_dim: Query branch hidden dimension.
        kv_dim: Key/Value branch hidden dimension.
        d_joint: Shared attention dimension.
        num_heads: Number of attention heads.
        qk_norm: Whether to apply RMSNorm to Q/K.
        eps: Epsilon for normalization.
    """

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        d_joint: int = 256,
        num_heads: int = 8,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert d_joint % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_joint // num_heads

        self.q = nn.Linear(q_dim, d_joint)
        self.k = nn.Linear(kv_dim, d_joint)
        self.v = nn.Linear(kv_dim, d_joint)
        self.o = nn.Linear(d_joint, q_dim)

        self.norm_q = WanRMSNorm(d_joint, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(d_joint, eps=eps) if qk_norm else nn.Identity()

        # Zero-init output projection so initial residual contribution is 0
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)

    def forward(self, q_x, kv_x, dtype=torch.bfloat16):
        """
        Args:
            q_x: (B, L_q, q_dim) — query tokens.
            kv_x: (B, L_kv, kv_dim) — key/value tokens.
        Returns:
            (B, L_q, q_dim) — cross-attention output.
        """
        b = q_x.shape[0]
        n, d = self.num_heads, self.head_dim
        w_dtype = self.q.weight.dtype  # Use model weight dtype for linear ops

        q = self.norm_q(self.q(q_x.to(w_dtype))).view(b, -1, n, d)
        k = self.norm_k(self.k(kv_x.to(w_dtype))).view(b, -1, n, d)
        v = self.v(kv_x.to(w_dtype)).view(b, -1, n, d)

        x = attention(q.to(dtype), k.to(dtype), v=v.to(dtype))
        x = x.to(w_dtype).flatten(2)
        x = self.o(x)
        return x


class JointAttention(nn.Module):
    """Bidirectional gated cross-modal attention between video and sim branches.

    Two cross-attention directions:
    - sim queries attending to video K/V (sim learns from video context)
    - video queries attending to sim K/V (video learns from physics)

    Each direction has a learnable sigmoid gate initialized near 0 for gradual
    activation during Stage 2 training. An external bias_scale further controls
    the gating based on training progress or diffusion timestep.

    Args:
        d_vid: Video branch hidden dimension (e.g., 2048).
        d_sim: Simulation branch hidden dimension (e.g., 512).
        d_joint: Shared attention dimension for cross-modal projections.
        num_heads: Number of attention heads.
        init_gate: Initial gate value. sigmoid(-10) ≈ 0.
        qk_norm: Whether to apply RMSNorm to Q/K.
        eps: Epsilon for normalization.
    """

    def __init__(
        self,
        d_vid: int = 2048,
        d_sim: int = 512,
        d_joint: int = 256,
        num_heads: int = 8,
        init_gate: float = -10.0,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()

        # sim queries → video K/V
        self.sim_cross = CrossModalAttention(
            q_dim=d_sim, kv_dim=d_vid, d_joint=d_joint,
            num_heads=num_heads, qk_norm=qk_norm, eps=eps,
        )
        # video queries → sim K/V
        self.vid_cross = CrossModalAttention(
            q_dim=d_vid, kv_dim=d_sim, d_joint=d_joint,
            num_heads=num_heads, qk_norm=qk_norm, eps=eps,
        )

        # Learnable gates: sigmoid(init_gate) ≈ 0 at start
        self.sim_gate = nn.Parameter(torch.tensor(init_gate))
        self.vid_gate = nn.Parameter(torch.tensor(init_gate))

    def forward(
        self,
        vid_x: torch.Tensor,
        sim_x: torch.Tensor,
        bias_scale: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Apply bidirectional gated cross-modal attention.

        Args:
            vid_x: (B, L_vid, d_vid) — video branch tokens.
            sim_x: (B, L_sim, d_sim) — simulation branch tokens.
            bias_scale: Multiplier for the gate offset. Controls transition:
                - During Stage 2 training: decays from 1.0 → 0.0 over warmup steps.
                  At 1.0, an extra -10 offset keeps gates closed.
                  At 0.0, gates are free to open (learned values).
                - During inference: set to (1 - t/T_max) so gates open more at
                  high noise (structure) and close at low noise (detail).
            dtype: Compute dtype for attention.
        Returns:
            (vid_out, sim_out) — updated tokens for each branch.
        """
        # Extra offset that decays during training
        gate_offset = -10.0 * bias_scale

        # Cross-modal attention
        sim_delta = self.sim_cross(sim_x, vid_x, dtype)
        vid_delta = self.vid_cross(vid_x, sim_x, dtype)

        # Gated residual
        sim_gate = torch.sigmoid(self.sim_gate + gate_offset)
        vid_gate = torch.sigmoid(self.vid_gate + gate_offset)

        vid_out = vid_x + vid_gate * vid_delta
        sim_out = sim_x + sim_gate * sim_delta

        return vid_out, sim_out

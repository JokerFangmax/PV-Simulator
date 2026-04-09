# Causal Temporal Encoder/Decoder for PV-Simulator Simulation Branch.
# Design mirrors wan_vae.py's Encoder3d/Decoder3d, adapted to 1D temporal convolutions.
# 4x temporal compression via 2 layers of stride-2 causal conv (kernel=3).
#
# Math (encoder):
#   T_raw = 4*(T-1)+1 = 4T-3
#   Layer 1: 4T-3 → floor((4T-4)/2)+1 = 2T-1
#   Layer 2: 2T-1 → floor((2T-2)/2)+1 = T
#
# Math (decoder):
#   Layer 1: T → 2T (channel-doubling + interleave)
#   Layer 2: 2T → 4T
#   Trim last 3 frames → 4T-3 = T_raw

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Conv1d):
    """1D causal convolution: moves all padding to the left (past frames only).

    Follows the same pattern as CausalConv3d in wan_vae.py:
    causal_pad = 2 * padding[0], applied only on the left.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._causal_pad = 2 * self.padding[0]
        self.padding = (0,)

    def forward(self, x):
        if self._causal_pad > 0:
            x = F.pad(x, (self._causal_pad, 0))
        return super().forward(x)


class RMSNorm1d(nn.Module):
    """RMS normalization for 1D temporal tensors of shape (N, C, T).

    Normalizes along channel dim (dim=1), following wan_vae.py's RMS_norm pattern.
    """

    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1))  # (C, 1) broadcasts over T

    def forward(self, x):
        return F.normalize(x, dim=1) * self.scale * self.gamma


class ResidualBlock1d(nn.Module):
    """Residual block for 1D temporal data: RMSNorm → SiLU → CausalConv1d × 2 + shortcut.

    Mirrors wan_vae.py's ResidualBlock but adapted to 1D (no spatial dims).
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.residual = nn.Sequential(
            RMSNorm1d(in_dim),
            nn.SiLU(),
            CausalConv1d(in_dim, out_dim, kernel_size=3, padding=1),
            RMSNorm1d(out_dim),
            nn.SiLU(),
            CausalConv1d(out_dim, out_dim, kernel_size=3, padding=1),
        )
        self.shortcut = (
            CausalConv1d(in_dim, out_dim, kernel_size=1)
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)


class CausalDownsample1d(nn.Module):
    """Temporal downsample 2x via stride-2 causal conv (kernel=3).

    For input length T_in, output length = floor((T_in - 1) / 2) + 1.
    """

    def __init__(self, dim):
        super().__init__()
        self.conv = CausalConv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class CausalUpsample1d(nn.Module):
    """Temporal upsample 2x via channel-doubling + interleave.

    Mirrors wan_vae.py's upsample3d: CausalConv produces 2C channels,
    then reshape + interleave to double the temporal dimension.
    """

    def __init__(self, dim):
        super().__init__()
        self.conv = CausalConv1d(dim, dim * 2, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv(x)  # (N, 2C, T)
        N, C2, T = x.shape
        C = C2 // 2
        # Interleave: split into two halves, stack along time
        x = x.reshape(N, 2, C, T)
        x = x.permute(0, 2, 3, 1).reshape(N, C, T * 2)  # (N, C, 2T)
        return x


class CausalTemporalEncoder(nn.Module):
    """4x causal temporal encoder: (1, T_raw, N, C_in) → (1, T, N, d_out).

    T_raw = 4*(T-1)+1. Uses 2 stride-2 downsample layers with residual blocks.

    Args:
        c_in: Input channels (e.g., 9 for pos+vel+ang_vel).
        c_mid: Intermediate channel width.
        d_out: Output feature dimension.
    """

    def __init__(self, c_in: int, c_mid: int, d_out: int):
        super().__init__()
        self.encoder = nn.Sequential(
            CausalConv1d(c_in, c_mid, kernel_size=3, padding=1),
            ResidualBlock1d(c_mid, c_mid),
            CausalDownsample1d(c_mid),        # T_raw → 2T-1
            ResidualBlock1d(c_mid, c_mid),
            CausalDownsample1d(c_mid),        # 2T-1 → T
            ResidualBlock1d(c_mid, d_out),
        )

    def forward(self, x):
        """
        Args:
            x: (1, T_raw, N, C_in)
        Returns:
            (1, T, N, d_out)
        """
        B, T_raw, N, C = x.shape
        # Treat N as batch dim, C as channels, T as temporal
        x = x.squeeze(0).permute(1, 2, 0)      # (N, C, T_raw)
        x = self.encoder(x)                      # (N, d_out, T)
        T = x.shape[2]
        x = x.permute(2, 0, 1).unsqueeze(0)     # (1, T, N, d_out)
        return x


class CausalTemporalDecoder(nn.Module):
    """4x causal temporal decoder: (1, T, N, d_feat) → (1, T_raw, N, c_out).

    Uses 2 upsample-by-2 layers: T → 2T → 4T, then trims to T_raw = 4T-3.

    Args:
        d_feat: Input feature dimension from DiT.
        c_mid: Intermediate channel width.
        c_out: Output channels (e.g., 6 for position+velocity prediction).
    """

    def __init__(self, d_feat: int, c_mid: int, c_out: int):
        super().__init__()
        self.decoder = nn.Sequential(
            CausalConv1d(d_feat, c_mid, kernel_size=3, padding=1),
            ResidualBlock1d(c_mid, c_mid),
            CausalUpsample1d(c_mid),             # T → 2T
            ResidualBlock1d(c_mid, c_mid),
            CausalUpsample1d(c_mid),             # 2T → 4T
        )
        self.head = CausalConv1d(c_mid, c_out, kernel_size=3, padding=1)

    def forward(self, x, t_raw: int):
        """
        Args:
            x: (1, T, N, d_feat)
            t_raw: Target temporal length (= 4*(T-1)+1).
        Returns:
            (1, T_raw, N, c_out)
        """
        B, T, N, D = x.shape
        x = x.squeeze(0).permute(1, 2, 0)       # (N, d_feat, T)
        x = self.decoder(x)                       # (N, c_mid, 4T)
        x = x[:, :, :t_raw]                       # trim to T_raw = 4T-3
        x = self.head(x)                           # (N, c_out, T_raw)
        x = x.permute(2, 0, 1).unsqueeze(0)       # (1, T_raw, N, c_out)
        return x

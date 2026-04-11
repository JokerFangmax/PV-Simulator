# Causal temporal building blocks and legacy encoder/decoder for PV-Simulator.
# Building blocks (CausalConv1d, ResidualBlock1d, etc.) are reused by both:
#   - CausalAE (sim_ae.py) for pre-trained state encoding (Stage 0+)
#   - CausalTemporalEncoder below for force condition encoding (sim_condition.py)
#
# NOTE: For point-state encoding, use CausalAE from sim_ae.py instead of
# CausalTemporalEncoder/Decoder below. The AE is pre-trained in Stage 0
# and frozen for subsequent stages.
#
# 4x temporal compression via 2 layers of stride-2 causal conv (kernel=3).
# Math: T_raw = 4k+1 → T_latent = k+1

import torch
import torch.nn as nn
import torch.nn.functional as F


CACHE_T = 2


class CausalConv1d(nn.Conv1d):
    """1D causal convolution with optional cache for chunked processing.

    Follows CausalConv3d in wan_vae.py: all temporal padding goes to the left.
    When ``cache_x`` is provided, it replaces part of the zero-padding with
    real data from the previous chunk.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._causal_pad = 2 * self.padding[0]
        self.padding = (0,)

    def forward(self, x, cache_x=None):
        pad = self._causal_pad
        if cache_x is not None and pad > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            pad -= cache_x.shape[2]
        if pad > 0:
            x = F.pad(x, (pad, 0))
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
    """Residual block with optional cache support for chunked processing.

    Architecture: RMSNorm → SiLU → CausalConv1d → RMSNorm → SiLU → CausalConv1d
    Plus shortcut (identity or 1×1 conv).

    When ``feat_cache`` / ``feat_idx`` are provided, each CausalConv1d layer
    stores and retrieves its tail frames from the cache, exactly like
    wan_vae.py's ResidualBlock.
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

    def forward(self, x, feat_cache=None, feat_idx=None):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv1d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:].clone()
                if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1:].to(cache_x.device),
                        cache_x,
                    ], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


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
    """4x causal temporal encoder: (B, T_raw, N, c_in) → (B, T, N, d_out).

    T_raw = 4k+1, T_latent = k+1. Uses 2 stride-2 downsample layers with residual blocks.

    Args:
        c_in: Input channels (e.g., 6 for pos+vel).
        c_mid: Intermediate channel width.
        d_out: Output feature dimension.
    """

    def __init__(self, c_in: int, c_mid: int, d_out: int):
        super().__init__()
        self.encoder = nn.Sequential(
            CausalConv1d(c_in, c_mid, kernel_size=3, padding=1),
            ResidualBlock1d(c_mid, c_mid),
            CausalDownsample1d(c_mid),        # 4k+1 → 2k+1
            ResidualBlock1d(c_mid, c_mid),
            CausalDownsample1d(c_mid),        # 2k+1 → k+1
            ResidualBlock1d(c_mid, d_out),
        )

    def forward(self, x):
        """
        Args:
            x: (B, T_raw, N, c_in)
        Returns:
            (B, T, N, d_out)
        """
        B, T_raw, N, C = x.shape
        # Reshape: treat N as an independent batch axis, C as channels, T as time
        x = x.permute(0, 2, 3, 1).reshape(B * N, C, T_raw)  # (B*N, C, T_raw)
        x = self.encoder(x)                                    # (B*N, d_out, T)
        T = x.shape[2]
        x = x.reshape(B, N, -1, T).permute(0, 3, 1, 2)       # (B, T, N, d_out)
        return x


class CausalTemporalDecoder(nn.Module):
    """4x causal temporal decoder: (B, T, N, d_feat) → (B, T_raw, N, c_out).

    Uses 2 upsample-by-2 layers: T → 2T → 4T, then trims to T_raw.

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
            x: (B, T, N, d_feat)
            t_raw: Target temporal length (= 4k+1).
        Returns:
            (B, T_raw, N, c_out)
        """
        B, T, N, D = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * N, D, T)         # (B*N, d_feat, T)
        x = self.decoder(x)                                       # (B*N, c_mid, 4T)
        x = x[:, :, :t_raw]                                       # trim to T_raw
        x = self.head(x)                                           # (B*N, c_out, T_raw)
        x = x.reshape(B, N, -1, t_raw).permute(0, 3, 1, 2)       # (B, T_raw, N, c_out)
        return x

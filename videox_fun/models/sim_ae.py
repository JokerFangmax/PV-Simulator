# Causal Autoencoder for PV-Simulator Stage 0.
# Compresses (B, T_raw, N, 3) → (B, T, N, 16) with 4x temporal reduction.
# Trained with WAE-style losses on synthesized data, then frozen for Stages 1+2.
# Applied separately to pos(3) and vel(3) → 32 total latent dims per point.

import os
from typing import Optional

import torch
import torch.nn as nn

from videox_fun.models.sim_causal_encoder import (
    CausalConv1d,
    CausalDownsample1d,
    CausalUpsample1d,
    ResidualBlock1d,
)


class CausalAEEncoder(nn.Module):
    """4x causal temporal encoder: (B, T_raw, N, c_in) → (B, T, N, d_latent).

    Architecture:
      CausalConv1d(c_in, c_mid, k=3) → ResidualBlock1d(c_mid, c_mid) →
      CausalDownsample1d(c_mid) → ResidualBlock1d(c_mid, c_mid) →
      CausalDownsample1d(c_mid) → ResidualBlock1d(c_mid, d_latent)

    Args:
        c_in: Input channels (e.g. 3 for position or velocity).
        c_mid: Intermediate channel width.
        d_latent: Output latent dimension.
    """

    def __init__(self, c_in: int = 3, c_mid: int = 64, d_latent: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            CausalConv1d(c_in, c_mid, kernel_size=3, padding=1),
            ResidualBlock1d(c_mid, c_mid),
            CausalDownsample1d(c_mid),       # T_raw → ~T_raw/2
            ResidualBlock1d(c_mid, c_mid),
            CausalDownsample1d(c_mid),       # → ~T_raw/4
            ResidualBlock1d(c_mid, d_latent),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T_raw, N, c_in)
        Returns:
            (B, T, N, d_latent) where T = (T_raw - 1) // 4 + 1
        """
        B, T_raw, N, C = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * N, C, T_raw)  # (B*N, C, T_raw)
        x = self.encoder(x)                                    # (B*N, d_latent, T)
        T = x.shape[2]
        x = x.reshape(B, N, -1, T).permute(0, 3, 1, 2)       # (B, T, N, d_latent)
        return x


class CausalAEDecoder(nn.Module):
    """4x causal temporal decoder: (B, T, N, d_latent) → (B, T_raw, N, c_out).

    Architecture:
      CausalConv1d(d_latent, c_mid, k=3) → ResidualBlock1d(c_mid, c_mid) →
      CausalUpsample1d(c_mid) → ResidualBlock1d(c_mid, c_mid) →
      CausalUpsample1d(c_mid) → CausalConv1d(c_mid, c_out, k=3), trim to T_raw

    Args:
        d_latent: Input latent dimension.
        c_mid: Intermediate channel width.
        c_out: Output channels (e.g. 3 for position or velocity).
    """

    def __init__(self, d_latent: int = 16, c_mid: int = 64, c_out: int = 3):
        super().__init__()
        self.decoder = nn.Sequential(
            CausalConv1d(d_latent, c_mid, kernel_size=3, padding=1),
            ResidualBlock1d(c_mid, c_mid),
            CausalUpsample1d(c_mid),          # T → 2T
            ResidualBlock1d(c_mid, c_mid),
            CausalUpsample1d(c_mid),          # 2T → 4T
        )
        self.head = CausalConv1d(c_mid, c_out, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor, t_raw: int) -> torch.Tensor:
        """
        Args:
            z: (B, T, N, d_latent)
            t_raw: Target temporal length (= 4k+1).
        Returns:
            (B, T_raw, N, c_out)
        """
        B, T, N, D = z.shape
        z = z.permute(0, 2, 3, 1).reshape(B * N, D, T)      # (B*N, d_latent, T)
        z = self.decoder(z)                                    # (B*N, c_mid, ~4T)
        z = z[:, :, :t_raw]                                    # trim to T_raw
        z = self.head(z)                                       # (B*N, c_out, T_raw)
        z = z.reshape(B, N, -1, t_raw).permute(0, 3, 1, 2)   # (B, T_raw, N, c_out)
        return z


class CausalAE(nn.Module):
    """Causal Autoencoder: 4x temporal compression for 3D trajectories.

    Trained in Stage 0 with WAE-style losses, then frozen for Stages 1+2.
    Applied separately to pos(3) and vel(3) slices of point states.

    Args:
        c_in: Input/output channels (default 3 for pos or vel).
        c_mid: Intermediate channel width.
        d_latent: Latent dimension per channel group.
    """

    def __init__(self, c_in: int = 3, c_mid: int = 64, d_latent: int = 16):
        super().__init__()
        self.c_in = c_in
        self.c_mid = c_mid
        self.d_latent = d_latent
        self.encoder = CausalAEEncoder(c_in, c_mid, d_latent)
        self.decoder = CausalAEDecoder(d_latent, c_mid, c_in)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw trajectories to latent.

        Args:
            x: (B, T_raw, N, c_in) or (B, T_raw, c_in) — if 3D, N=1 is assumed.
        Returns:
            (B, T, N, d_latent) or (B, T, d_latent)
        """
        squeeze = x.ndim == 3
        if squeeze:
            x = x.unsqueeze(2)  # (B, T_raw, 1, c_in)
        z = self.encoder(x)
        if squeeze:
            z = z.squeeze(2)
        return z

    def decode(self, z: torch.Tensor, t_raw: int) -> torch.Tensor:
        """Decode latent back to raw trajectories.

        Args:
            z: (B, T, N, d_latent) or (B, T, d_latent)
            t_raw: Target temporal length.
        Returns:
            (B, T_raw, N, c_in) or (B, T_raw, c_in)
        """
        squeeze = z.ndim == 3
        if squeeze:
            z = z.unsqueeze(2)
        x_hat = self.decoder(z, t_raw)
        if squeeze:
            x_hat = x_hat.squeeze(2)
        return x_hat

    def forward(self, x: torch.Tensor):
        """Full encode-decode pass.

        Args:
            x: (B, T_raw, N, c_in) or (B, T_raw, c_in)
        Returns:
            (x_hat, z) — reconstruction and latent.
        """
        squeeze = x.ndim == 3
        if squeeze:
            x = x.unsqueeze(2)
        t_raw = x.shape[1]
        z = self.encoder(x)
        x_hat = self.decoder(z, t_raw)
        if squeeze:
            x_hat = x_hat.squeeze(2)
            z = z.squeeze(2)
        return x_hat, z

    def save(self, path: str):
        """Save model config and weights."""
        os.makedirs(path, exist_ok=True)
        config = {
            'c_in': self.c_in,
            'c_mid': self.c_mid,
            'd_latent': self.d_latent,
        }
        torch.save(config, os.path.join(path, 'config.pt'))
        torch.save(self.state_dict(), os.path.join(path, 'causal_ae.pt'))

    @classmethod
    def load(cls, path: str, map_location='cpu') -> 'CausalAE':
        """Load model from checkpoint directory."""
        config = torch.load(os.path.join(path, 'config.pt'), map_location=map_location)
        model = cls(**config)
        state_dict = torch.load(os.path.join(path, 'causal_ae.pt'), map_location=map_location)
        model.load_state_dict(state_dict)
        return model

# Causal Autoencoder for PV-Simulator Stage 0.
# Compresses (B, T_raw, N, 3) → (B, T, N, 16) with 4x temporal reduction.
# Trained with WAE-style losses on synthesized data, then frozen for Stages 1+2.
# Applied separately to pos(3) and vel(3) → 32 total latent dims per point.
#
# Follows WAN VAE's chunked causal design (wan_vae.py):
#   Encoder: chunks input as [1, 4, 4, ...] frames, uses feat_cache across chunks
#   Decoder: processes one latent frame at a time, producing [1, 4, 4, ...] output
#   Temporal compression is achieved through the chunking + cache mechanism,
#   NOT through simple strided convolution on the full sequence.

import os
from typing import Optional

import torch
import torch.nn as nn

from videox_fun.models.sim_causal_encoder import (
    CACHE_T,
    CausalConv1d,
    ResidualBlock1d,
)


def _count_causal_conv1d(model):
    """Count CausalConv1d modules for cache slot allocation."""
    return sum(1 for m in model.modules() if isinstance(m, CausalConv1d))


class _Downsample1d(nn.Module):
    """Temporal downsample 2x with cache, following WAN VAE's downsample3d.

    First chunk: cache the input, pass through unchanged (T stays the same).
    Later chunks: prepend cached last frame, apply stride-2 conv.
    """

    def __init__(self, dim):
        super().__init__()
        # No causal padding — temporal context comes from the cache.
        self.time_conv = CausalConv1d(
            dim, dim, kernel_size=3, stride=2, padding=0,
        )

    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = x.clone()
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -1:].clone()
                x = self.time_conv(
                    torch.cat([feat_cache[idx][:, :, -1:], x], dim=2)
                )
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
        return x


class _Upsample1d(nn.Module):
    """Temporal upsample 2x with cache, following WAN VAE's upsample3d.

    First chunk: mark cache as 'Rep', pass through unchanged (T stays the same).
    Later chunks: apply time_conv (2C channels) + interleave to double T.
    """

    def __init__(self, dim):
        super().__init__()
        self.time_conv = CausalConv1d(
            dim, dim * 2, kernel_size=3, padding=1,
        )

    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = 'Rep'
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -CACHE_T:].clone()
                if (cache_x.shape[2] < CACHE_T
                        and feat_cache[idx] is not None
                        and feat_cache[idx] != 'Rep'):
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1:].to(cache_x.device),
                        cache_x,
                    ], dim=2)
                if (cache_x.shape[2] < CACHE_T
                        and feat_cache[idx] is not None
                        and feat_cache[idx] == 'Rep'):
                    cache_x = torch.cat([
                        torch.zeros_like(cache_x).to(cache_x.device),
                        cache_x,
                    ], dim=2)

                if feat_cache[idx] == 'Rep':
                    x = self.time_conv(x)
                else:
                    x = self.time_conv(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1

                # Interleave: (N, 2C, T) → (N, C, 2T)
                N, C2, T = x.shape
                C = C2 // 2
                x = x.reshape(N, 2, C, T)
                x = torch.stack([x[:, 0], x[:, 1]], dim=3).reshape(N, C, T * 2)
        return x


class CausalAEEncoder(nn.Module):
    """4x causal temporal encoder with cache support.

    Architecture::

        CausalConv1d(c_in → c_mid, k=3)
        ResidualBlock1d(c_mid → c_mid)
        Downsample(c_mid)                  # T → ~T/2 (via cache)
        ResidualBlock1d(c_mid → c_mid)
        Downsample(c_mid)                  # → ~T/4 (via cache)
        ResidualBlock1d(c_mid → d_latent)

    Must be called with ``feat_cache`` / ``feat_idx`` for correct temporal
    compression (see :class:`CausalAE`).
    """

    def __init__(self, c_in: int = 3, c_mid: int = 64, d_latent: int = 16):
        super().__init__()
        self.conv1 = CausalConv1d(c_in, c_mid, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            ResidualBlock1d(c_mid, c_mid),
            _Downsample1d(c_mid),
            ResidualBlock1d(c_mid, c_mid),
            _Downsample1d(c_mid),
            ResidualBlock1d(c_mid, d_latent),
        ])

    def forward(self, x, feat_cache=None, feat_idx=None):
        # conv1 with cache
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:].clone()
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1:].to(cache_x.device),
                    cache_x,
                ], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for block in self.blocks:
            if feat_cache is not None:
                x = block(x, feat_cache, feat_idx)
            else:
                x = block(x)

        return x


class CausalAEDecoder(nn.Module):
    """4x causal temporal decoder with cache support.

    Architecture::

        CausalConv1d(d_latent → c_mid, k=3)
        ResidualBlock1d(c_mid → c_mid)
        Upsample(c_mid)                    # T → 2T (via cache)
        ResidualBlock1d(c_mid → c_mid)
        Upsample(c_mid)                    # → 4T (via cache)
        CausalConv1d(c_mid → c_out, k=3)   # head

    Must be called with ``feat_cache`` / ``feat_idx`` (see :class:`CausalAE`).
    """

    def __init__(self, d_latent: int = 16, c_mid: int = 64, c_out: int = 3):
        super().__init__()
        self.conv1 = CausalConv1d(d_latent, c_mid, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            ResidualBlock1d(c_mid, c_mid),
            _Upsample1d(c_mid),
            ResidualBlock1d(c_mid, c_mid),
            _Upsample1d(c_mid),
        ])
        self.head = CausalConv1d(c_mid, c_out, kernel_size=3, padding=1)

    def forward(self, x, feat_cache=None, feat_idx=None):
        # conv1 with cache
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:].clone()
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1:].to(cache_x.device),
                    cache_x,
                ], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for block in self.blocks:
            if feat_cache is not None:
                x = block(x, feat_cache, feat_idx)
            else:
                x = block(x)

        # head with cache
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:].clone()
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1:].to(cache_x.device),
                    cache_x,
                ], dim=2)
            x = self.head(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.head(x)

        return x


class CausalAE(nn.Module):
    """Causal Autoencoder: 4x temporal compression for 3D trajectories.

    Follows WAN VAE's chunked causal architecture:
    - Encoder splits input into chunks [1, 4, 4, ...], processes each through
      the encoder network with ``feat_cache`` carrying context across chunks.
    - Decoder processes one latent frame at a time, producing [1, 4, 4, ...]
      output frames, again using ``feat_cache`` for cross-chunk context.

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

    @staticmethod
    def _init_cache(model):
        return [None] * _count_causal_conv1d(model)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw trajectories to latent with chunked processing.

        Input is split into chunks [T=1, T=4, T=4, ...] along the time axis.
        Each chunk is fed through the encoder with ``feat_cache`` propagating
        temporal context from previous chunks (like WAN VAE's ``encode``).

        Args:
            x: ``(B, T_raw, N, c_in)`` or ``(B, T_raw, c_in)`` — if 3-D, N=1 is assumed.
        Returns:
            ``(B, T, N, d_latent)`` or ``(B, T, d_latent)``
            where ``T = (T_raw - 1) // 4 + 1``.
        """
        squeeze = x.ndim == 3
        if squeeze:
            x = x.unsqueeze(2)

        B, T_raw, N, C = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * N, C, T_raw)  # (BN, C, T_raw)

        enc_cache = self._init_cache(self.encoder)
        iter_ = 1 + (T_raw - 1) // 4

        for i in range(iter_):
            feat_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1],
                    feat_cache=enc_cache, feat_idx=feat_idx,
                )
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1):1 + 4 * i],
                    feat_cache=enc_cache, feat_idx=feat_idx,
                )
                out = torch.cat([out, out_], dim=2)

        T = out.shape[2]
        z = out.reshape(B, N, -1, T).permute(0, 3, 1, 2)  # (B, T, N, d_latent)

        if squeeze:
            z = z.squeeze(2)
        return z

    def decode(self, z: torch.Tensor, t_raw: int) -> torch.Tensor:
        """Decode latent back to raw trajectories with chunked processing.

        Each latent frame is fed through the decoder one at a time.
        ``feat_cache`` carries context from all previous frames, so every
        output frame has access to all preceding latent information
        (unlike a single-pass causal decoder where receptive field is limited).

        Args:
            z: ``(B, T, N, d_latent)`` or ``(B, T, d_latent)``
            t_raw: Target temporal length (= 4k+1).
        Returns:
            ``(B, T_raw, N, c_in)`` or ``(B, T_raw, c_in)``
        """
        squeeze = z.ndim == 3
        if squeeze:
            z = z.unsqueeze(2)

        B, T, N, D = z.shape
        z = z.permute(0, 2, 3, 1).reshape(B * N, D, T)  # (BN, D, T)

        dec_cache = self._init_cache(self.decoder)

        for i in range(T):
            feat_idx = [0]
            if i == 0:
                out = self.decoder(
                    z[:, :, i:i + 1],
                    feat_cache=dec_cache, feat_idx=feat_idx,
                )
            else:
                out_ = self.decoder(
                    z[:, :, i:i + 1],
                    feat_cache=dec_cache, feat_idx=feat_idx,
                )
                out = torch.cat([out, out_], dim=2)

        out = out[:, :, :t_raw]
        out = out.reshape(B, N, -1, t_raw).permute(0, 3, 1, 2)  # (B, T_raw, N, c_in)

        if squeeze:
            out = out.squeeze(2)
        return out

    def forward(self, x: torch.Tensor):
        """Full encode-decode pass.

        Args:
            x: ``(B, T_raw, N, c_in)`` or ``(B, T_raw, c_in)``
        Returns:
            ``(x_hat, z)`` — reconstruction and latent.
        """
        squeeze = x.ndim == 3
        if squeeze:
            x = x.unsqueeze(2)

        t_raw = x.shape[1]
        z = self.encode(x)
        x_hat = self.decode(z, t_raw)

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
        config = torch.load(
            os.path.join(path, 'config.pt'),
            map_location=map_location, weights_only=True,
        )
        model = cls(**config)
        state_dict = torch.load(
            os.path.join(path, 'causal_ae.pt'),
            map_location=map_location, weights_only=True,
        )
        model.load_state_dict(state_dict)
        return model

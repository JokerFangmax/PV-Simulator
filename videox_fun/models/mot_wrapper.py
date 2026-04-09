# Mixture-of-Transformers (MoT) Wrapper for PV-Simulator.
# Orchestrates the paired forward pass of Video Branch (WanTransformer3DModel)
# and Simulation Branch (SimTransformer) with JointAttention at paired blocks.
#
# Block pairing: dense at shallow layers, dilated at deep layers.
# Stage 1: SimTransformer runs standalone (no MoTWrapper needed).
# Stage 2: MoTWrapper manages both branches with gated cross-modal attention.

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.cuda.amp as amp
from diffusers.utils.import_utils import is_torch_version

from videox_fun.models.wan_transformer3d import (
    WanTransformer3DModel,
    sinusoidal_embedding_1d,
)
from videox_fun.models.sim_transformer import SimTransformer
from videox_fun.models.sim_causal_encoder import CausalTemporalEncoder, CausalTemporalDecoder
from videox_fun.models.sim_condition import SimConditionEmbedder
from videox_fun.models.joint_attention import JointAttention


# Default block pairing schedule:
# Video has 30 blocks (0-29); Sim has 10 blocks (0-9)
# Dense first 4 blocks, then dilated every 4 video blocks.
DEFAULT_PAIRING = {
    0: 0, 1: 1, 2: 2, 3: 3,                          # dense
    8: 4, 12: 5, 16: 6, 20: 7, 24: 8, 28: 9,         # dilated
}


class MoTWrapper(nn.Module):
    """Mixture-of-Transformers: wraps Video + Sim branches with JointAttention.

    Manages:
    - Causal temporal encoder/decoder for point states
    - Condition embedding for physics properties
    - Interleaved block execution with joint attention at paired blocks
    - Bias scale scheduling for gradual attention transition

    Args:
        video_transformer: Pre-trained WanTransformer3DModel (frozen or LoRA-wrapped).
        d_state: Encoded point state dimension.
        d_cond: Condition embedding dimension (from SimConditionEmbedder).
        d_sim: Simulation branch hidden dimension.
        sim_ffn_dim: Simulation branch FFN dimension.
        sim_num_heads: Simulation branch attention heads.
        sim_num_layers: Number of simulation transformer blocks.
        d_joint: Shared dimension for cross-modal attention.
        joint_num_heads: Cross-modal attention heads.
        init_gate: Initial gate value for JointAttention (sigmoid ≈ 0).
        pairing: Dict mapping video_block_idx → sim_block_idx.
        max_objects: Maximum number of objects per scene.
        state_enc_mid: Intermediate channels for causal temporal encoder.
        force_enc_mid: Intermediate channels for force encoder.
    """

    def __init__(
        self,
        video_transformer: WanTransformer3DModel,
        d_state: int = 256,
        d_cond: int = 432,
        d_sim: int = 512,
        sim_ffn_dim: int = 2048,
        sim_num_heads: int = 8,
        sim_num_layers: int = 10,
        d_joint: int = 256,
        joint_num_heads: int = 8,
        init_gate: float = -10.0,
        pairing: Optional[Dict[int, int]] = None,
        max_objects: int = 16,
        state_enc_mid: int = 128,
        force_enc_mid: int = 128,
    ):
        super().__init__()

        self.video = video_transformer
        self.pairing = pairing or DEFAULT_PAIRING

        # Simulation branch
        self.sim = SimTransformer(
            d_state=d_state,
            d_cond=d_cond,
            d_sim=d_sim,
            ffn_dim=sim_ffn_dim,
            num_heads=sim_num_heads,
            num_layers=sim_num_layers,
        )

        # Causal temporal encoder/decoder for point states
        self.x_s_encoder = CausalTemporalEncoder(9, state_enc_mid, d_state)
        self.x_s_decoder = CausalTemporalDecoder(d_state, state_enc_mid, 6)

        # Condition embedder
        self.sim_cond_embedder = SimConditionEmbedder(
            d_cond=(d_cond - 0),  # SimConditionEmbedder computes d_cond internally
            max_objects=max_objects,
            force_encoder_mid=force_enc_mid,
        )

        # Joint attention modules — one per paired block
        d_vid = video_transformer.dim
        self.joint_attns = nn.ModuleDict({
            str(vid_idx): JointAttention(
                d_vid=d_vid,
                d_sim=d_sim,
                d_joint=d_joint,
                num_heads=joint_num_heads,
                init_gate=init_gate,
            )
            for vid_idx in self.pairing.keys()
        })

    def forward_joint(
        self,
        # Video inputs (same as WanTransformer3DModel.forward)
        x: List[torch.Tensor],
        t: torch.Tensor,
        context: List[torch.Tensor],
        seq_len: int,
        clip_fea: Optional[torch.Tensor] = None,
        y: Optional[List[torch.Tensor]] = None,
        # Simulation inputs
        x_s_enc: torch.Tensor = None,      # (1, T, N, d_state) — already encoded
        sim_e0: torch.Tensor = None,        # (1, 6, d_sim) — sim timestep modulation
        sim_x: torch.Tensor = None,         # (1, T*N, d_sim) — sim tokens after input_proj
        # Joint attention control
        bias_scale: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Run interleaved forward pass of both branches with joint attention.

        The video branch preprocessing (patch embedding, text embedding, timestep
        embedding) is replicated from WanTransformer3DModel.forward(). The block
        loop is then run manually, injecting joint attention at paired blocks.

        Args:
            x, t, context, seq_len, clip_fea, y: Video branch inputs (see WanTransformer3DModel).
            sim_x: Pre-processed simulation tokens (1, T*N, d_sim).
            sim_e0: Simulation timestep modulation (1, 6, d_sim).
            bias_scale: Joint attention gate control (1.0 = closed, 0.0 = open).
            dtype: Compute dtype.

        Returns:
            vid_out: List[Tensor] — denoised video tensors.
            sim_out: (1, T*N, d_sim) — simulation branch output tokens.
        """
        video = self.video
        device = video.patch_embedding.weight.device

        if video.freqs.device != device:
            video.freqs = video.freqs.to(device)

        # --- Video Branch Preprocessing (replicates WanTransformer3DModel.forward) ---
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x = [video.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        vid_x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
            for u in x
        ])

        # Time embeddings
        with amp.autocast(dtype=torch.float32):
            if t.dim() != 1:
                bt = t.size(0)
                ft = t.flatten()
                e = video.time_embedding(
                    sinusoidal_embedding_1d(video.freq_dim, ft).unflatten(0, (bt, seq_len)).float())
                vid_e0 = video.time_projection(e).unflatten(2, (6, video.dim))
            else:
                e = video.time_embedding(
                    sinusoidal_embedding_1d(video.freq_dim, t).float())
                vid_e0 = video.time_projection(e).unflatten(1, (6, video.dim))

        # Context (text) embeddings
        context_lens = None
        context = video.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(video.text_len - u.size(0), u.size(1))])
                for u in context
            ]))
        if clip_fea is not None:
            context_clip = video.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)

        # --- Interleaved Block Loop ---
        vid_kwargs = dict(
            e=vid_e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=video.freqs,
            context=context,
            context_lens=context_lens,
            dtype=dtype,
            t=t,
        )

        for vid_idx, vid_block in enumerate(video.blocks):
            vid_x = vid_block(vid_x, **vid_kwargs)

            if vid_idx in self.pairing:
                sim_idx = self.pairing[vid_idx]
                # Run corresponding sim block
                sim_x = self.sim.blocks[sim_idx](sim_x, sim_e0, dtype)
                # Apply joint attention
                joint_attn = self.joint_attns[str(vid_idx)]
                vid_x, sim_x = joint_attn(vid_x, sim_x, bias_scale=bias_scale, dtype=dtype)

        # --- Video Branch Postprocessing ---
        if torch.is_grad_enabled() and video.gradient_checkpointing:
            ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            vid_x = torch.utils.checkpoint.checkpoint(
                lambda x_in, e_in: video.head(x_in, e_in), vid_x, e, **ckpt_kwargs
            )
        else:
            vid_x = video.head(vid_x, e)

        vid_out = video.unpatchify(vid_x, grid_sizes)
        vid_out = torch.stack(vid_out)

        return vid_out, sim_x

    def prepare_sim_inputs(
        self,
        noisy_x_s_enc: torch.Tensor,   # (1, T, N, d_state)
        c_sim: torch.Tensor,            # (1, T, N, d_cond)
        t: torch.Tensor,                # (1,) diffusion timestep
    ):
        """Prepare simulation tokens and modulation for the interleaved loop.

        Returns:
            sim_x: (1, T*N, d_sim) — projected sim tokens.
            sim_e0: (1, 6, d_sim) — sim timestep modulation.
        """
        B, T, N, _ = noisy_x_s_enc.shape

        # Concatenate and project
        x = torch.cat([noisy_x_s_enc, c_sim], dim=-1)
        x = self.sim.input_proj(x)
        sim_x = x.view(B, T * N, self.sim.d_sim)

        # Timestep modulation
        with amp.autocast(dtype=torch.float32):
            e = self.sim.time_embedding(
                sinusoidal_embedding_1d(self.sim.freq_dim, t).float()
            )
            sim_e0 = self.sim.time_projection(e).unflatten(1, (6, self.sim.d_sim))

        return sim_x, sim_e0

    def decode_sim_output(
        self,
        sim_x: torch.Tensor,   # (1, T*N, d_sim) — raw sim output from interleaved loop
        T: int,
        N: int,
        T_raw: int,
    ):
        """Decode sim tokens back to predicted point states in raw space.

        Args:
            sim_x: (1, T*N, d_sim) — simulation output tokens.
            T: Number of latent time frames.
            N: Number of points (ΣN_i).
            T_raw: Raw temporal length (4*(T-1)+1).

        Returns:
            (1, T, N, d_state) — predicted value in latent space (before causal decoder).
        """
        x = sim_x.view(1, T, N, self.sim.d_sim)
        x = self.sim.head_proj(self.sim.head_norm(x))
        return x

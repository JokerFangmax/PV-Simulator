"""Stage 1 inference pipeline for PV-Simulator: Simulation Branch only.

Runs the SimTransformer standalone (no video branch) to denoise physics
trajectories from pure noise given physics conditions.

Usage:
    from videox_fun.pipeline.pipeline_simulation import SimulationPipeline

    pipeline = SimulationPipeline.from_pretrained("outputs/stage1/final")
    result = pipeline(
        c_floor=torch.tensor([0.0]),          # (1,)
        c_id=torch.tensor([[0, 1]]),          # (1, n_objects)
        c_mat=torch.tensor([[[0.5, 0.3], [0.4, 0.8]]]),  # (1, n_objects, 2)
        c_mass=torch.tensor([[1.0, 2.0]]),    # (1, n_objects)
        c_static=torch.tensor([[0, 0]]),      # (1, n_objects)
        c_force_raw=torch.zeros(1, T_raw, N, 6),          # (1, T_raw, N, 6)
        c_init=torch.zeros(1, n_objects, 7),  # (1, n_objects, 7) pos+vel+mask
        point_obj_idx=point_obj_idx,          # (1, N)
        T=6,
        num_inference_steps=50,
    )
    x_s_decoded = result['x_s']   # (1, T_raw, N, 6)  predicted pos + vel
"""

import os
from typing import Optional

import torch
import torch.nn as nn
from diffusers import FlowMatchEulerDiscreteScheduler
from tqdm.auto import tqdm

from videox_fun.models.sim_causal_encoder import CausalTemporalEncoder, CausalTemporalDecoder
from videox_fun.models.sim_condition import SimConditionEmbedder
from videox_fun.models.sim_transformer import SimTransformer


class SimulationPipeline:
    """Inference pipeline for the Stage 1 Simulation Branch.

    Loads encoder/decoder/conditions/transformer from a Stage 1 checkpoint
    directory and runs DDIM-style flow-matching denoising.

    Args:
        sim_transformer: SimTransformer model.
        x_s_encoder: CausalTemporalEncoder.
        x_s_decoder: CausalTemporalDecoder.
        sim_cond_embedder: SimConditionEmbedder.
        scheduler: FlowMatchEulerDiscreteScheduler.
    """

    def __init__(
        self,
        sim_transformer: SimTransformer,
        x_s_encoder: CausalTemporalEncoder,
        x_s_decoder: CausalTemporalDecoder,
        sim_cond_embedder: SimConditionEmbedder,
        scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None,
    ):
        self.sim_transformer = sim_transformer
        self.x_s_encoder = x_s_encoder
        self.x_s_decoder = x_s_decoder
        self.sim_cond_embedder = sim_cond_embedder
        self.scheduler = scheduler or FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
        self.device = next(sim_transformer.parameters()).device
        self.dtype = next(sim_transformer.parameters()).dtype

    @classmethod
    def from_pretrained(cls, ckpt_dir: str, device: str = "cuda", dtype=torch.bfloat16,
                        max_objects: int = 16, d_state: int = 256, d_sim: int = 512,
                        sim_ffn_dim: int = 2048, sim_num_heads: int = 8,
                        sim_num_layers: int = 10, state_enc_mid: int = 128):
        """Load a SimulationPipeline from a Stage 1 checkpoint directory."""
        sim_cond_embedder = SimConditionEmbedder(max_objects=max_objects)
        d_cond = sim_cond_embedder.d_cond

        x_s_encoder = CausalTemporalEncoder(c_in=6, c_mid=state_enc_mid, d_out=d_state)
        x_s_decoder = CausalTemporalDecoder(d_feat=d_state, c_mid=state_enc_mid, c_out=6)
        sim_transformer = SimTransformer(
            d_state=d_state, d_cond=d_cond, d_sim=d_sim,
            ffn_dim=sim_ffn_dim, num_heads=sim_num_heads, num_layers=sim_num_layers,
        )

        sim_transformer.load_state_dict(
            torch.load(os.path.join(ckpt_dir, "sim_transformer.pt"), map_location="cpu"))
        x_s_encoder.load_state_dict(
            torch.load(os.path.join(ckpt_dir, "x_s_encoder.pt"), map_location="cpu"))
        x_s_decoder.load_state_dict(
            torch.load(os.path.join(ckpt_dir, "x_s_decoder.pt"), map_location="cpu"))
        sim_cond_embedder.load_state_dict(
            torch.load(os.path.join(ckpt_dir, "sim_cond_embedder.pt"), map_location="cpu"))

        sim_transformer = sim_transformer.to(device, dtype=dtype)
        x_s_encoder = x_s_encoder.to(device, dtype=dtype)
        x_s_decoder = x_s_decoder.to(device, dtype=dtype)
        sim_cond_embedder = sim_cond_embedder.to(device, dtype=dtype)

        return cls(sim_transformer, x_s_encoder, x_s_decoder, sim_cond_embedder)

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        for m in [self.sim_transformer, self.x_s_encoder, self.x_s_decoder, self.sim_cond_embedder]:
            m.to(device=device, dtype=dtype)
        return self

    @torch.no_grad()
    def __call__(
        self,
        c_floor: torch.Tensor,          # (B,) float
        c_id: torch.Tensor,              # (B, n_objects) int
        c_mat: torch.Tensor,             # (B, n_objects, 2) float
        c_mass: torch.Tensor,            # (B, n_objects) float
        c_static: torch.Tensor,          # (B, n_objects) int
        c_force_raw: torch.Tensor,       # (B, T_raw, N, 6)
        c_init: torch.Tensor,            # (B, n_objects, 7) float  pos+vel+mask
        point_obj_idx: torch.Tensor,     # (B, N) int
        T: int,                          # number of latent frames
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,     # reserved for future CFG
        generator: Optional[torch.Generator] = None,
        show_progress: bool = True,
        point_mask: Optional[torch.Tensor] = None,    # (B, N) bool — valid points (padded batch)
        valid_seq_mask: Optional[torch.Tensor] = None,  # (B, T*N) bool — valid tokens (padded batch)
    ):
        """Run simulation denoising from noise.

        Args:
            point_mask: Optional (B, N) bool tensor marking valid (non-padded) points.
                Pass when using padded batch mode to zero out padding in conditions.
            valid_seq_mask: Optional (B, T*N) bool tensor marking valid tokens.
                Pass when using padded batch mode to mask padding in self-attention.

        Returns:
            dict with:
              'x_s': (B, T_raw, N, 6) — decoded predicted point states (pos + vel)
              'x_s_latent': (B, T, N, d_state) — latent-space output before decoding
        """
        device = self.device
        dtype = self.dtype
        T_raw = 4 * (T - 1) + 1
        B = c_floor.shape[0]
        N = point_obj_idx.shape[1]

        # Move inputs to device
        c_floor = c_floor.to(device, dtype=dtype)
        c_id = c_id.to(device)
        c_mat = c_mat.to(device, dtype=dtype)
        c_mass = c_mass.to(device, dtype=dtype)
        c_static = c_static.to(device)
        c_force_raw = c_force_raw.to(device, dtype=dtype)
        c_init = c_init.to(device, dtype=dtype)
        point_obj_idx = point_obj_idx.to(device)
        if point_mask is not None:
            point_mask = point_mask.to(device)
        if valid_seq_mask is not None:
            valid_seq_mask = valid_seq_mask.to(device)

        # --- Encode conditions (constant across timesteps) ---
        c_sim = self.sim_cond_embedder(
            c_floor=c_floor,
            c_id=c_id,
            c_mat=c_mat,
            c_mass=c_mass,
            c_static=c_static,
            c_force_raw=c_force_raw,
            c_init=c_init,
            point_obj_idx=point_obj_idx,
            T=T,
            point_mask=point_mask,
        )  # (B, T, N, d_cond)

        # --- Start from pure noise in latent space ---
        latents = torch.randn(
            B, T, N, self.sim_transformer.d_state,
            device=device, dtype=dtype, generator=generator,
        )

        # --- Denoising loop ---
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        progress_bar = tqdm(timesteps, desc="Denoising", disable=not show_progress)
        for t in progress_bar:
            t_batch = t.unsqueeze(0)

            pred = self.sim_transformer(
                latents, c_sim, t_batch, dtype=dtype,
                valid_seq_mask=valid_seq_mask,
            )  # (B, T, N, d_state)

            # Scheduler step
            latents = self.scheduler.step(pred, t, latents).prev_sample

        # --- Decode latents back to raw state space ---
        x_s = self.x_s_decoder(latents, T_raw)  # (B, T_raw, N, 6)

        return {
            'x_s': x_s,
            'x_s_latent': latents,
        }

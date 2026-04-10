"""Stage 1 inference pipeline for PV-Simulator: Simulation Branch only.

Runs the full denoising network (frozen AE Encoder → SimTransformer → frozen
AE Decoder) to denoise physics trajectories from pure noise in raw state
space.  The AE encoder/decoder are pre-trained in Stage 0 and frozen; the
diffusion process operates in raw (T_raw, N, 6) space while the DiT works
in compressed latent space.

Initial frame conditioning: the first frame of the trajectory is encoded
with the same frozen AE, zero-padded to T latent frames, and concatenated
with an inpainting mask (0=given, 1=unknown) as additional DiT inputs.

Usage:
    from videox_fun.pipeline.pipeline_simulation import SimulationPipeline

    pipeline = SimulationPipeline.from_pretrained(
        "outputs/stage1/final", ae_ckpt_dir="outputs/ae/final")
    result = pipeline(
        c_floor=torch.tensor([0.0]),
        c_id=torch.tensor([[0, 1]]),
        c_mat=torch.tensor([[[0.5, 0.3], [0.4, 0.8]]]),
        c_mass=torch.tensor([[1.0, 2.0]]),
        c_static=torch.tensor([[0, 0]]),
        c_force_raw=torch.zeros(1, T_raw, N, 6),
        x_s_init=x_s_raw[:, :1],             # (1, 1, N, 6) first frame
        point_obj_idx=point_obj_idx,
        T=6,
        num_inference_steps=50,
    )
    x_s_pred = result['x_s']   # (1, T_raw, N, 6) predicted pos + vel
"""

import os
from typing import Optional

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from tqdm.auto import tqdm

from videox_fun.models.sim_ae import CausalAE
from videox_fun.models.sim_condition import SimConditionEmbedder
from videox_fun.models.sim_transformer import SimTransformer


class SimulationPipeline:
    """Inference pipeline for the Stage 1 Simulation Branch.

    The denoising network is frozen AE Encoder → DiT → frozen AE Decoder:
      - AE Encoder projects noisy raw states (T_raw, N, 3) → latent (T, N, 16)
        applied separately to pos and vel → (T, N, 32) concatenated
      - DiT denoises in latent space with init_enc + init_mask conditioning
      - AE Decoder projects latent prediction → raw (T_raw, N, 3) per channel group

    The diffusion process (noise addition / scheduler steps) operates in the
    original raw state space.

    Args:
        sim_transformer: SimTransformer model.
        ae: CausalAE (frozen, pre-trained in Stage 0).
        sim_cond_embedder: SimConditionEmbedder.
        scheduler: FlowMatchEulerDiscreteScheduler.
    """

    def __init__(
        self,
        sim_transformer: SimTransformer,
        ae: CausalAE,
        sim_cond_embedder: SimConditionEmbedder,
        scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None,
    ):
        self.sim_transformer = sim_transformer
        self.ae = ae
        self.sim_cond_embedder = sim_cond_embedder
        self.scheduler = scheduler or FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
        self.device = next(sim_transformer.parameters()).device
        self.dtype = next(sim_transformer.parameters()).dtype

    @classmethod
    def from_pretrained(cls, ckpt_dir: str, ae_ckpt_dir: str,
                        device: str = "cuda", dtype=torch.bfloat16,
                        max_objects: int = 16, d_state: int = 64, d_sim: int = 512,
                        sim_ffn_dim: int = 2048, sim_num_heads: int = 8,
                        sim_num_layers: int = 10):
        """Load a SimulationPipeline from a Stage 1 checkpoint directory."""
        # Load frozen AE
        ae = CausalAE.load(ae_ckpt_dir)
        ae.to(device, dtype=dtype)
        ae.eval()
        for p in ae.parameters():
            p.requires_grad_(False)

        sim_cond_embedder = SimConditionEmbedder(max_objects=max_objects)
        d_cond = sim_cond_embedder.d_cond

        sim_transformer = SimTransformer(
            d_state=d_state, d_cond=d_cond, d_sim=d_sim,
            ffn_dim=sim_ffn_dim, num_heads=sim_num_heads, num_layers=sim_num_layers,
        )

        sim_transformer.load_state_dict(
            torch.load(os.path.join(ckpt_dir, "sim_transformer.pt"), map_location="cpu"))
        sim_cond_embedder.load_state_dict(
            torch.load(os.path.join(ckpt_dir, "sim_cond_embedder.pt"), map_location="cpu"))

        sim_transformer = sim_transformer.to(device, dtype=dtype)
        sim_cond_embedder = sim_cond_embedder.to(device, dtype=dtype)

        return cls(sim_transformer, ae, sim_cond_embedder)

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        for m in [self.sim_transformer, self.ae, self.sim_cond_embedder]:
            m.to(device=device, dtype=dtype)
        return self

    def _encode_state(self, x_raw):
        """Encode raw states (B, T_raw, N, 6) → latent (B, T, N, d_state)."""
        pos_enc = self.ae.encode(x_raw[..., :3])    # (B, T, N, d_latent)
        vel_enc = self.ae.encode(x_raw[..., 3:6])   # (B, T, N, d_latent)
        return torch.cat([pos_enc, vel_enc], dim=-1)  # (B, T, N, d_state)

    def _decode_state(self, z, t_raw):
        """Decode latent (B, T, N, d_state) → raw states (B, T_raw, N, 6)."""
        d = self.ae.d_latent
        pos_raw = self.ae.decode(z[..., :d], t_raw)   # (B, T_raw, N, 3)
        vel_raw = self.ae.decode(z[..., d:], t_raw)    # (B, T_raw, N, 3)
        return torch.cat([pos_raw, vel_raw], dim=-1)   # (B, T_raw, N, 6)

    @torch.no_grad()
    def __call__(
        self,
        c_floor: torch.Tensor,          # (B,) float
        c_id: torch.Tensor,             # (B, n_objects) int
        c_mat: torch.Tensor,            # (B, n_objects, 2) float
        c_mass: torch.Tensor,           # (B, n_objects) float
        c_static: torch.Tensor,         # (B, n_objects) int
        c_force_raw: torch.Tensor,      # (B, T_raw, N, 6)
        x_s_init: torch.Tensor,         # (B, 1, N, 6) float — initial frame (per-point)
        point_obj_idx: torch.Tensor,    # (B, N) int
        T: int,                         # number of latent frames
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,    # reserved for future CFG
        generator: Optional[torch.Generator] = None,
        show_progress: bool = True,
        point_mask: Optional[torch.Tensor] = None,    # (B, N) bool
        valid_seq_mask: Optional[torch.Tensor] = None,  # (B, T*N) bool
    ):
        """Run simulation denoising from noise in raw state space.

        At each denoising step the full network runs:
          noisy_raw → AE Encoder → latent → DiT (with init conditioning)
          → latent → AE Decoder → velocity_raw
        The scheduler then updates the raw-space sample.

        Returns:
            dict with 'x_s': (B, T_raw, N, 6) — predicted point states
        """
        device = self.device
        dtype = self.dtype
        T_raw = 4 * (T - 1) + 1
        B = c_floor.shape[0]
        N = point_obj_idx.shape[1]
        d_state = self.sim_transformer.d_state

        # Move inputs to device
        c_floor = c_floor.to(device, dtype=dtype)
        c_id = c_id.to(device)
        c_mat = c_mat.to(device, dtype=dtype)
        c_mass = c_mass.to(device, dtype=dtype)
        c_static = c_static.to(device)
        c_force_raw = c_force_raw.to(device, dtype=dtype)
        x_s_init = x_s_init.to(device, dtype=dtype)
        point_obj_idx = point_obj_idx.to(device)
        if point_mask is not None:
            point_mask = point_mask.to(device)
        if valid_seq_mask is not None:
            valid_seq_mask = valid_seq_mask.to(device)

        # --- Encode initial frame conditioning (constant across denoising steps) ---
        init_enc_1 = self._encode_state(x_s_init)         # (B, 1, N, d_state)
        init_enc_padded = torch.cat([
            init_enc_1,
            torch.zeros(B, T - 1, N, d_state, device=device, dtype=dtype),
        ], dim=1)                                           # (B, T, N, d_state)
        init_mask = torch.ones(B, T, N, 1, device=device, dtype=dtype)
        init_mask[:, 0, :, :] = 0.0  # first latent frame is given

        # --- Encode conditions (constant across timesteps) ---
        c_sim = self.sim_cond_embedder(
            c_floor=c_floor,
            c_id=c_id,
            c_mat=c_mat,
            c_mass=c_mass,
            c_static=c_static,
            c_force_raw=c_force_raw,
            point_obj_idx=point_obj_idx,
            T=T,
            point_mask=point_mask,
        )  # (B, T, N, d_cond)

        # --- Start from pure noise in raw state space ---
        sample = torch.randn(
            B, T_raw, N, 6,
            device=device, dtype=dtype, generator=generator,
        )

        # --- Denoising loop (diffusion in raw space) ---
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        progress_bar = tqdm(timesteps, desc="Denoising", disable=not show_progress)
        for t in progress_bar:
            t_batch = t.unsqueeze(0)

            # Encode current noisy raw sample to latent via frozen AE
            encoded = self._encode_state(sample)  # (B, T, N, d_state)

            # DiT forward in latent space with init conditioning
            pred_latent = self.sim_transformer(
                encoded, init_enc_padded, init_mask, c_sim, t_batch,
                dtype=dtype, valid_seq_mask=valid_seq_mask,
            )  # (B, T, N, d_state)

            # Decode back to raw space (velocity prediction)
            pred_raw = self._decode_state(pred_latent, T_raw)  # (B, T_raw, N, 6)

            # Euler step in raw space
            sample = self.scheduler.step(pred_raw, t, sample).prev_sample

        return {
            'x_s': sample,
        }

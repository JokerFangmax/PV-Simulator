"""Stage 1 inference pipeline for PV-Simulator: Simulation Branch only.

LDM-style denoising: the frozen AE encoder encodes the initial frame once
before the diffusion loop, the SimTransformer iteratively denoises in
latent space (B, T, N, d_state), and the AE decoder is applied a single
time at the end to recover raw states (B, T_raw, N, 6). The AE is pre-
trained in Stage 0 and is completely detached from the diffusion process.

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

    LDM-style latent-space denoising:
      - The AE encoder encodes x_s_init (B, 1, N, 6) once → init_enc (B, 1, N, d_state).
      - The DiT iteratively denoises a latent (B, T, N, d_state) starting from pure noise.
      - After all denoising steps, the AE decoder is applied once to recover raw
        states (B, T_raw, N, 6).
    The diffusion process (noise addition / scheduler steps) operates entirely in
    latent space; the AE is detached from the DiT.

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
        anchor_mode: str = "local",
    ):
        self.sim_transformer = sim_transformer
        self.ae = ae
        self.sim_cond_embedder = sim_cond_embedder
        self.scheduler = scheduler or FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
        self.device = next(sim_transformer.parameters()).device
        self.dtype = next(sim_transformer.parameters()).dtype
        self.anchor_mode = anchor_mode

    @classmethod
    def from_pretrained(cls, ckpt_dir: str, ae_ckpt_dir: str,
                        device: str = "cuda", dtype=torch.bfloat16,
                        max_objects: int = 16, d_state: int = 32, d_sim: int = 256,
                        sim_ffn_dim: int = 1024, sim_num_heads: int = 8,
                        sim_num_layers: int = 10, anchor_mode: str = "local"):
        """Load a SimulationPipeline from a Stage 1 checkpoint directory."""
        # Load frozen AE
        ae = CausalAE.load(ae_ckpt_dir)
        ae.to(device, dtype=dtype)
        ae.eval()
        for p in ae.parameters():
            p.requires_grad_(False)

        sim_cond_state = torch.load(
            os.path.join(ckpt_dir, "sim_cond_embedder.pt"), map_location="cpu")
        ckpt_max_objects = int(sim_cond_state["id_embed.weight"].shape[0])
        sim_cond_embedder = SimConditionEmbedder(
            max_objects=ckpt_max_objects,
            d_force=2 * ae.d_latent,
        )
        d_cond = sim_cond_embedder.d_cond

        sim_transformer_state = torch.load(
            os.path.join(ckpt_dir, "sim_transformer.pt"), map_location="cpu")

        ckpt_d_state = int(sim_transformer_state["head_proj.bias"].shape[0])
        ckpt_d_sim = int(sim_transformer_state["head_proj.weight"].shape[1])
        ckpt_ffn_dim = int(sim_transformer_state["blocks.0.ffn.0.weight"].shape[0])
        ckpt_num_layers = 1 + max(
            int(key.split(".")[1])
            for key in sim_transformer_state.keys()
            if key.startswith("blocks.") and key.split(".")[1].isdigit()
        )
        ckpt_d_anchor = int(
            sim_transformer_state["input_proj.weight"].shape[1]
            - (2 * ckpt_d_state + 1 + d_cond)
        )
        ckpt_use_temporal_correspondence = any(
            key.startswith("temporal_corr_block.") for key in sim_transformer_state.keys()
        )
        ckpt_use_temporal_rope = bool(
            sim_transformer_state.get("temporal_corr_block.temporal_attn.rope_indicator", torch.tensor(0.0)).item()
        )
        if ckpt_d_anchor < 0:
            raise ValueError(
                f"Invalid inferred d_anchor={ckpt_d_anchor} from checkpoint {ckpt_dir}. "
                f"input_proj.in_features={sim_transformer_state['input_proj.weight'].shape[1]}, "
                f"d_state={ckpt_d_state}, d_cond={d_cond}."
            )

        sim_transformer = SimTransformer(
            d_state=ckpt_d_state,
            d_cond=d_cond,
            d_anchor=ckpt_d_anchor,
            d_sim=ckpt_d_sim,
            ffn_dim=ckpt_ffn_dim,
            num_heads=sim_num_heads,
            num_layers=ckpt_num_layers,
            use_temporal_correspondence=ckpt_use_temporal_correspondence,
            use_temporal_rope=ckpt_use_temporal_rope,
        )

        sim_transformer.load_state_dict(sim_transformer_state)
        sim_cond_embedder.load_state_dict(sim_cond_state)

        sim_transformer = sim_transformer.to(device, dtype=dtype)
        sim_cond_embedder = sim_cond_embedder.to(device, dtype=dtype)

        return cls(sim_transformer, ae, sim_cond_embedder, anchor_mode=anchor_mode)

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

    def _compute_point_anchor(self, x_s_init: torch.Tensor, point_obj_idx: torch.Tensor) -> torch.Tensor:
        """Build per-point anchors from the initial frame.

        Modes:
          - local: centered world coordinates
        Shape: (B, 1, N, 3)
        """
        init_pos = x_s_init[..., :3]  # (B, 1, N, 3)
        B, _, N, _ = init_pos.shape
        anchor = torch.zeros_like(init_pos)
        for b in range(B):
            obj_ids = torch.unique(point_obj_idx[b])
            for obj_id in obj_ids.tolist():
                obj_mask = point_obj_idx[b] == obj_id
                if not torch.any(obj_mask):
                    continue
                obj_pos = init_pos[b, 0, obj_mask]
                centroid = obj_pos.mean(dim=0, keepdim=True)
                centered = obj_pos - centroid
                # NOTE:
                # The previous canonical_pca branch is intentionally commented out.
                # For near-symmetric objects or noisy point samples, the PCA basis
                # is not stable enough to serve as a cross-sample point identity.
                #
                # if self.anchor_mode == "canonical_pca" and centered.shape[0] >= 3:
                #     cov = centered.T @ centered
                #     eigvals, eigvecs = torch.linalg.eigh(cov)
                #     order = torch.argsort(eigvals, descending=True)
                #     basis = eigvecs[:, order]
                #     proj = centered @ basis
                #     for ax_i in range(3):
                #         idx = torch.argmax(proj[:, ax_i].abs())
                #         sign = torch.sign(proj[idx, ax_i])
                #         if sign == 0:
                #             sign = centered.new_tensor(1.0)
                #         basis[:, ax_i] = basis[:, ax_i] * sign
                #         proj[:, ax_i] = proj[:, ax_i] * sign
                #     if torch.det(basis) < 0:
                #         basis[:, -1] = -basis[:, -1]
                #     centered = centered @ basis
                anchor[b, 0, obj_mask] = centered
        return anchor

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
        """Run LDM-style simulation denoising in latent space.

        Flow:
          - Encode x_s_init once → init_enc_padded (condition).
          - Sample pure noise in latent space (B, T, N, d_state).
          - For each scheduler timestep, DiT predicts latent velocity and the
            scheduler updates the latent sample.
          - After the loop, AE decoder projects the final latent → raw states.

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
        # Repeat frame-0 latent on every timestep token so each future token keeps
        # direct access to its own initial point identity signal.
        init_enc_1 = self._encode_state(x_s_init)         # (B, 1, N, d_state)
        init_enc_padded = init_enc_1.expand(-1, T, -1, -1).contiguous()  # (B, T, N, d_state)
        init_mask = torch.ones(B, T, N, 1, device=device, dtype=dtype)
        init_mask[:, 0, :, :] = 0.0  # first latent frame is given
        point_anchor_1 = self._compute_point_anchor(x_s_init, point_obj_idx).to(dtype)
        if self.sim_transformer.d_anchor == 0:
            point_anchor_1 = point_anchor_1[..., :0]
        elif point_anchor_1.shape[-1] != self.sim_transformer.d_anchor:
            point_anchor_1 = point_anchor_1[..., :self.sim_transformer.d_anchor]
        point_anchor = point_anchor_1.expand(-1, T, -1, -1).contiguous()

        # --- Encode force+contact via frozen AE (constant across timesteps) ---
        c_force_enc = torch.cat([
            self.ae.encode(c_force_raw[..., :3]),
            self.ae.encode(c_force_raw[..., 3:6]),
        ], dim=-1)  # (B, T, N, 2*d_latent)

        # --- Encode conditions (constant across timesteps) ---
        c_sim = self.sim_cond_embedder(
            c_floor=c_floor,
            c_id=c_id,
            c_mat=c_mat,
            c_mass=c_mass,
            c_static=c_static,
            c_force_enc=c_force_enc,
            point_obj_idx=point_obj_idx,
            T=T,
            point_mask=point_mask,
        )  # (B, T, N, d_cond)

        # --- Start from pure noise in latent space (LDM-style) ---
        sample = torch.randn(
            B, T, N, d_state,
            device=device, dtype=dtype, generator=generator,
        )
        sample[:, :1] = init_enc_1

        # --- Denoising loop (diffusion in latent space) ---
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        progress_bar = tqdm(timesteps, desc="Denoising", disable=not show_progress)
        for t in progress_bar:
            t_batch = t.unsqueeze(0)

            # DiT forward in latent space with init conditioning
            pred_latent = self.sim_transformer(
                sample, init_enc_padded, init_mask, point_anchor, c_sim, t_batch,
                dtype=dtype, valid_seq_mask=valid_seq_mask,
            )  # (B, T, N, d_state) — predicted latent velocity

            # Euler step in latent space
            sample = self.scheduler.step(pred_latent, t, sample).prev_sample
            sample[:, :1] = init_enc_1

        # --- Decode once at the end ---
        x_s_pred = self._decode_state(sample, T_raw)  # (B, T_raw, N, 6)

        return {
            'x_s': x_s_pred,
        }

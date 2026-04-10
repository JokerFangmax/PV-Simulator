# Simulation condition embedders for PV-Simulator.
# Encodes physics conditions (floor, object ID, material, mass, static, force)
# into per-point, per-timestep feature tensors.
#
# Initial state conditioning is handled separately via the frozen CausalAE
# encoder — see sim_ae.py and the init_enc/init_mask inputs to SimTransformer.
#
# Per-object properties are expanded to per-point via point_obj_idx (N,) mapping.

from typing import Optional

import torch
import torch.nn as nn

from videox_fun.models.sim_causal_encoder import CausalTemporalEncoder


class ConditionMLP(nn.Module):
    """Simple MLP encoder: input_dim → hidden → output_dim."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = out_dim * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimConditionEmbedder(nn.Module):
    """Encodes all simulation conditions into a unified tensor c_sim.

    Handles per-object → per-point expansion via point_obj_idx.

    Output shape: (B, T, N, d_cond) where d_cond = sum of all sub-condition dims.

    Note: Initial state conditioning is no longer handled here. It is encoded
    via the frozen CausalAE (same encoder as the noisy state) and passed
    directly to SimTransformer as init_enc + init_mask.

    Args:
        d_floor: Floor height embedding dim.
        d_id: Object ID embedding dim.
        d_mat: Material (friction, restitution) embedding dim.
        d_mass: Mass embedding dim.
        d_static: Static flag embedding dim.
        d_force: Force embedding dim (via causal temporal encoder).
        max_objects: Maximum number of objects per scene.
        force_encoder_mid: Intermediate channel width for force causal encoder.
    """

    def __init__(
        self,
        d_floor: int = 64,
        d_id: int = 64,
        d_mat: int = 64,
        d_mass: int = 32,
        d_static: int = 16,
        d_force: int = 128,
        max_objects: int = 16,
        force_encoder_mid: int = 128,
    ):
        super().__init__()
        self.d_cond = d_floor + d_id + d_mat + d_mass + d_static + d_force

        # Scalar / per-object encoders
        self.floor_mlp = ConditionMLP(1, d_floor)
        self.id_embed = nn.Embedding(max_objects, d_id)
        self.mat_mlp = ConditionMLP(2, d_mat)
        self.mass_mlp = ConditionMLP(1, d_mass)
        self.static_embed = nn.Embedding(2, d_static)

        # Time-varying force: (B, T_raw, N, 6) → (B, T, N, d_force)
        self.force_encoder = CausalTemporalEncoder(6, force_encoder_mid, d_force)

    def forward(
        self,
        c_floor: torch.Tensor,        # (B,) float — floor height
        c_id: torch.Tensor,            # (B, n_objects) int — object IDs [0..max_objects)
        c_mat: torch.Tensor,           # (B, n_objects, 2) float — (friction, restitution)
        c_mass: torch.Tensor,          # (B, n_objects,) float — mass
        c_static: torch.Tensor,        # (B, n_objects,) int — static flag {0, 1}
        c_force_raw: torch.Tensor,     # (B, T_raw, N, 6) float — force + contact point
        point_obj_idx: torch.Tensor,   # (B, N) int — maps each point to its object
        T: int,                        # Number of latent time frames
        point_mask: Optional[torch.Tensor] = None,  # (B, N) bool — valid points (padded batch mode)
    ) -> torch.Tensor:
        """Encode all conditions and return (B, T, N, d_cond)."""
        B = c_floor.shape[0]
        N = point_obj_idx.shape[1]

        # --- Per-object embeddings → gather to per-point → expand over T ---
        # Cast to the model's weight dtype so we work correctly under any mixed precision.
        w_dtype = next(self.parameters()).dtype

        # Floor: (B,) → (B, d_floor)
        e_floor = self.floor_mlp(c_floor.view(B, 1).to(w_dtype))    # (B, d_floor)
        e_floor = e_floor.view(B, 1, 1, -1).expand(B, T, N, -1)

        # Object ID: (B, n_objects) → embed → (B, n_objects, d_id)
        e_id = self.id_embed(c_id)                                   # (B, n_obj, d_id)
        idx = point_obj_idx.unsqueeze(-1).expand(-1, -1, e_id.size(-1))  # (B, N, d_id)
        e_id = e_id.gather(1, idx)                                   # (B, N, d_id)
        e_id = e_id.unsqueeze(1).expand(-1, T, -1, -1)

        # Material: (B, n_objects, 2) → MLP → gather → (B, T, N, d_mat)
        e_mat = self.mat_mlp(c_mat.to(w_dtype))                     # (B, n_obj, d_mat)
        idx = point_obj_idx.unsqueeze(-1).expand(-1, -1, e_mat.size(-1))
        e_mat = e_mat.gather(1, idx)                                 # (B, N, d_mat)
        e_mat = e_mat.unsqueeze(1).expand(-1, T, -1, -1)

        # Mass: (B, n_objects) → MLP → gather → (B, T, N, d_mass)
        e_mass = self.mass_mlp(c_mass.to(w_dtype).unsqueeze(-1))    # (B, n_obj, d_mass)
        idx = point_obj_idx.unsqueeze(-1).expand(-1, -1, e_mass.size(-1))
        e_mass = e_mass.gather(1, idx)                               # (B, N, d_mass)
        e_mass = e_mass.unsqueeze(1).expand(-1, T, -1, -1)

        # Static flag: (B, n_objects) → embed → gather → (B, T, N, d_static)
        e_static = self.static_embed(c_static)                       # (B, n_obj, d_static)
        idx = point_obj_idx.unsqueeze(-1).expand(-1, -1, e_static.size(-1))
        e_static = e_static.gather(1, idx)                           # (B, N, d_static)
        e_static = e_static.unsqueeze(1).expand(-1, T, -1, -1)

        # --- Time-varying conditions ---

        # Force: (B, T_raw, N, 6) → causal encoder → (B, T, N, d_force)
        e_force = self.force_encoder(c_force_raw)

        # --- Concatenate all ---
        c_sim = torch.cat(
            [e_floor, e_id, e_mat, e_mass, e_static, e_force],
            dim=-1,
        )  # (B, T, N, d_cond)

        # Zero out embeddings at padded point positions (padded batch mode)
        if point_mask is not None:
            # point_mask: (B, N) → (B, 1, N, 1); cast to c_sim dtype to avoid fp32 promotion
            c_sim = c_sim * point_mask.to(dtype=c_sim.dtype).unsqueeze(1).unsqueeze(-1)

        return c_sim

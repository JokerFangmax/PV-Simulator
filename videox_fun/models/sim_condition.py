# Simulation condition embedders for PV-Simulator.
# Encodes physics conditions (floor, object ID, material, mass, static, force)
# into per-point, per-timestep feature tensors.
#
# Force is pre-encoded by the frozen CausalAE outside this module (same path
# as the state encoding) — the embedder receives c_force_enc already at latent
# temporal resolution (T, not T_raw).
#
# Initial state conditioning is handled separately via the frozen CausalAE
# encoder — see sim_ae.py and the init_enc/init_mask inputs to SimTransformer.
#
# Per-object properties are expanded to per-point via point_obj_idx (N,) mapping.

from typing import Optional

import torch
import torch.nn as nn


class ConditionMLP(nn.Module):
    """Simple MLP encoder: input_dim → hidden → output_dim."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(out_dim * 2, 8)
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

    Force is expected to be pre-encoded by the frozen CausalAE (force-3 and
    contact-3 channels separately → 2 * d_latent dims), yielding
    c_force_enc of shape (B, T, N, d_force).

    Args:
        d_floor:  Floor height embedding dim.
        d_id:     Object ID embedding dim.
        d_mat:    Material (friction, restitution) embedding dim.
        d_mass:   Mass embedding dim.
        d_static: Static flag embedding dim.
        d_force:  Force embedding dim (matches 2 * AE d_latent = 32).
        max_objects: Maximum number of objects per scene.
    """

    def __init__(
        self,
        d_floor: int = 4,
        d_id: int = 8,
        d_mat: int = 8,
        d_mass: int = 4,
        d_static: int = 4,
        d_force: int = 32,
        max_objects: int = 16,
    ):
        super().__init__()
        self.d_floor = d_floor
        self.d_id = d_id
        self.d_mat = d_mat
        self.d_mass = d_mass
        self.d_static = d_static
        self.d_force = d_force
        self.d_cond = d_floor + d_id + d_mat + d_mass + d_static + d_force

        # Scalar / per-object encoders
        self.floor_mlp = ConditionMLP(1, d_floor)
        self.id_embed = nn.Embedding(max_objects, d_id)
        self.mat_mlp = ConditionMLP(2, d_mat)
        self.mass_mlp = ConditionMLP(1, d_mass)
        self.static_embed = nn.Embedding(2, d_static)

    def forward(
        self,
        c_floor: torch.Tensor,         # (B,) float — floor height
        c_id: torch.Tensor,            # (B, n_objects) int — object IDs [0..max_objects)
        c_mat: torch.Tensor,           # (B, n_objects, 2) float — (friction, restitution)
        c_mass: torch.Tensor,          # (B, n_objects,) float — mass
        c_static: torch.Tensor,        # (B, n_objects,) int — static flag {0, 1}
        c_force_enc: torch.Tensor,     # (B, T, N, d_force) — AE-encoded force+contact
        point_obj_idx: torch.Tensor,   # (B, N) int — maps each point to its object
        T: int,                        # Number of latent time frames
        point_mask: Optional[torch.Tensor] = None,  # (B, N) bool — valid points (padded batch mode)
    ) -> torch.Tensor:
        """Encode all conditions and return (B, T, N, d_cond)."""
        B = c_floor.shape[0]
        N = point_obj_idx.shape[1]

        # Cast to the model's weight dtype so we work correctly under any mixed precision.
        w_dtype = next(self.parameters()).dtype

        # Floor: (B,) → (B, d_floor) → broadcast over (T, N)
        e_floor = self.floor_mlp(c_floor.view(B, 1).to(w_dtype))
        e_floor = e_floor.view(B, 1, 1, -1).expand(B, T, N, -1)

        # Object ID: (B, n_objects) → embed → gather per-point → expand over T
        e_id = self.id_embed(c_id)                                      # (B, n_obj, d_id)
        idx = point_obj_idx.unsqueeze(-1).expand(-1, -1, e_id.size(-1)) # (B, N, d_id)
        e_id = e_id.gather(1, idx)                                      # (B, N, d_id)
        e_id = e_id.unsqueeze(1).expand(-1, T, -1, -1)                  # (B, T, N, d_id)

        # Material: (B, n_objects, 2) → MLP → gather → expand
        e_mat = self.mat_mlp(c_mat.to(w_dtype))                         # (B, n_obj, d_mat)
        idx = point_obj_idx.unsqueeze(-1).expand(-1, -1, e_mat.size(-1))# (B, N, d_mat)
        e_mat = e_mat.gather(1, idx)                      # (B, N, d_mat)
        e_mat = e_mat.unsqueeze(1).expand(-1, T, -1, -1)    # (B, T, N, d_mat)

        # Mass: (B, n_objects) → MLP → gather → expand
        e_mass = self.mass_mlp(c_mass.to(w_dtype).unsqueeze(-1))    # (B, n_obj, d_mass)
        idx = point_obj_idx.unsqueeze(-1).expand(-1, -1, e_mass.size(-1)) # (B, N, d_mass)
        e_mass = e_mass.gather(1, idx)                # (B, N, d_mass)
        e_mass = e_mass.unsqueeze(1).expand(-1, T, -1, -1)  # (B, T, N, d_mass)

        # Static flag: (B, n_objects) → embed → gather → expand
        e_static = self.static_embed(c_static)                      # (B, n_obj, d_static)
        idx = point_obj_idx.unsqueeze(-1).expand(-1, -1, e_static.size(-1)) # (B, N, d_static)
        e_static = e_static.gather(1, idx)              # (B, N, d_static)
        e_static = e_static.unsqueeze(1).expand(-1, T, -1, -1)          # (B, T, N, d_static)

        # Force + contact: already AE-encoded upstream → (B, T, N, d_force)
        e_force = c_force_enc.to(w_dtype)

        # --- Concatenate all ---
        c_sim = torch.cat(
            [e_floor, e_id, e_mat, e_mass, e_static, e_force],
            dim=-1,
        )  # (B, T, N, d_cond)

        # Zero out embeddings at padded point positions (padded batch mode)
        if point_mask is not None:
            c_sim = c_sim * point_mask.to(dtype=c_sim.dtype).unsqueeze(1).unsqueeze(-1)

        return c_sim

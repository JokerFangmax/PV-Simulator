# Simulation condition embedders for PV-Simulator.
# Encodes physics conditions (floor, object ID, material, mass, static, force, init)
# into per-point, per-timestep feature tensors.
#
# Per-object properties are expanded to per-point via point_obj_idx (N,) mapping.

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

    Output shape: (1, T, N, d_cond) where d_cond = sum of all sub-condition dims.

    Args:
        d_floor: Floor height embedding dim.
        d_id: Object ID embedding dim.
        d_mat: Material (friction, restitution) embedding dim.
        d_mass: Mass embedding dim.
        d_static: Static flag embedding dim.
        d_force: Force embedding dim (via causal temporal encoder).
        d_init: Initial state embedding dim.
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
        d_init: int = 64,
        max_objects: int = 16,
        force_encoder_mid: int = 128,
    ):
        super().__init__()
        self.d_cond = d_floor + d_id + d_mat + d_mass + d_static + d_force + d_init

        # Scalar / per-object encoders
        self.floor_mlp = ConditionMLP(1, d_floor)
        self.id_embed = nn.Embedding(max_objects, d_id)
        self.mat_mlp = ConditionMLP(2, d_mat)
        self.mass_mlp = ConditionMLP(1, d_mass)
        self.static_embed = nn.Embedding(2, d_static)

        # Time-varying force: (1, T_raw, N, 6) → (1, T, N, d_force)
        self.force_encoder = CausalTemporalEncoder(6, force_encoder_mid, d_force)

        # Initial state: pos(3) + vel(3) + ang_vel(3) + mask(1) = 10 dims
        self.init_mlp = ConditionMLP(10, d_init)

    def forward(
        self,
        c_floor: torch.Tensor,        # (1,) float — floor height
        c_id: torch.Tensor,            # (n_objects,) int — object IDs [0..max_objects)
        c_mat: torch.Tensor,           # (n_objects, 2) float — (friction, restitution)
        c_mass: torch.Tensor,          # (n_objects,) float — mass
        c_static: torch.Tensor,        # (n_objects,) int — static flag {0, 1}
        c_force_raw: torch.Tensor,     # (1, T_raw, N, 6) float — force + contact point
        c_init: torch.Tensor,          # (n_objects, 10) float — initial state + mask
        point_obj_idx: torch.Tensor,   # (N,) int — maps each point to its object
        T: int,                        # Number of latent time frames
    ) -> torch.Tensor:
        """Encode all conditions and return (1, T, N, d_cond)."""
        N = point_obj_idx.shape[0]

        # --- Per-object embeddings → gather to per-point → expand over T ---

        # Floor: scalar → (1, T, N, d_floor)
        e_floor = self.floor_mlp(c_floor.view(1, 1).float())  # (1, d_floor)
        e_floor = e_floor.view(1, 1, 1, -1).expand(1, T, N, -1)

        # Object ID: (n_objects,) → embed → gather → (1, T, N, d_id)
        e_id = self.id_embed(c_id)                          # (n_objects, d_id)
        e_id = e_id[point_obj_idx]                           # (N, d_id)
        e_id = e_id.view(1, 1, N, -1).expand(1, T, N, -1)

        # Material: (n_objects, 2) → MLP → gather → (1, T, N, d_mat)
        e_mat = self.mat_mlp(c_mat.float())                  # (n_objects, d_mat)
        e_mat = e_mat[point_obj_idx]                          # (N, d_mat)
        e_mat = e_mat.view(1, 1, N, -1).expand(1, T, N, -1)

        # Mass: (n_objects,) → MLP → gather → (1, T, N, d_mass)
        e_mass = self.mass_mlp(c_mass.float().unsqueeze(-1))  # (n_objects, d_mass)
        e_mass = e_mass[point_obj_idx]                         # (N, d_mass)
        e_mass = e_mass.view(1, 1, N, -1).expand(1, T, N, -1)

        # Static flag: (n_objects,) → embed → gather → (1, T, N, d_static)
        e_static = self.static_embed(c_static)               # (n_objects, d_static)
        e_static = e_static[point_obj_idx]                    # (N, d_static)
        e_static = e_static.view(1, 1, N, -1).expand(1, T, N, -1)

        # Init state: (n_objects, 10) → MLP → gather → (1, T, N, d_init)
        e_init = self.init_mlp(c_init.float())               # (n_objects, d_init)
        e_init = e_init[point_obj_idx]                        # (N, d_init)
        e_init = e_init.view(1, 1, N, -1).expand(1, T, N, -1)

        # --- Time-varying conditions ---

        # Force: (1, T_raw, N, 6) → causal encoder → (1, T, N, d_force)
        e_force = self.force_encoder(c_force_raw)

        # --- Concatenate all ---
        c_sim = torch.cat(
            [e_floor, e_id, e_mat, e_mass, e_static, e_force, e_init],
            dim=-1,
        )  # (1, T, N, d_cond)

        return c_sim

"""Physics-conditioned rigid-motion plus residual point-cloud decoder."""

from __future__ import annotations

import torch
import torch.nn as nn


class PhysicsEncoder(nn.Module):
    """Embed [stiffness, friction, restitution, mass] per object."""

    def __init__(self, attr_dim: int = 4, physics_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(attr_dim, physics_dim),
            nn.SiLU(),
            nn.Linear(physics_dim, physics_dim),
            nn.SiLU(),
        )

    def forward(self, physics_attrs: torch.Tensor) -> torch.Tensor:
        return self.net(physics_attrs)


def _rodrigues(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle rotations (..., 3) to matrices (..., 3, 3)."""
    theta = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / theta.clamp_min(1e-8)
    x, y, z = axis.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack([
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros,
    ], dim=-1).view(*axis.shape[:-1], 3, 3)
    identity = torch.eye(3, device=axis.device, dtype=axis.dtype).expand_as(skew)
    theta = theta.unsqueeze(-1)
    return identity + torch.sin(theta) * skew + (1.0 - torch.cos(theta)) * (skew @ skew)


class PhysicsConditionedRigidResidualDecoder(nn.Module):
    """Decode backbone point features as SE(3) coarse motion plus deformation.

    ``stiffness`` is physics attribute 0.  It exactly suppresses residuals and
    permits full rotation for rigid objects; soft objects receive the converse.
    """

    def __init__(self, latent_dim: int = 3, physics_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.physics_encoder = PhysicsEncoder(4, physics_dim)
        self.object_trunk = nn.Sequential(
            nn.Linear(latent_dim + physics_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.pose_head = nn.Linear(hidden_dim, 6)
        self.deformability_head = nn.Linear(hidden_dim, 1)
        self.residual_head = nn.Sequential(
            nn.Linear(latent_dim + physics_dim + 3, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.pose_head.weight)
        nn.init.zeros_(self.pose_head.bias)
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    @staticmethod
    def _object_mask(point_obj_idx, point_mask, num_objects):
        B, N = point_obj_idx.shape
        mask = torch.zeros(B, num_objects, device=point_obj_idx.device, dtype=torch.bool)
        for b in range(B):
            valid = point_mask[b] if point_mask is not None else torch.ones(N, device=mask.device, dtype=torch.bool)
            for obj_id in torch.unique(point_obj_idx[b][valid]).tolist():
                if 0 <= obj_id < num_objects:
                    mask[b, obj_id] = True
        return mask

    def forward(
        self,
        latent: torch.Tensor,
        canonical_points: torch.Tensor,
        physics_attrs: torch.Tensor,
        point_obj_idx: torch.Tensor,
        point_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return final positions, SE(3) coarse positions, residual, and gates."""
        B, T, N, C = latent.shape
        num_objects = physics_attrs.shape[1]
        if canonical_points.shape != (B, N, 3):
            raise ValueError(f"canonical_points must be {(B, N, 3)}, got {tuple(canonical_points.shape)}")

        physics = self.physics_encoder(physics_attrs)
        object_mask = self._object_mask(point_obj_idx, point_mask, num_objects)
        pooled = latent.new_zeros(B, T, num_objects, C)
        for b in range(B):
            for obj_id in range(num_objects):
                points = (point_obj_idx[b] == obj_id)
                if point_mask is not None:
                    points = points & point_mask[b]
                if torch.any(points):
                    pooled[b, :, obj_id] = latent[b, :, points].mean(dim=1)

        object_feature = self.object_trunk(torch.cat([
            pooled, physics.unsqueeze(1).expand(-1, T, -1, -1),
        ], dim=-1))
        pose = self.pose_head(object_feature)
        stiffness = physics_attrs[..., 0].clamp(0.0, 1.0).unsqueeze(1).unsqueeze(-1)
        axis_angle = pose[..., :3] * stiffness
        translation = pose[..., 3:]
        learned_deformability = torch.sigmoid(self.deformability_head(object_feature))
        deformability = (1.0 - stiffness) * learned_deformability

        coarse = latent.new_zeros(B, T, N, 3)
        for b in range(B):
            for obj_id in range(num_objects):
                points = (point_obj_idx[b] == obj_id)
                if point_mask is not None:
                    points = points & point_mask[b]
                if not torch.any(points):
                    continue
                rotation = _rodrigues(axis_angle[b, :, obj_id])
                template = canonical_points[b, points]
                coarse[b, :, points] = (
                    template.unsqueeze(0) @ rotation.transpose(-1, -2)
                    + translation[b, :, obj_id].unsqueeze(1)
                )

        point_object_idx = point_obj_idx.clamp(0, num_objects - 1)
        physics_per_point = physics[:, None].expand(-1, T, -1, -1).gather(
            2, point_object_idx[:, None, :, None].expand(-1, T, -1, physics.shape[-1]),
        )
        deformability_per_point = deformability.gather(
            2, point_object_idx[:, None, :, None].expand(-1, T, -1, 1),
        )
        residual = self.residual_head(torch.cat([
            latent, physics_per_point, canonical_points[:, None].expand(-1, T, -1, -1),
        ], dim=-1)) * deformability_per_point
        final = coarse + residual
        # Frame 0 is a hard condition, not a prediction target.
        coarse[:, 0] = canonical_points
        residual[:, 0] = 0
        final[:, 0] = canonical_points
        return {
            "positions": final,
            "coarse_positions": coarse,
            "residual": residual,
            "deformability": deformability,
            "object_mask": object_mask,
        }

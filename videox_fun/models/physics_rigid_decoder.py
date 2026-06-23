"""Physics-conditioned rigid alignment plus residual point-cloud decoder."""

from __future__ import annotations

import torch
import torch.nn as nn


class PhysicsEncoder(nn.Module):
    """Embed [stiffness, friction, restitution, mass] per object."""

    def __init__(self, attr_dim: int = 4, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(attr_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.SiLU(),
        )

    def forward(self, physics_attrs: torch.Tensor) -> torch.Tensor:
        return self.net(physics_attrs)


class PhysicsConditionedRigidResidualDecoder(nn.Module):
    """Project predicted coordinates onto per-object rigid motion via Kabsch."""

    def __init__(self, physics_dim: int = 64, num_objects: int = 1):
        super().__init__()
        self.physics_encoder = PhysicsEncoder(attr_dim=4, out_dim=physics_dim)
        self.num_objects = num_objects

        self.residual_mlp = nn.Sequential(
            nn.Linear(3 + 3 + physics_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )
        self.deformability_gate = nn.Sequential(
            nn.Linear(physics_dim, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)

    @staticmethod
    def _object_mask(
        point_obj_idx: torch.Tensor,
        point_mask: torch.Tensor | None,
        num_objects: int,
    ) -> torch.Tensor:
        B, N = point_obj_idx.shape
        object_mask = torch.zeros(
            B, num_objects, device=point_obj_idx.device, dtype=torch.bool,
        )
        for b in range(B):
            valid = (
                point_mask[b]
                if point_mask is not None
                else torch.ones(N, device=point_obj_idx.device, dtype=torch.bool)
            )
            for object_id in torch.unique(point_obj_idx[b, valid]).tolist():
                if 0 <= object_id < num_objects:
                    object_mask[b, object_id] = True
        return object_mask

    def kabsch_alignment(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        point_obj_idx: torch.Tensor,
        point_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Fit per-object masked SE(3) transforms from source to target.

        R uses a column-vector convention: target = R @ source + t.
        Therefore row-vector points use source @ R.transpose(-1, -2) + t.
        """
        B, T, N, _ = target.shape
        identity = torch.eye(3, device=target.device, dtype=target.dtype)
        R = identity.view(1, 1, 1, 3, 3).expand(
            B, T, self.num_objects, -1, -1,
        ).clone()
        t = target.new_zeros(B, T, self.num_objects, 3)
        svd_condition_number = target.new_full(
            (B, T, self.num_objects), float("nan"),
        )
        fallback_count = target.new_zeros(B, T, self.num_objects)
        centroid_error = target.new_zeros(B, T, self.num_objects)

        for b in range(B):
            valid_points = (
                point_mask[b]
                if point_mask is not None
                else torch.ones(N, device=target.device, dtype=torch.bool)
            )

            for object_id in range(self.num_objects):
                object_points = (point_obj_idx[b] == object_id) & valid_points
                num_points = int(object_points.sum().item())
                if num_points == 0:
                    continue

                source_obj = source[b, object_points]       # (P, 3)
                target_obj = target[b, :, object_points]    # (T, P, 3)

                source_mean = source_obj.mean(dim=0)        # (3,)
                target_mean = target_obj.mean(dim=1)        # (T, 3)
                centroid_error[b, :, object_id] = torch.linalg.vector_norm(
                    target_mean - source_mean,
                    dim=-1,
                )

                # Default and degenerate fallback: identity rotation plus
                # centroid translation.
                t[b, :, object_id] = target_mean - source_mean
                fallback_count[b, :, object_id] = 1.0
                if num_points < 3:
                    continue

                source_centered = source_obj - source_mean
                target_centered = target_obj - target_mean.unsqueeze(1)
                covariance = torch.einsum(
                    "pi,tpj->tij",
                    source_centered,
                    target_centered,
                )

                U, singular_values, Vh = torch.linalg.svd(
                    covariance,
                    full_matrices=False,
                )
                svd_condition_number[b, :, object_id] = (
                    singular_values[:, 0]
                    / singular_values[:, -1].clamp_min(1e-12)
                )
                non_degenerate = singular_values[:, -1] >= 1e-6
                if not torch.any(non_degenerate):
                    continue

                U_valid = U[non_degenerate]
                V_valid = Vh[non_degenerate].transpose(-1, -2)

                # Kabsch reflection correction: R = V D U^T.
                uncorrected_R = V_valid @ U_valid.transpose(-1, -2)
                correction = identity.expand(
                    uncorrected_R.shape[0], -1, -1,
                ).clone()
                correction[:, 2, 2] = torch.where(
                    torch.det(uncorrected_R) < 0,
                    correction.new_tensor(-1.0),
                    correction.new_tensor(1.0),
                )
                fitted_R = V_valid @ correction @ U_valid.transpose(-1, -2)

                R[b, non_degenerate, object_id] = fitted_R
                fallback_count[b, non_degenerate, object_id] = 0.0

                rotated_source_mean = torch.matmul(
                    source_mean.expand(fitted_R.shape[0], -1).unsqueeze(1),
                    fitted_R.transpose(-1, -2),
                ).squeeze(1)
                t[b, non_degenerate, object_id] = (
                    target_mean[non_degenerate] - rotated_source_mean
                )

        relative_rotation_angle = target.new_zeros(B, T, self.num_objects)
        if T > 1:
            relative_rotation = R[:, 1:] @ R[:, :-1].transpose(-1, -2)
            trace = relative_rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
            angles = torch.rad2deg(torch.acos(cosine))
            # Avoid reporting numerical trace noise as a physical rotation.
            relative_rotation_angle[:, 1:] = torch.where(
                cosine > 1.0 - 1e-6,
                torch.zeros_like(angles),
                angles,
            )

        diagnostics = {
            "svd_condition_number": svd_condition_number,
            "fallback_count": fallback_count,
            "centroid_error": centroid_error,
            "relative_rotation_angle": relative_rotation_angle,
        }
        return R, t, diagnostics

    def forward(
        self,
        latent: torch.Tensor,
        canonical_points: torch.Tensor,
        physics_attrs: torch.Tensor,
        point_obj_idx: torch.Tensor,
        point_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return Kabsch-aligned positions and gated pointwise residuals."""
        B, T, N, C = latent.shape
        if C != 3:
            raise ValueError(f"latent must have 3 channels, got {C}")
        if canonical_points.shape != (B, N, 3):
            raise ValueError(
                f"canonical_points must be {(B, N, 3)}, got "
                f"{tuple(canonical_points.shape)}"
            )
        if physics_attrs.shape[:2] != (B, self.num_objects):
            raise ValueError(
                f"physics_attrs must be ({B}, {self.num_objects}, 4), got "
                f"{tuple(physics_attrs.shape)}"
            )

        physics_cond = self.physics_encoder(physics_attrs)
        stiffness = physics_attrs[..., 0].clamp(0.0, 1.0)
        deformability = (
            (1.0 - stiffness).unsqueeze(-1)
            * self.deformability_gate(physics_cond)
        ).unsqueeze(1).expand(-1, T, -1, -1)

        object_mask = self._object_mask(
            point_obj_idx,
            point_mask,
            self.num_objects,
        )
        R, t, kabsch_diagnostics = self.kabsch_alignment(
            canonical_points,
            latent,
            point_obj_idx,
            point_mask,
        )

        coarse = latent.new_zeros(B, T, N, 3)
        for b in range(B):
            for object_id in range(self.num_objects):
                object_points = point_obj_idx[b] == object_id
                if point_mask is not None:
                    object_points = object_points & point_mask[b]
                if not torch.any(object_points):
                    continue

                source_obj = canonical_points[b, object_points]
                coarse[b, :, object_points] = (
                    torch.matmul(
                        source_obj.unsqueeze(0),
                        R[b, :, object_id].transpose(-1, -2),
                    )
                    + t[b, :, object_id].unsqueeze(1)
                )

        point_object_idx = point_obj_idx.clamp(0, self.num_objects - 1)
        physics_per_point = physics_cond[:, None].expand(-1, T, -1, -1).gather(
            2,
            point_object_idx[:, None, :, None].expand(
                -1, T, -1, physics_cond.shape[-1],
            ),
        )
        deformability_per_point = deformability.gather(
            2,
            point_object_idx[:, None, :, None].expand(-1, T, -1, 1),
        )

        residual = self.residual_mlp(torch.cat([
            latent,
            canonical_points[:, None].expand(-1, T, -1, -1),
            physics_per_point,
        ], dim=-1)) * deformability_per_point

        if point_mask is not None:
            residual = residual * point_mask[:, None, :, None].to(residual.dtype)

        positions = coarse + residual
        return {
            "positions": positions,
            "coarse_positions": coarse,
            "residual": residual,
            "deformability": deformability,
            "object_mask": object_mask,
            "diagnostics": kabsch_diagnostics,
        }

"""Mask-aware diagnostics for point-cloud trajectory simulation.

All position tensors use ``(B, T, N, 3)``.  ``point_mask`` selects valid
points in padded batches, while ``valid_frame_mask`` selects valid raw frames.
"""

from __future__ import annotations

from typing import Optional

import torch


def raw_frame_mask_from_valid_seq_mask(
    valid_seq_mask: Optional[torch.Tensor],
    t_latent: int,
    n_points: int,
    t_raw: int,
    temporal_stride: int = 4,
) -> Optional[torch.Tensor]:
    """Expand latent token validity to raw-frame validity for a causal AE.

    Latent frame 0 represents raw frame 0.  Every later latent frame represents
    the following ``temporal_stride`` raw frames, matching CausalAE's [1, 4,
    4, ...] chunking convention.
    """
    if valid_seq_mask is None:
        return None
    if valid_seq_mask.ndim != 2 or valid_seq_mask.shape[1] != t_latent * n_points:
        raise ValueError(
            "valid_seq_mask must have shape (B, t_latent * n_points); got "
            f"{tuple(valid_seq_mask.shape)} for t_latent={t_latent}, n_points={n_points}."
        )

    latent_frame_mask = valid_seq_mask.view(-1, t_latent, n_points).any(dim=2)
    raw_to_latent = torch.zeros(t_raw, device=valid_seq_mask.device, dtype=torch.long)
    if t_raw > 1:
        raw_to_latent[1:] = (
            (torch.arange(1, t_raw, device=valid_seq_mask.device) - 1) // temporal_stride + 1
        ).clamp_max(t_latent - 1)
    return latent_frame_mask[:, raw_to_latent]


def _point_mask_or_all(
    positions: torch.Tensor,
    point_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    B, _, N, _ = positions.shape
    if point_mask is None:
        return torch.ones(B, N, device=positions.device, dtype=torch.bool)
    return point_mask.to(device=positions.device, dtype=torch.bool)


def _frame_mask_or_all(
    positions: torch.Tensor,
    valid_frame_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    B, T, _, _ = positions.shape
    if valid_frame_mask is None:
        return torch.ones(B, T, device=positions.device, dtype=torch.bool)
    return valid_frame_mask.to(device=positions.device, dtype=torch.bool)


def _symmetric_chamfer(pred_points: torch.Tensor, target_points: torch.Tensor) -> torch.Tensor:
    squared_distances = torch.cdist(pred_points, target_points).square()
    return 0.5 * (
        squared_distances.min(dim=1).values.mean()
        + squared_distances.min(dim=0).values.mean()
    )


@torch.no_grad()
def knn_edge_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    point_obj_idx: torch.Tensor,
    k: int = 5,
    point_mask: Optional[torch.Tensor] = None,
    valid_frame_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean absolute error of frame-0, same-object KNN edge lengths over time."""
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {pred.shape} and {target.shape}")
    if k < 1:
        raise ValueError("k must be >= 1")

    B, _, N, _ = target.shape
    valid_points = _point_mask_or_all(target, point_mask)
    valid_frames = _frame_mask_or_all(target, valid_frame_mask)
    total_error = target.new_tensor(0.0)
    total_edges = target.new_tensor(0.0)

    for batch_idx in range(B):
        frame_idx = valid_frames[batch_idx].nonzero(as_tuple=True)[0]
        if frame_idx.numel() == 0:
            continue
        for obj_id in torch.unique(point_obj_idx[batch_idx][valid_points[batch_idx]]).tolist():
            obj_points = (point_obj_idx[batch_idx] == obj_id) & valid_points[batch_idx]
            point_idx = obj_points.nonzero(as_tuple=True)[0]
            if point_idx.numel() < 2:
                continue

            target_frame0 = target[batch_idx, 0, point_idx]
            pairwise = torch.cdist(target_frame0, target_frame0)
            pairwise.fill_diagonal_(torch.inf)
            k_eff = min(k, point_idx.numel() - 1)
            neighbors = pairwise.topk(k_eff, largest=False).indices

            pred_points = pred[batch_idx, frame_idx][:, point_idx]
            target_points = target[batch_idx, frame_idx][:, point_idx]
            pred_edges = torch.linalg.vector_norm(
                pred_points.unsqueeze(2) - pred_points[:, neighbors], dim=-1
            )
            target_edges = torch.linalg.vector_norm(
                target_points.unsqueeze(2) - target_points[:, neighbors], dim=-1
            )
            total_error += (pred_edges - target_edges).abs().sum()
            total_edges += pred_edges.numel()

    return total_error / total_edges.clamp_min(1)


@torch.no_grad()
def frame0_error(
    decoded_trajectory: torch.Tensor,
    gt_first_frame: torch.Tensor,
    point_mask: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Return position MSE and Chamfer distance for the decoded first frame."""
    pred_frame0 = decoded_trajectory[:, 0]
    target_frame0 = gt_first_frame[:, 0] if gt_first_frame.ndim == 4 else gt_first_frame
    valid_points = _point_mask_or_all(decoded_trajectory, point_mask)
    total_sq_error = pred_frame0.new_tensor(0.0)
    total_values = pred_frame0.new_tensor(0.0)
    total_chamfer = pred_frame0.new_tensor(0.0)
    total_samples = pred_frame0.new_tensor(0.0)

    for batch_idx in range(pred_frame0.shape[0]):
        valid_idx = valid_points[batch_idx].nonzero(as_tuple=True)[0]
        if valid_idx.numel() == 0:
            continue
        pred_points = pred_frame0[batch_idx, valid_idx]
        target_points = target_frame0[batch_idx, valid_idx]
        total_sq_error += (pred_points - target_points).square().sum()
        total_values += pred_points.numel()
        total_chamfer += _symmetric_chamfer(pred_points, target_points)
        total_samples += 1

    return {
        "frame0_position_mse": total_sq_error / total_values.clamp_min(1),
        "frame0_chamfer": total_chamfer / total_samples.clamp_min(1),
    }


@torch.no_grad()
def velocity_drift(
    pred: torch.Tensor,
    target: torch.Tensor,
    point_mask: Optional[torch.Tensor] = None,
    valid_frame_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MSE between finite-difference position velocities on valid adjacent frames."""
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {pred.shape} and {target.shape}")
    pred_velocity = pred[:, 1:] - pred[:, :-1]
    target_velocity = target[:, 1:] - target[:, :-1]
    valid_points = _point_mask_or_all(pred, point_mask)
    valid_frames = _frame_mask_or_all(pred, valid_frame_mask)
    valid_pairs = valid_frames[:, 1:] & valid_frames[:, :-1]
    mask = valid_pairs.unsqueeze(-1) & valid_points.unsqueeze(1)
    return (
        (pred_velocity - target_velocity).square() * mask.unsqueeze(-1)
    ).sum() / (mask.sum() * 3).clamp_min(1)


@torch.no_grad()
def ae_reconstruction_chamfer(
    ae,
    x_raw: torch.Tensor,
    point_mask: Optional[torch.Tensor] = None,
    valid_frame_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Encode/decode positions and return mean per-frame symmetric Chamfer distance."""
    t_raw = x_raw.shape[1]
    target_positions = x_raw[..., :3]
    reconstructed_positions = ae.decode(ae.encode(target_positions), t_raw)
    valid_points = _point_mask_or_all(target_positions, point_mask)
    valid_frames = _frame_mask_or_all(target_positions, valid_frame_mask)
    total_chamfer = target_positions.new_tensor(0.0)
    total_frames = target_positions.new_tensor(0.0)

    for batch_idx in range(target_positions.shape[0]):
        point_idx = valid_points[batch_idx].nonzero(as_tuple=True)[0]
        if point_idx.numel() == 0:
            continue
        for frame_idx in valid_frames[batch_idx].nonzero(as_tuple=True)[0]:
            total_chamfer += _symmetric_chamfer(
                reconstructed_positions[batch_idx, frame_idx, point_idx],
                target_positions[batch_idx, frame_idx, point_idx],
            )
            total_frames += 1

    return total_chamfer / total_frames.clamp_min(1)

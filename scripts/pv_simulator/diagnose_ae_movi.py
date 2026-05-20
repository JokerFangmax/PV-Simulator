"""AE-only diagnostics for MOVI point-cloud trajectories.

This script isolates the Stage 0 autoencoder by running:

    x_s_raw --encode--> z --decode--> x_hat

on a MOVI sample, then reporting reconstruction and shape-preservation metrics.
It is intended to answer:

1. Does the frozen AE alone already distort object shape?
2. If yes, how much of the error is global motion drift vs local shape damage?

Outputs:
  - gt.mp4 / recon.mp4: point-cloud motion videos
  - gt.npy / recon.npy: raw arrays for further inspection
  - metrics.json: scalar diagnostics

Usage:
    python scripts/pv_simulator/diagnose_ae_movi.py \
        --ae_ckpt_dir outputs/stage0/diag/stage_mlp/causal-mae-long-fp32-50k/final \
        --shard_root datasets/movi_ab_500_1_shards \
        --sample_idx 0 \
        --output_dir outputs/ae_diag/sample0
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

current_file_path = os.path.abspath(__file__)
for _root in [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]:
    if _root not in sys.path:
        sys.path.insert(0, _root)

from infer_stage1 import load_movi_sample, load_movi_shard_sample
from visualize import visualize_point_cloud_motion
from videox_fun.models.sim_ae import CausalAE


def parse_args():
    parser = argparse.ArgumentParser(description="AE-only MOVI reconstruction diagnostics")
    parser.add_argument("--ae_ckpt_dir", type=str, required=True,
                        help="Path to Stage 0 AE checkpoint directory.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data_dir", type=str,
                       help="MOVI sample directory containing point_cloud_states.pkl + metadata.json")
    group.add_argument("--shard_root", type=str,
                       help="Sharded MOVI dataset root (dataset root or webdataset root)")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Sample index inside manifest.jsonl when --shard_root is used.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save videos, arrays, and metrics.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--views", type=str, nargs="+",
                        default=["birdseye", "side", "iso"],
                        choices=["birdseye", "side", "front", "iso"])
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--max_pair_samples", type=int, default=4096,
                        help="Maximum number of within-object point pairs to evaluate per object/frame.")
    return parser.parse_args()


def _encode_decode_state(ae, x_raw):
    """Encode/decode full state tensor (B, T_raw, N, 6) via the frozen AE."""
    pos_enc = ae.encode(x_raw[..., :3])
    vel_enc = ae.encode(x_raw[..., 3:6])
    pos_hat = ae.decode(pos_enc, x_raw.shape[1])
    vel_hat = ae.decode(vel_enc, x_raw.shape[1])
    z = torch.cat([pos_enc, vel_enc], dim=-1)
    x_hat = torch.cat([pos_hat, vel_hat], dim=-1)
    return z, x_hat


def _sample_pairs(num_points: int, max_pairs: int) -> np.ndarray:
    """Return point-index pairs for pairwise distance diagnostics."""
    if num_points < 2:
        return np.zeros((0, 2), dtype=np.int64)

    full_pairs = num_points * (num_points - 1) // 2
    if full_pairs <= max_pairs:
        pairs = []
        for i in range(num_points):
            for j in range(i + 1, num_points):
                pairs.append((i, j))
        return np.asarray(pairs, dtype=np.int64)

    rng = np.random.default_rng(0)
    pairs = set()
    while len(pairs) < max_pairs:
        ij = rng.integers(0, num_points, size=2)
        i, j = int(ij[0]), int(ij[1])
        if i == j:
            continue
        if i > j:
            i, j = j, i
        pairs.add((i, j))
    return np.asarray(sorted(pairs), dtype=np.int64)


def _kabsch_aligned_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """RMSE after best-fit rigid alignment from pred -> target."""
    if pred.shape[0] < 3:
        return float(np.sqrt(np.mean((pred - target) ** 2)))

    pred_center = pred.mean(axis=0, keepdims=True)
    target_center = target.mean(axis=0, keepdims=True)
    pred_c = pred - pred_center
    target_c = target - target_center

    cov = pred_c.T @ target_c
    u, _, vh = np.linalg.svd(cov, full_matrices=False)
    rot = vh.T @ u.T
    if np.linalg.det(rot) < 0:
        vh[-1, :] *= -1
        rot = vh.T @ u.T

    pred_aligned = pred_c @ rot + target_center
    return float(np.sqrt(np.mean((pred_aligned - target) ** 2)))


def _compute_shape_metrics(
    pos_gt: np.ndarray,
    pos_hat: np.ndarray,
    point_obj_idx: np.ndarray,
    max_pair_samples: int,
) -> Dict[str, float]:
    """Compute shape-preservation metrics on position channels only."""
    obj_ids = np.unique(point_obj_idx)
    pairwise_errs: List[float] = []
    rigid_errs: List[float] = []
    obj_pairwise_means: Dict[str, float] = {}
    obj_rigid_means: Dict[str, float] = {}

    for obj_id in obj_ids:
        obj_mask = point_obj_idx == obj_id
        obj_gt = pos_gt[:, obj_mask, :]
        obj_hat = pos_hat[:, obj_mask, :]
        if obj_gt.shape[1] == 0:
            continue

        pair_idx = _sample_pairs(obj_gt.shape[1], max_pair_samples)
        obj_pairwise_errs: List[float] = []
        obj_rigid_errs: List[float] = []

        for t in range(obj_gt.shape[0]):
            gt_t = obj_gt[t]
            hat_t = obj_hat[t]

            if len(pair_idx) > 0:
                gt_d = np.linalg.norm(gt_t[pair_idx[:, 0]] - gt_t[pair_idx[:, 1]], axis=-1)
                hat_d = np.linalg.norm(hat_t[pair_idx[:, 0]] - hat_t[pair_idx[:, 1]], axis=-1)
                obj_pairwise_errs.append(float(np.sqrt(np.mean((hat_d - gt_d) ** 2))))

            obj_rigid_errs.append(_kabsch_aligned_rmse(hat_t, gt_t))

        if obj_pairwise_errs:
            obj_pairwise_means[f"obj{int(obj_id)}"] = float(np.mean(obj_pairwise_errs))
            pairwise_errs.extend(obj_pairwise_errs)
        obj_rigid_means[f"obj{int(obj_id)}"] = float(np.mean(obj_rigid_errs))
        rigid_errs.extend(obj_rigid_errs)

    metrics: Dict[str, float] = {
        "pairwise_dist_rmse_mean": float(np.mean(pairwise_errs)) if pairwise_errs else 0.0,
        "rigid_aligned_pos_rmse_mean": float(np.mean(rigid_errs)) if rigid_errs else 0.0,
    }
    for key, value in obj_pairwise_means.items():
        metrics[f"{key}_pairwise_dist_rmse"] = value
    for key, value in obj_rigid_means.items():
        metrics[f"{key}_rigid_aligned_pos_rmse"] = value
    return metrics


def _summarize_metrics(x_gt: np.ndarray, x_hat: np.ndarray, point_obj_idx: np.ndarray,
                       max_pair_samples: int) -> Dict[str, float]:
    diff = x_hat - x_gt
    pos_diff = diff[..., :3]
    vel_diff = diff[..., 3:6]

    metrics: Dict[str, float] = {
        "pos_mse": float(np.mean(pos_diff ** 2)),
        "pos_mae": float(np.mean(np.abs(pos_diff))),
        "vel_mse": float(np.mean(vel_diff ** 2)),
        "vel_mae": float(np.mean(np.abs(vel_diff))),
        "all_mse": float(np.mean(diff ** 2)),
        "all_mae": float(np.mean(np.abs(diff))),
        "gt_pos_norm_mean": float(np.mean(np.linalg.norm(x_gt[..., :3], axis=-1))),
        "recon_pos_norm_mean": float(np.mean(np.linalg.norm(x_hat[..., :3], axis=-1))),
    }
    metrics.update(
        _compute_shape_metrics(
            pos_gt=x_gt[..., :3],
            pos_hat=x_hat[..., :3],
            point_obj_idx=point_obj_idx,
            max_pair_samples=max_pair_samples,
        )
    )
    return metrics


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    device = torch.device(args.device)

    if args.data_dir is not None:
        sample = load_movi_sample(args.data_dir)
        sample_ref = args.data_dir
    else:
        sample = load_movi_shard_sample(args.shard_root, args.sample_idx)
        sample_ref = f"{args.shard_root}#{args.sample_idx}"

    x_gt = sample["point_states"].unsqueeze(0).to(device=device, dtype=dtype)
    point_obj_idx = sample["point_obj_idx"].numpy().astype(np.int64)

    ae = CausalAE.load(args.ae_ckpt_dir)
    ae = ae.to(device=device, dtype=dtype)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        z, x_hat = _encode_decode_state(ae, x_gt)

    x_gt_np = x_gt[0].float().cpu().numpy()
    x_hat_np = x_hat[0].float().cpu().numpy()
    z_np = z[0].float().cpu().numpy()

    metrics = _summarize_metrics(
        x_gt=x_gt_np,
        x_hat=x_hat_np,
        point_obj_idx=point_obj_idx,
        max_pair_samples=args.max_pair_samples,
    )
    metrics.update({
        "sample_ref": sample_ref,
        "ae_ckpt_dir": args.ae_ckpt_dir,
        "ae_class": ae.__class__.__name__,
        "d_latent": int(ae.d_latent),
        "latent_shape_T_N_C": list(z_np.shape),
        "raw_shape_T_N_C": list(x_gt_np.shape),
        "num_objects": int(sample["n_objects"]),
        "num_points": int(sample["N"]),
        "T_raw": int(sample["T_raw"]),
    })

    np.save(os.path.join(args.output_dir, "gt.npy"), x_gt_np)
    np.save(os.path.join(args.output_dir, "recon.npy"), x_hat_np)
    np.save(os.path.join(args.output_dir, "latent.npy"), z_np)

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    visualize_point_cloud_motion(
        point_states=x_gt_np,
        point_obj_idx=point_obj_idx,
        output_path=os.path.join(args.output_dir, "gt.mp4"),
        fps=args.fps,
        views=args.views,
        dpi=args.dpi,
    )
    visualize_point_cloud_motion(
        point_states=x_hat_np,
        point_obj_idx=point_obj_idx,
        output_path=os.path.join(args.output_dir, "recon.mp4"),
        fps=args.fps,
        views=args.views,
        dpi=args.dpi,
    )

    print(json.dumps(metrics, indent=2))
    print(f"Saved diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()

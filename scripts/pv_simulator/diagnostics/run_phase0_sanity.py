"""Phase 0 sanity checks for PV-Simulator Stage 1.

This script diagnoses the existing Stage 1 simulation trajectory pipeline
without changing SimTransformer architecture, losses, Stage 2, PBD, XPBD, or
contact-aware modules.

Outputs are written to:
    experiments/stage1_diagnostics/<YYYYMMDD_HHMMSS>_phase0/

Example:
    python scripts/pv_simulator/diagnostics/run_phase0_sanity.py \
        --dataset_type movi \
        --data_root datasets/movi_ab_50k_shards \
        --ae_ckpt_dir outputs/stage0/sweeps_mae/training/lr1e-3-cosine/final \
        --padded_batch \
        --max_objects 5 \
        --max_T_raw 21 \
        --max_points_per_object 200 \
        --modes audit ae baselines deterministic_overfit \
        --overfit_steps 50 \
        --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from datetime import datetime
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_loss_weighting_for_sd3
from torch.utils.data import DataLoader, Subset

CURRENT_FILE = os.path.abspath(__file__)
for _root in [
    os.path.dirname(CURRENT_FILE),
    os.path.dirname(os.path.dirname(CURRENT_FILE)),
    os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE))),
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE)))),
]:
    if _root not in sys.path:
        sys.path.insert(0, _root)

from videox_fun.data.dataset_simulation import (  # noqa: E402
    MoviSimulationDataset,
    SimulationDataset,
    sim_collate_fn,
    sim_collate_fn_padded,
)
from videox_fun.models.sim_ae import CausalAE  # noqa: E402
from videox_fun.models.sim_condition import SimConditionEmbedder  # noqa: E402
from videox_fun.models.sim_transformer import SimTransformer  # noqa: E402


PHASE0_MODES = [
    "audit",
    "ae",
    "deterministic_overfit",
    "stochastic_overfit",
    "baselines",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 0 sanity checks for Stage 1 simulation training."
    )

    parser.add_argument("--modes", nargs="+", default=["audit", "ae", "baselines"],
                        choices=PHASE0_MODES + ["all"],
                        help="Diagnostics to run. Use 'all' for every Phase 0 mode.")
    parser.add_argument("--output_root", type=str,
                        default="experiments/stage1_diagnostics",
                        help="Parent directory for timestamped diagnostic outputs.")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Optional suffix. Defaults to <timestamp>_phase0.")

    parser.add_argument("--dataset_type", type=str, default="movi",
                        choices=["movi", "simulation"])
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--ann_path", type=str, default=None,
                        help="Required for --dataset_type simulation.")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="First sample index for diagnostics.")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of sequential samples to include in one batch.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--padded_batch", action="store_true",
                        help="Use Stage 1 padded batching.")
    parser.add_argument("--max_T_raw", type=int, default=21)
    parser.add_argument("--max_objects", type=int, default=5)
    parser.add_argument("--max_points_per_object", type=int, default=200)
    parser.add_argument("--center_on_contact", action="store_true")

    parser.add_argument("--ae_ckpt_dir", type=str, default=None,
                        help="Stage 0 AE checkpoint. Required for AE and overfit modes.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "bf16"])

    parser.add_argument("--d_state", type=int, default=None,
                        help="Defaults to 2 * AE d_latent.")
    parser.add_argument("--d_sim", type=int, default=256)
    parser.add_argument("--sim_ffn_dim", type=int, default=1024)
    parser.add_argument("--sim_num_heads", type=int, default=8)
    parser.add_argument("--sim_num_layers", type=int, default=10)
    parser.add_argument("--use_factorized_attention", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--use_temporal_correspondence", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--use_temporal_rope", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--use_object_local_attention", action="store_true")
    parser.add_argument("--disable_point_anchor", action="store_true")

    parser.add_argument("--train_sampling_steps", type=int, default=1000)
    parser.add_argument("--weighting_scheme", type=str, default="logit_normal",
                        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])
    parser.add_argument("--fixed_timestep_index", type=int, default=None,
                        help="Deterministic overfit timestep index. Defaults to schedule midpoint.")
    parser.add_argument("--overfit_steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--snapshot_every", type=int, default=0,
                        help="If >0, save decoded prediction npz snapshots every N steps.")

    parser.add_argument("--dt", type=float, default=1.0 / 12.0,
                        help="Timestep for naive baselines.")
    parser.add_argument("--gravity", type=float, default=-9.81)
    parser.add_argument("--gravity_axis", type=str, default="auto",
                        choices=["auto", "x", "y", "z"])

    parser.add_argument("--max_chamfer_points", type=int, default=512)
    parser.add_argument("--knn_k", type=int, default=8)
    parser.add_argument("--collapse_ratio_threshold", type=float, default=0.2)
    parser.add_argument("--contact_threshold", type=float, default=1e-8)
    parser.add_argument("--contact_window_radius", type=int, default=2)
    parser.add_argument("--save_npz_snapshots", action="store_true")
    parser.add_argument("--save_visualizations", action="store_true")
    parser.add_argument("--vis_fps", type=int, default=10)

    args = parser.parse_args()
    if "all" in args.modes:
        args.modes = PHASE0_MODES
    args.modes = sorted(set(args.modes), key=PHASE0_MODES.index)
    if args.dataset_type == "simulation" and args.ann_path is None:
        parser.error("--ann_path is required when --dataset_type simulation")
    if args.num_samples < 1:
        parser.error("--num_samples must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch_size must be >= 1")
    if args.overfit_steps < 1:
        parser.error("--overfit_steps must be >= 1")
    if args.knn_k < 1:
        parser.error("--knn_k must be >= 1")
    if any(mode in args.modes for mode in ("ae", "deterministic_overfit", "stochastic_overfit")):
        if args.ae_ckpt_dir is None:
            parser.error("--ae_ckpt_dir is required for AE and overfit modes")
    return args


def make_output_dir(args: argparse.Namespace) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{stamp}_phase0"
    out_dir = os.path.join(args.output_root, run_name)
    os.makedirs(out_dir, exist_ok=False)
    return out_dir


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "bf16" and device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def build_dataset(args: argparse.Namespace):
    if args.dataset_type == "movi":
        return MoviSimulationDataset(
            data_root=args.data_root,
            max_objects=args.max_objects,
            max_T_raw=args.max_T_raw if args.padded_batch else None,
            center_on_contact=args.center_on_contact,
        )
    return SimulationDataset(
        ann_path=args.ann_path,
        data_root=args.data_root,
        load_video=False,
    )


def build_batch(args: argparse.Namespace) -> dict[str, Any]:
    dataset = build_dataset(args)
    if args.sample_idx < 0 or args.sample_idx >= len(dataset):
        raise IndexError(f"sample_idx={args.sample_idx} out of range for len={len(dataset)}")
    end_idx = min(args.sample_idx + args.num_samples, len(dataset))
    subset = Subset(dataset, list(range(args.sample_idx, end_idx)))
    if args.padded_batch:
        collate_fn = partial(
            sim_collate_fn_padded,
            max_T_raw=args.max_T_raw,
            max_objects=args.max_objects,
            max_points_per_object=args.max_points_per_object,
        )
    else:
        collate_fn = sim_collate_fn
    loader = DataLoader(
        subset,
        batch_size=min(args.batch_size, len(subset)),
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
    )
    batch = next(iter(loader))
    batch["_dataset_len"] = len(dataset)
    batch["_sample_indices"] = list(range(args.sample_idx, end_idx))[: batch["x_s_raw"].shape[0]]
    return batch


def tensor_to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return tensor_to_jsonable(value.item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): tensor_to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_to_jsonable(v) for v in value]
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def scalarize(prefix: str, value: Any, rows: list[dict[str, Any]]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            scalarize(f"{prefix}.{key}" if prefix else str(key), child, rows)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            scalarize(f"{prefix}.{idx}", child, rows)
    elif isinstance(value, (float, int, str, bool)) or value is None:
        rows.append({"metric": prefix, "value": value})


def save_metrics(metrics: dict[str, Any], out_dir: str) -> None:
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(tensor_to_jsonable(metrics), f, indent=2, sort_keys=True)
    rows: list[dict[str, Any]] = []
    scalarize("", tensor_to_jsonable(metrics), rows)
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def save_config(args: argparse.Namespace, out_dir: str) -> None:
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)


def move_batch_to_device(batch: dict[str, Any], device: torch.device,
                         dtype: torch.dtype) -> dict[str, Any]:
    moved = {}
    float_keys = {"x_s_raw", "c_force_raw", "c_floor", "c_mat", "c_mass"}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if key in float_keys:
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def get_masks(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x_s_raw = batch["x_s_raw"]
    B, T_raw, N, _ = x_s_raw.shape
    if "point_mask" in batch:
        point_mask = batch["point_mask"].to(device=device, dtype=torch.bool)
    else:
        point_mask = torch.ones(B, N, device=device, dtype=torch.bool)

    if isinstance(batch.get("T_raw"), torch.Tensor) and batch["T_raw"].ndim == 1:
        t_raw = batch["T_raw"].to(device=device)
        t_idx = torch.arange(T_raw, device=device).unsqueeze(0)
        frame_mask = t_idx < t_raw.unsqueeze(1)
    else:
        frame_mask = torch.ones(B, T_raw, device=device, dtype=torch.bool)
    return point_mask, frame_mask


def valid_position_stats(pos: torch.Tensor, point_mask: torch.Tensor,
                         frame_mask: torch.Tensor) -> dict[str, float]:
    mask = frame_mask.unsqueeze(-1) & point_mask.unsqueeze(1)
    vals = pos[mask]
    if vals.numel() == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "bbox_diag": 1.0}
    flat = vals.reshape(-1, 3)
    mins = flat.min(dim=0).values
    maxs = flat.max(dim=0).values
    return {
        "min": float(vals.min().item()),
        "max": float(vals.max().item()),
        "mean": float(vals.mean().item()),
        "std": float(vals.std(unbiased=False).item()),
        "bbox_diag": float(torch.linalg.vector_norm(maxs - mins).clamp_min(1e-8).item()),
    }


def masked_position_rmse(pred_pos: torch.Tensor, gt_pos: torch.Tensor,
                         point_mask: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
    mask = frame_mask.unsqueeze(-1) & point_mask.unsqueeze(1)
    sq = (pred_pos - gt_pos).square().sum(dim=-1)
    return torch.sqrt((sq * mask.float()).sum() / (mask.float().sum() * 3).clamp_min(1))


def sampled_chamfer(pred_pos: torch.Tensor, gt_pos: torch.Tensor,
                    point_mask: torch.Tensor, frame_mask: torch.Tensor,
                    max_points: int) -> torch.Tensor:
    total = pred_pos.new_tensor(0.0)
    count = pred_pos.new_tensor(0.0)
    gen = torch.Generator(device=pred_pos.device).manual_seed(0)
    for b in range(pred_pos.shape[0]):
        valid_points = point_mask[b].nonzero(as_tuple=True)[0]
        if valid_points.numel() == 0:
            continue
        if valid_points.numel() > max_points:
            perm = torch.randperm(valid_points.numel(), device=pred_pos.device, generator=gen)
            valid_points = valid_points[perm[:max_points]]
        for t in frame_mask[b].nonzero(as_tuple=True)[0]:
            p = pred_pos[b, t, valid_points]
            g = gt_pos[b, t, valid_points]
            d = torch.cdist(p.float(), g.float()).square()
            total = total + 0.5 * (d.min(dim=1).values.mean() + d.min(dim=0).values.mean())
            count = count + 1
    return total / count.clamp_min(1)


def local_geometry_metrics(pred_pos: torch.Tensor, gt_pos: torch.Tensor,
                           point_obj_idx: torch.Tensor, point_mask: torch.Tensor,
                           frame_mask: torch.Tensor, k: int,
                           collapse_threshold: float) -> dict[str, float]:
    total_sq = pred_pos.new_tensor(0.0)
    total_abs = pred_pos.new_tensor(0.0)
    total_edges = pred_pos.new_tensor(0.0)
    collapse_count = pred_pos.new_tensor(0.0)
    ratio_sum = pred_pos.new_tensor(0.0)
    eps = 1e-6
    for b in range(pred_pos.shape[0]):
        obj_ids = torch.unique(point_obj_idx[b][point_mask[b]])
        frames = frame_mask[b].nonzero(as_tuple=True)[0]
        for obj_id in obj_ids.tolist():
            obj_idx = ((point_obj_idx[b] == obj_id) & point_mask[b]).nonzero(as_tuple=True)[0]
            if obj_idx.numel() < 2:
                continue
            frame0 = gt_pos[b, 0, obj_idx].float()
            pairwise = torch.cdist(frame0, frame0)
            pairwise.fill_diagonal_(torch.inf)
            k_eff = min(k, obj_idx.numel() - 1)
            nbr = torch.topk(pairwise, k_eff, largest=False).indices
            for t in frames:
                pred_obj = pred_pos[b, t, obj_idx].float()
                gt_obj = gt_pos[b, t, obj_idx].float()
                pred_edges = torch.linalg.vector_norm(
                    pred_obj.unsqueeze(1) - pred_obj[nbr], dim=-1,
                )
                gt_edges = torch.linalg.vector_norm(
                    gt_obj.unsqueeze(1) - gt_obj[nbr], dim=-1,
                )
                diff = pred_edges - gt_edges
                ratio = pred_edges / gt_edges.clamp_min(eps)
                total_sq = total_sq + diff.square().sum()
                total_abs = total_abs + diff.abs().sum()
                ratio_sum = ratio_sum + ratio.sum()
                collapse_count = collapse_count + (ratio < collapse_threshold).float().sum()
                total_edges = total_edges + diff.numel()
    denom = total_edges.clamp_min(1)
    return {
        "rigidity_edge_rmse": float(torch.sqrt(total_sq / denom).item()),
        "rigidity_edge_mae": float((total_abs / denom).item()),
        "local_collapse_score": float((collapse_count / denom).item()),
        "local_edge_ratio_mean": float((ratio_sum / denom).item()),
        "local_edge_count": int(total_edges.item()),
    }


def contact_frame_mask(batch: dict[str, Any], threshold: float,
                       frame_mask: torch.Tensor) -> torch.Tensor | None:
    c_force_raw = batch.get("c_force_raw")
    if c_force_raw is None:
        return None
    force_norm = torch.linalg.vector_norm(c_force_raw[..., :3].float(), dim=-1)
    contact_norm = torch.linalg.vector_norm(c_force_raw[..., 3:6].float(), dim=-1)
    contact = ((force_norm > threshold) | (contact_norm > threshold)).any(dim=-1)
    contact = contact & frame_mask
    if not contact.any():
        return None
    return contact


def expand_contact_window(contact: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return contact
    B, T = contact.shape
    expanded = contact.clone()
    for offset in range(1, radius + 1):
        expanded[:, offset:] |= contact[:, :-offset]
        expanded[:, : T - offset] |= contact[:, offset:]
    return expanded


def compute_position_metrics(name: str, pred: torch.Tensor, target: torch.Tensor,
                             batch: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    point_mask, frame_mask = get_masks(batch, pred.device)
    pred_pos = pred[..., :3]
    gt_pos = target[..., :3]
    stats = valid_position_stats(gt_pos, point_mask, frame_mask)
    rmse = masked_position_rmse(pred_pos, gt_pos, point_mask, frame_mask)
    metrics: dict[str, Any] = {
        "position_rmse": float(rmse.item()),
        "normalized_position_rmse": float((rmse / stats["bbox_diag"]).item()),
        "position_bbox_diag": stats["bbox_diag"],
        "chamfer": float(sampled_chamfer(
            pred_pos, gt_pos, point_mask, frame_mask, args.max_chamfer_points,
        ).item()),
    }
    metrics.update(local_geometry_metrics(
        pred_pos=pred_pos,
        gt_pos=gt_pos,
        point_obj_idx=batch["point_obj_idx"].to(pred.device),
        point_mask=point_mask,
        frame_mask=frame_mask,
        k=args.knn_k,
        collapse_threshold=args.collapse_ratio_threshold,
    ))
    contact = contact_frame_mask(batch, args.contact_threshold, frame_mask)
    if contact is not None:
        contact_window = expand_contact_window(contact, args.contact_window_radius)
        metrics["contact_frames"] = int(contact.sum().item())
        metrics["contact_window_frames"] = int(contact_window.sum().item())
        metrics["contact_window_position_rmse"] = float(masked_position_rmse(
            pred_pos,
            gt_pos,
            point_mask,
            contact_window,
        ).item())
    else:
        metrics["contact_frames"] = 0
        metrics["contact_window_frames"] = 0
        metrics["contact_window_position_rmse"] = None
    return {name: metrics}


def encode_state(ae: CausalAE, x_raw: torch.Tensor) -> torch.Tensor:
    pos_enc = ae.encode(x_raw[..., :3])
    vel_enc = ae.encode(x_raw[..., 3:6])
    return torch.cat([pos_enc, vel_enc], dim=-1)


def decode_state(ae: CausalAE, z: torch.Tensor, t_raw: int) -> torch.Tensor:
    d = ae.d_latent
    pos = ae.decode(z[..., :d], t_raw)
    vel = ae.decode(z[..., d:], t_raw)
    return torch.cat([pos, vel], dim=-1)


def load_ae(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> CausalAE:
    ae = CausalAE.load(args.ae_ckpt_dir)
    ae = ae.to(device=device, dtype=dtype).eval()
    for param in ae.parameters():
        param.requires_grad_(False)
    return ae


def run_ae_check(args: argparse.Namespace, batch: dict[str, Any],
                 out_dir: str, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    ae = load_ae(args, device, dtype)
    batch_d = move_batch_to_device(batch, device, dtype)
    x_raw = batch_d["x_s_raw"]
    with torch.no_grad():
        z = encode_state(ae, x_raw)
        recon = decode_state(ae, z, x_raw.shape[1])
    if args.save_npz_snapshots:
        np.savez_compressed(
            os.path.join(out_dir, "ae_reconstruction_snapshot.npz"),
            gt=x_raw.float().cpu().numpy(),
            recon=recon.float().cpu().numpy(),
            point_obj_idx=batch_d["point_obj_idx"].cpu().numpy(),
        )
    if args.save_visualizations:
        save_visualization_pair(
            out_dir, "ae_reconstruction", x_raw, recon, batch_d["point_obj_idx"],
            args.vis_fps,
        )
    metrics = compute_position_metrics("ae_reconstruction", recon, x_raw, batch_d, args)
    metrics["ae_reconstruction"].update({
        "latent_shape": list(z.shape),
        "ae_class": ae.__class__.__name__,
        "d_latent": int(ae.d_latent),
    })
    return metrics


def axis_index(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[axis]


def guess_coordinate_axis(batch: dict[str, Any]) -> dict[str, Any]:
    x = batch["x_s_raw"].float()
    floor = batch["c_floor"].float().view(-1)
    point_mask, frame_mask = get_masks(batch, x.device)
    pos = x[..., :3]
    scores = {}
    for i, name in enumerate(["x", "y", "z"]):
        vals = []
        for b in range(pos.shape[0]):
            mask = frame_mask[b].unsqueeze(-1) & point_mask[b].unsqueeze(0)
            axis_vals = pos[b, :, :, i][mask]
            if axis_vals.numel() == 0:
                continue
            q01 = torch.quantile(axis_vals, 0.01)
            min_dist = torch.abs(axis_vals.min() - floor[b])
            q01_dist = torch.abs(q01 - floor[b])
            span = axis_vals.max() - axis_vals.min()
            vals.append(float((min_dist + q01_dist + 0.01 * span.abs()).item()))
        scores[name] = float(np.mean(vals)) if vals else math.inf
    guessed = min(scores, key=scores.get)
    return {
        "axis_guess": guessed,
        "floor_axis_scores_lower_is_better": scores,
        "basis": "axis whose min/lower-percentile is closest to c_floor",
    }


def run_baselines(args: argparse.Namespace, batch: dict[str, Any], out_dir: str) -> dict[str, Any]:
    x = batch["x_s_raw"].float()
    B, T, N, C = x.shape
    pos0 = x[:, :1, :, :3]
    t = torch.arange(T, device=x.device, dtype=x.dtype).view(1, T, 1, 1) * args.dt
    baselines: dict[str, torch.Tensor] = {}

    const_pos = x.clone()
    const_pos[..., :3] = pos0.expand(-1, T, -1, -1)
    const_pos[..., 3:6] = 0.0
    baselines["constant_position"] = const_pos

    finite_vel = torch.zeros(B, 1, N, 3, dtype=x.dtype, device=x.device)
    if T > 1:
        finite_vel = (x[:, 1:2, :, :3] - x[:, 0:1, :, :3]) / max(args.dt, 1e-8)
    const_vel = x.clone()
    const_vel[..., :3] = pos0 + finite_vel * t
    const_vel[..., 3:6] = finite_vel.expand(-1, T, -1, -1)
    baselines["constant_velocity_from_position_delta"] = const_vel

    gt_v0 = x[:, :1, :, 3:6]
    gt_init_vel = x.clone()
    gt_init_vel[..., :3] = pos0 + gt_v0 * t
    gt_init_vel[..., 3:6] = gt_v0.expand(-1, T, -1, -1)
    baselines["gt_initial_velocity_extrapolation"] = gt_init_vel

    axis_guess = guess_coordinate_axis(batch)["axis_guess"] if args.gravity_axis == "auto" else args.gravity_axis
    g_axis = axis_index(axis_guess)
    gravity = x.clone()
    gravity[..., :3] = pos0.expand(-1, T, -1, -1)
    gravity[..., 3:6] = 0.0
    gravity[..., g_axis] = pos0[..., g_axis] + 0.5 * args.gravity * t.squeeze(-1)
    gravity[..., 3 + g_axis] = args.gravity * t.squeeze(-1)
    baselines["gravity_only_extrapolation"] = gravity

    metrics = {"baselines": {"gravity_axis": axis_guess, "dt": args.dt}}
    for name, pred in baselines.items():
        metrics["baselines"][name] = compute_position_metrics(name, pred, x, batch, args)[name]

    if args.save_npz_snapshots:
        np.savez_compressed(
            os.path.join(out_dir, "naive_baselines_snapshot.npz"),
            gt=x.cpu().numpy(),
            point_obj_idx=batch["point_obj_idx"].cpu().numpy(),
            **{name: pred.cpu().numpy() for name, pred in baselines.items()},
        )
    if args.save_visualizations:
        first_name = "constant_position"
        save_visualization_pair(
            out_dir, f"baseline_{first_name}", x, baselines[first_name],
            batch["point_obj_idx"], args.vis_fps,
        )
    return metrics


def tensor_summary(t: torch.Tensor) -> dict[str, Any]:
    if t.numel() == 0:
        return {"shape": list(t.shape), "dtype": str(t.dtype), "numel": 0}
    tf = t.float() if not torch.is_floating_point(t) else t.float()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "min": float(tf.min().item()),
        "max": float(tf.max().item()),
        "mean": float(tf.mean().item()),
        "std": float(tf.std(unbiased=False).item()) if tf.numel() > 1 else 0.0,
    }


def build_audit(args: argparse.Namespace, batch: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    x = batch["x_s_raw"].float()
    point_mask, frame_mask = get_masks(batch, x.device)
    audit: dict[str, Any] = {
        "dataset": {
            "dataset_type": args.dataset_type,
            "data_root": args.data_root,
            "ann_path": args.ann_path,
            "dataset_len": batch.get("_dataset_len"),
            "sample_indices": batch.get("_sample_indices"),
        },
        "tensors": {},
        "masks": {},
        "objects": {},
        "coordinates": {},
    }
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            audit["tensors"][key] = tensor_summary(value)

    audit["coordinates"]["position_channels"] = {
        "x": tensor_summary(x[..., 0]),
        "y": tensor_summary(x[..., 1]),
        "z": tensor_summary(x[..., 2]),
    }
    audit["coordinates"]["velocity_channels"] = {
        "vx": tensor_summary(x[..., 3]),
        "vy": tensor_summary(x[..., 4]),
        "vz": tensor_summary(x[..., 5]),
    }
    audit["coordinates"]["coordinate_axis_guess"] = guess_coordinate_axis(batch)
    audit["coordinates"]["floor_height"] = tensor_summary(batch["c_floor"])

    audit["masks"]["point_mask_present"] = "point_mask" in batch
    audit["masks"]["valid_seq_mask_interpretation"] = (
        "valid sequence tokens are latent-frame validity crossed with point_mask; "
        "raw frame validity uses T_raw in padded batches"
    )
    audit["masks"]["valid_points_per_sample"] = point_mask.sum(dim=1).cpu().tolist()
    audit["masks"]["valid_frames_per_sample"] = frame_mask.sum(dim=1).cpu().tolist()
    if "obj_mask" in batch:
        audit["masks"]["obj_mask_present"] = True
        audit["masks"]["valid_objects_per_sample"] = batch["obj_mask"].bool().sum(dim=1).cpu().tolist()
    else:
        audit["masks"]["obj_mask_present"] = False

    point_obj_idx = batch["point_obj_idx"].long()
    audit["objects"]["point_obj_idx_min"] = int(point_obj_idx.min().item())
    audit["objects"]["point_obj_idx_max"] = int(point_obj_idx.max().item())
    per_sample = []
    for b in range(point_obj_idx.shape[0]):
        ids = torch.unique(point_obj_idx[b][point_mask[b]])
        counts = {}
        consistent = True
        n_objects = int(batch["n_objects"][b].item()) if isinstance(batch.get("n_objects"), torch.Tensor) else int(batch["n_objects"][b])
        for obj_id in ids.tolist():
            count = int(((point_obj_idx[b] == obj_id) & point_mask[b]).sum().item())
            counts[str(int(obj_id))] = count
            if obj_id < 0 or obj_id >= n_objects:
                consistent = False
        per_sample.append({
            "sample": b,
            "n_objects": n_objects,
            "object_ids_present": [int(i) for i in ids.tolist()],
            "valid_points_per_object": counts,
            "point_obj_idx_consistent_with_n_objects": consistent,
        })
    audit["objects"]["per_sample"] = per_sample

    contact = contact_frame_mask(batch, args.contact_threshold, frame_mask)
    audit["coordinates"]["contact_labels_available_from_c_force_raw"] = bool(contact is not None)
    if contact is not None:
        audit["coordinates"]["contact_frames_per_sample"] = contact.sum(dim=1).cpu().tolist()

    lines = []
    lines.append("Phase 0 Data Loader / Coordinate / Mask Audit")
    lines.append(f"dataset_type: {args.dataset_type}")
    lines.append(f"data_root: {args.data_root}")
    lines.append(f"sample_indices: {audit['dataset']['sample_indices']}")
    lines.append("")
    for key, value in audit["tensors"].items():
        lines.append(f"{key}: {value}")
    lines.append("")
    lines.append(f"point_mask present: {audit['masks']['point_mask_present']}")
    lines.append(f"valid points per sample: {audit['masks']['valid_points_per_sample']}")
    lines.append(f"valid frames per sample: {audit['masks']['valid_frames_per_sample']}")
    lines.append(f"point_obj_idx range: {audit['objects']['point_obj_idx_min']}..{audit['objects']['point_obj_idx_max']}")
    lines.append(f"coordinate axis guess: {audit['coordinates']['coordinate_axis_guess']}")
    for sample_info in per_sample:
        lines.append(f"sample {sample_info['sample']} objects: {sample_info}")
    return "\n".join(lines) + "\n", audit


def run_audit(args: argparse.Namespace, batch: dict[str, Any], out_dir: str) -> dict[str, Any]:
    audit_txt, audit = build_audit(args, batch)
    with open(os.path.join(out_dir, "audit.txt"), "w") as f:
        f.write(audit_txt)
    print(audit_txt)
    return {"audit": audit}


def ensure_audit_file(args: argparse.Namespace, batch: dict[str, Any], out_dir: str) -> None:
    audit_path = os.path.join(out_dir, "audit.txt")
    if os.path.exists(audit_path):
        return
    audit_txt, _ = build_audit(args, batch)
    with open(audit_path, "w") as f:
        f.write(audit_txt)


def compute_point_anchor(x_s_init: torch.Tensor, point_obj_idx: torch.Tensor) -> torch.Tensor:
    init_pos = x_s_init[..., :3]
    B, _, N, _ = init_pos.shape
    anchor = torch.zeros_like(init_pos)
    for b in range(B):
        obj_ids = torch.unique(point_obj_idx[b])
        for obj_id in obj_ids.tolist():
            obj_mask = point_obj_idx[b] == obj_id
            if not torch.any(obj_mask):
                continue
            obj_pos = init_pos[b, 0, obj_mask]
            anchor[b, 0, obj_mask] = obj_pos - obj_pos.mean(dim=0, keepdim=True)
    return anchor


def build_valid_seq_mask(batch: dict[str, Any], T: int, device: torch.device) -> torch.Tensor | None:
    if "point_mask" not in batch:
        return None
    point_mask = batch["point_mask"].to(device=device, dtype=torch.bool)
    T_raw_tensor = batch["T_raw"].to(device=device)
    t_state = (T_raw_tensor - 1) // 4 + 1
    t_idx = torch.arange(T, device=device).unsqueeze(0)
    t_valid = t_idx < t_state.unsqueeze(1)
    return (t_valid.unsqueeze(2) & point_mask.unsqueeze(1)).view(point_mask.shape[0], T * point_mask.shape[1])


def build_stage1_modules(args: argparse.Namespace, ae: CausalAE,
                         device: torch.device, dtype: torch.dtype):
    d_state = args.d_state or (2 * ae.d_latent)
    sim_cond = SimConditionEmbedder(
        max_objects=args.max_objects,
        d_force=2 * ae.d_latent,
    )
    sim = SimTransformer(
        d_state=d_state,
        d_cond=sim_cond.d_cond,
        d_anchor=0 if args.disable_point_anchor else 3,
        d_sim=args.d_sim,
        ffn_dim=args.sim_ffn_dim,
        num_heads=args.sim_num_heads,
        num_layers=args.sim_num_layers,
        use_factorized_attention=args.use_factorized_attention,
        use_temporal_correspondence=args.use_temporal_correspondence,
        use_temporal_rope=args.use_temporal_rope,
        use_object_local_attention=args.use_object_local_attention,
        input_representation="latent",
    )
    return sim.to(device=device, dtype=dtype), sim_cond.to(device=device, dtype=dtype)


def stage1_overfit_step(
    args: argparse.Namespace,
    ae: CausalAE,
    sim: SimTransformer,
    sim_cond: SimConditionEmbedder,
    batch: dict[str, Any],
    scheduler: FlowMatchEulerDiscreteScheduler,
    noise: torch.Tensor | None,
    timestep_index: int | None,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, torch.Tensor]:
    x_s_raw = batch["x_s_raw"]
    c_force_raw = batch["c_force_raw"]
    point_obj_idx = batch["point_obj_idx"]
    point_mask = batch.get("point_mask")

    with torch.no_grad():
        x_s_enc = encode_state(ae, x_s_raw)
        c_force_enc = encode_state(ae, c_force_raw)
    B, T, N, d_state = x_s_enc.shape
    if noise is None:
        noise = torch.randn_like(x_s_enc)

    if timestep_index is None:
        indices = torch.randint(
            low=0,
            high=scheduler.config.num_train_timesteps,
            size=(B,),
            device=x_s_enc.device,
        )
    else:
        idx = max(0, min(int(timestep_index), scheduler.config.num_train_timesteps - 1))
        indices = torch.full((B,), idx, device=x_s_enc.device, dtype=torch.long)

    timesteps = scheduler.timesteps[indices.detach().cpu()].to(device=x_s_enc.device)
    sigmas = scheduler.sigmas.to(device=x_s_enc.device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(x_s_enc.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while sigma.ndim < x_s_enc.ndim:
        sigma = sigma.unsqueeze(-1)

    noisy = (1.0 - sigma) * x_s_enc + sigma * noise
    noisy[:, :1] = x_s_enc[:, :1]
    target = noise - x_s_enc

    init_enc = x_s_enc[:, :1].expand(-1, T, -1, -1).contiguous()
    init_mask = torch.ones(B, T, N, 1, device=x_s_enc.device, dtype=dtype)
    init_mask[:, :1] = 0.0
    point_anchor = compute_point_anchor(x_s_raw[:, :1], point_obj_idx).to(dtype=dtype)
    if sim.d_anchor == 0:
        point_anchor = point_anchor[..., :0]
    point_anchor = point_anchor.expand(-1, T, -1, -1).contiguous()

    c_sim = sim_cond(
        c_floor=batch["c_floor"],
        c_id=batch["c_id"],
        c_mat=batch["c_mat"],
        c_mass=batch["c_mass"],
        c_static=batch["c_static"],
        c_force_enc=c_force_enc,
        point_obj_idx=point_obj_idx,
        T=T,
        point_mask=point_mask,
    )
    valid_seq_mask = build_valid_seq_mask(batch, T, x_s_enc.device)
    pred = sim(
        noisy,
        init_enc,
        init_mask,
        point_anchor,
        c_sim,
        timesteps,
        dtype=dtype,
        valid_seq_mask=valid_seq_mask,
        point_obj_idx=point_obj_idx if args.use_object_local_attention else None,
    )

    weighting = compute_loss_weighting_for_sd3(
        weighting_scheme=args.weighting_scheme,
        sigmas=sigma,
    )
    loss_per_elem = F.mse_loss(pred.float(), target.float(), reduction="none")
    if point_mask is not None:
        T_raw_tensor = batch["T_raw"].to(x_s_enc.device)
        t_state = (T_raw_tensor - 1) // 4 + 1
        t_idx = torch.arange(T, device=x_s_enc.device).unsqueeze(0)
        state_mask = (t_idx < t_state.unsqueeze(1)).unsqueeze(2) & point_mask.unsqueeze(1)
    else:
        state_mask = torch.ones(B, T, N, device=x_s_enc.device, dtype=torch.bool)
    state_mask[:, :1] = False
    n_valid = state_mask.float().sum() * d_state
    loss = (
        loss_per_elem
        * state_mask.unsqueeze(-1).float()
        * weighting.float()
    ).sum() / n_valid.clamp_min(1)
    pred_err = torch.sqrt(
        (
            (pred.float() - target.float()).square()
            * state_mask.unsqueeze(-1).float()
        ).sum() / n_valid.clamp_min(1)
    )
    diagnostics = {
        "loss": float(loss.detach().item()),
        "pred_target_rmse": float(pred_err.detach().item()),
        "timestep_mean": float(timesteps.float().mean().item()),
        "sigma_mean": float(sigma.float().mean().item()),
    }
    pred_x0 = noise - pred.detach()
    pred_x0[:, :1] = x_s_enc[:, :1]
    return loss, diagnostics, pred_x0, target.detach()


def grad_global_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        norm = param.grad.detach().float().norm(2).item()
        total += norm * norm
    return math.sqrt(total)


def param_delta_norm(before: list[torch.Tensor], parameters) -> float:
    total = 0.0
    for old, param in zip(before, parameters):
        delta = (param.detach().float().cpu() - old).norm(2).item()
        total += delta * delta
    return math.sqrt(total)


def run_overfit(args: argparse.Namespace, batch: dict[str, Any], out_dir: str,
                device: torch.device, dtype: torch.dtype, deterministic: bool) -> dict[str, Any]:
    mode_name = "deterministic_one_batch_overfit" if deterministic else "stochastic_one_batch_overfit"
    ae = load_ae(args, device, dtype)
    batch_d = move_batch_to_device(batch, device, dtype)
    sim, sim_cond = build_stage1_modules(args, ae, device, dtype)
    sim.train()
    sim_cond.train()
    params = list(sim.parameters()) + list(sim_cond.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=args.train_sampling_steps)

    with torch.no_grad():
        fixed_state = encode_state(ae, batch_d["x_s_raw"])
    fixed_noise = torch.randn_like(fixed_state) if deterministic else None
    fixed_timestep = args.fixed_timestep_index
    if fixed_timestep is None and deterministic:
        fixed_timestep = args.train_sampling_steps // 2

    history = []
    finite = True
    for step in range(1, args.overfit_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        before = [param.detach().float().cpu().clone() for param in params if param.requires_grad]
        loss, step_metrics, pred_x0, _ = stage1_overfit_step(
            args=args,
            ae=ae,
            sim=sim,
            sim_cond=sim_cond,
            batch=batch_d,
            scheduler=scheduler,
            noise=fixed_noise,
            timestep_index=fixed_timestep if deterministic else None,
            dtype=dtype,
        )
        if not torch.isfinite(loss):
            finite = False
        loss.backward()
        grad_norm = grad_global_norm(params)
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        optimizer.step()
        update_norm = param_delta_norm(before, [p for p in params if p.requires_grad])
        row = {
            "step": step,
            **step_metrics,
            "grad_norm": grad_norm,
            "param_update_norm": update_norm,
            "parameters_updating": update_norm > 0.0,
            "finite": bool(torch.isfinite(loss).item()),
        }
        history.append(row)
        if step == 1 or step % args.log_every == 0 or step == args.overfit_steps:
            print(f"{mode_name} step {step}: loss={row['loss']:.6g} grad={grad_norm:.6g} update={update_norm:.6g}")
        if (
            args.save_npz_snapshots
            and args.snapshot_every > 0
            and (step % args.snapshot_every == 0 or step == args.overfit_steps)
        ):
            with torch.no_grad():
                decoded = decode_state(ae, pred_x0, batch_d["x_s_raw"].shape[1])
            np.savez_compressed(
                os.path.join(out_dir, f"{mode_name}_step{step:06d}.npz"),
                gt=batch_d["x_s_raw"].float().cpu().numpy(),
                pred=decoded.float().cpu().numpy(),
                point_obj_idx=batch_d["point_obj_idx"].cpu().numpy(),
            )

    losses = np.asarray([row["loss"] for row in history], dtype=np.float64)
    pred_errors = np.asarray([row["pred_target_rmse"] for row in history], dtype=np.float64)
    grad_norms = np.asarray([row["grad_norm"] for row in history], dtype=np.float64)
    updates = np.asarray([row["param_update_norm"] for row in history], dtype=np.float64)
    metrics = {
        mode_name: {
            "history": history,
            "initial_loss": float(losses[0]),
            "final_loss": float(losses[-1]),
            "loss_reduction_ratio": float(losses[-1] / max(losses[0], 1e-12)),
            "min_loss": float(losses.min()),
            "loss_std": float(losses.std()),
            "final_pred_target_rmse": float(pred_errors[-1]),
            "max_grad_norm": float(grad_norms.max()),
            "mean_grad_norm": float(grad_norms.mean()),
            "max_param_update_norm": float(updates.max()),
            "parameters_updated_any_step": bool(np.any(updates > 0.0)),
            "all_losses_finite": bool(finite and np.all(np.isfinite(losses))),
        }
    }
    if not deterministic:
        metrics[mode_name]["stability"] = {
            "loss_is_finite": bool(np.all(np.isfinite(losses))),
            "loss_max_to_min_ratio": float(losses.max() / max(losses.min(), 1e-12)),
            "nan_or_inf_steps": int(np.sum(~np.isfinite(losses))),
        }
    return metrics


def save_visualization_pair(out_dir: str, name: str, gt: torch.Tensor, pred: torch.Tensor,
                            point_obj_idx: torch.Tensor, fps: int) -> None:
    try:
        from visualize import visualize_point_cloud_motion
    except Exception as exc:
        print(f"Skipping visualization import for {name}: {exc}")
        return
    gt_np = gt[0].float().cpu().numpy()
    pred_np = pred[0].float().cpu().numpy()
    obj_np = point_obj_idx[0].long().cpu().numpy()
    visualize_point_cloud_motion(gt_np, obj_np, os.path.join(out_dir, f"{name}_gt.mp4"), fps=fps)
    visualize_point_cloud_motion(pred_np, obj_np, os.path.join(out_dir, f"{name}_pred.mp4"), fps=fps)


def main() -> None:
    args = parse_args()
    set_reproducible_seed(args.seed)
    out_dir = make_output_dir(args)
    save_config(args, out_dir)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    print(f"Phase 0 output_dir: {out_dir}")
    print(f"device={device} dtype={dtype} modes={args.modes}")

    batch = build_batch(args)
    metrics: dict[str, Any] = {
        "output_dir": out_dir,
        "modes": args.modes,
        "device": str(device),
        "dtype": str(dtype),
    }

    if "audit" in args.modes:
        metrics.update(run_audit(args, batch, out_dir))
    else:
        ensure_audit_file(args, batch, out_dir)
    if "ae" in args.modes:
        metrics.update(run_ae_check(args, batch, out_dir, device, dtype))
    if "baselines" in args.modes:
        metrics.update(run_baselines(args, batch, out_dir))
    if "deterministic_overfit" in args.modes:
        metrics.update(run_overfit(args, batch, out_dir, device, dtype, deterministic=True))
    if "stochastic_overfit" in args.modes:
        metrics.update(run_overfit(args, batch, out_dir, device, dtype, deterministic=False))

    save_metrics(metrics, out_dir)
    print(f"Saved metrics.json, metrics.csv, audit.txt/config.json in {out_dir}")


if __name__ == "__main__":
    main()

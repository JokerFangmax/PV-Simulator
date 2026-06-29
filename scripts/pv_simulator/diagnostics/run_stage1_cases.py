"""Run six representative Stage 1 diagnostic cases.

The goal is to evaluate the current Simulation Branch before deciding whether
to add Contact-Aware Sampling, Rigid-Residual Representation, Data Augmentation,
Violation Feedback, or Stage 2 demos. This script does not add or modify model
modules.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from typing import Any

import numpy as np
import torch

CURRENT_FILE = os.path.abspath(__file__)
DIAG_DIR = os.path.dirname(CURRENT_FILE)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE))))
for _root in [DIAG_DIR, REPO_ROOT]:
    if _root not in sys.path:
        sys.path.insert(0, _root)

from audit_dataset_distribution import (  # noqa: E402
    CASE_NAMES,
    HEURISTIC_LIMITATIONS,
    collate_one,
    compute_sample_features,
    summarize_distribution,
)
from run_phase0_sanity import (  # noqa: E402
    axis_index,
    build_dataset,
    decode_state,
    encode_state,
    expand_contact_window,
    get_masks,
    guess_coordinate_axis,
    load_ae,
    local_geometry_metrics,
    masked_position_rmse,
    move_batch_to_device,
    sampled_chamfer,
    save_config,
    scalarize,
    set_reproducible_seed,
    tensor_to_jsonable,
    valid_position_stats,
)
from videox_fun.pipeline.pipeline_simulation import SimulationPipeline  # noqa: E402


CASE_TITLES = {
    "free_fall": "Free Fall",
    "vertical_bounce": "Vertical Bounce",
    "oblique_impact": "Oblique Impact",
    "rolling_sliding": "Rolling / Sliding",
    "multi_object_floor": "Multi-Object Floor",
    "collision_heavy": "Collision-Heavy / Contact-Dense",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 1 six-case diagnostics.")
    parser.add_argument("--dataset_type", type=str, default="movi", choices=["movi", "simulation"])
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--ann_path", type=str, default=None)
    parser.add_argument("--output_root", type=str, default="experiments/stage1_diagnostics")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--max_scan_samples", type=int, default=512)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--case_sample_idx", action="append", default=[],
                        help="Override case selection, e.g. free_fall:12. Can be repeated.")
    parser.add_argument("--cases", nargs="+", default=CASE_NAMES, choices=CASE_NAMES + ["all"])

    parser.add_argument("--padded_batch", action="store_true")
    parser.add_argument("--max_T_raw", type=int, default=21)
    parser.add_argument("--max_objects", type=int, default=5)
    parser.add_argument("--max_points_per_object", type=int, default=200)
    parser.add_argument("--center_on_contact", action="store_true")

    parser.add_argument("--ae_ckpt_dir", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="Stage 1 checkpoint directory. Required unless --skip_stage1_pred is set.")
    parser.add_argument("--skip_stage1_pred", action="store_true",
                        help="Do not run SimDiT prediction; useful for dataset/metric smoke tests.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=50)

    parser.add_argument("--dt", type=float, default=1.0 / 12.0)
    parser.add_argument("--gravity", type=float, default=-9.81)
    parser.add_argument("--gravity_axis", type=str, default="auto", choices=["auto", "x", "y", "z"])
    parser.add_argument("--max_chamfer_points", type=int, default=512)
    parser.add_argument("--knn_k", type=int, default=8)
    parser.add_argument("--collapse_ratio_threshold", type=float, default=0.2)
    parser.add_argument("--contact_threshold", type=float, default=1e-8)
    parser.add_argument("--contact_window_radius", type=int, default=2)

    parser.add_argument("--floor_eps", type=float, default=0.08)
    parser.add_argument("--contact_ratio_heavy", type=float, default=0.25)
    parser.add_argument("--horizontal_speed_threshold", type=float, default=0.25)
    parser.add_argument("--horizontal_displacement_threshold", type=float, default=0.25)
    parser.add_argument("--vertical_motion_threshold", type=float, default=0.2)
    parser.add_argument("--bounce_velocity_threshold", type=float, default=0.05)

    parser.add_argument("--save_visualizations", action="store_true")
    parser.add_argument("--vis_fps", type=int, default=10)
    parser.add_argument("--vis_max_points_per_object", type=int, default=120)
    args = parser.parse_args()
    if "all" in args.cases:
        args.cases = CASE_NAMES
    args.cases = sorted(set(args.cases), key=CASE_NAMES.index)
    if args.dataset_type == "simulation" and args.ann_path is None:
        parser.error("--ann_path is required when --dataset_type simulation")
    if not args.skip_stage1_pred and args.ckpt_dir is None:
        parser.error("--ckpt_dir is required unless --skip_stage1_pred is set")
    if args.max_scan_samples < 1:
        parser.error("--max_scan_samples must be >= 1")
    if args.stride < 1:
        parser.error("--stride must be >= 1")
    return args


def make_output_dir(args: argparse.Namespace) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{stamp}_six_cases"
    out_dir = os.path.join(args.output_root, run_name)
    os.makedirs(out_dir, exist_ok=False)
    return out_dir


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "bf16" and device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def parse_overrides(values: list[str]) -> dict[str, int]:
    overrides = {}
    for value in values:
        if ":" not in value:
            raise ValueError(f"Invalid --case_sample_idx {value!r}; expected case:idx")
        case, idx = value.split(":", 1)
        if case not in CASE_NAMES:
            raise ValueError(f"Unknown case {case!r}; expected one of {CASE_NAMES}")
        overrides[case] = int(idx)
    return overrides


def select_case_samples(dataset, args: argparse.Namespace) -> tuple[dict[str, int], list[dict[str, Any]]]:
    overrides = parse_overrides(args.case_sample_idx)
    selected = {case: overrides[case] for case in args.cases if case in overrides}
    scanned_rows: list[dict[str, Any]] = []
    max_idx = min(len(dataset), args.sample_start + args.max_scan_samples * args.stride)
    for sample_idx in range(args.sample_start, max_idx, args.stride):
        if all(case in selected for case in args.cases):
            break
        sample = dataset[sample_idx]
        batch = collate_one(sample, args)
        batch["_sample_indices"] = [sample_idx]
        features = compute_sample_features(batch, args)
        scanned_rows.append(features)
        for case in args.cases:
            if case not in selected and features["cases"].get(case, False):
                selected[case] = sample_idx
    return selected, scanned_rows


def _contact_mask_from_heuristics(batch: dict[str, Any], args: argparse.Namespace,
                                  axis: str) -> torch.Tensor:
    x = batch["x_s_raw"].float()
    point_mask, frame_mask = get_masks(batch, x.device)
    g_axis = axis_index(axis)
    floor = batch["c_floor"].float().view(-1)
    min_h = x[..., g_axis].masked_fill(~point_mask.unsqueeze(1), float("inf")).min(dim=2).values
    near_floor = min_h <= floor.unsqueeze(1) + args.floor_eps
    c_force_raw = batch.get("c_force_raw")
    force_contact = torch.zeros_like(frame_mask)
    if isinstance(c_force_raw, torch.Tensor):
        force_norm = torch.linalg.vector_norm(c_force_raw[..., :3].float(), dim=-1)
        contact_norm = torch.linalg.vector_norm(c_force_raw[..., 3:6].float(), dim=-1)
        force_contact = ((force_norm > args.contact_threshold) | (contact_norm > args.contact_threshold)).any(dim=-1)
    return (near_floor | force_contact) & frame_mask


def make_naive_baseline(batch: dict[str, Any], args: argparse.Namespace,
                        case: str) -> tuple[str, torch.Tensor]:
    x = batch["x_s_raw"].float()
    B, T, _, _ = x.shape
    pos0 = x[:, :1, :, :3]
    t = torch.arange(T, device=x.device, dtype=x.dtype).view(1, T, 1, 1) * args.dt
    finite_vel = torch.zeros_like(x[:, :1, :, :3])
    if T > 1:
        finite_vel = (x[:, 1:2, :, :3] - x[:, 0:1, :, :3]) / max(args.dt, 1e-8)
    gt_v0 = x[:, :1, :, 3:6]

    axis_guess = guess_coordinate_axis(batch)["axis_guess"] if args.gravity_axis == "auto" else args.gravity_axis
    g_axis = axis_index(axis_guess)
    baseline = x.clone()
    name = "constant_velocity_from_position_delta"
    baseline[..., :3] = pos0 + finite_vel * t
    baseline[..., 3:6] = finite_vel.expand(-1, T, -1, -1)

    if case == "free_fall":
        name = "gravity_only_extrapolation"
        baseline[..., :3] = pos0 + gt_v0 * t
        baseline[..., g_axis] = (
            pos0[..., g_axis]
            + gt_v0[..., g_axis] * t.squeeze(-1)
            + 0.5 * args.gravity * t.squeeze(-1).square()
        )
        baseline[..., 3:6] = gt_v0.expand(-1, T, -1, -1)
        baseline[..., 3 + g_axis] = gt_v0[..., g_axis] + args.gravity * t.squeeze(-1)
    elif case in {"vertical_bounce", "collision_heavy", "multi_object_floor"}:
        name = "constant_position"
        baseline[..., :3] = pos0.expand(-1, T, -1, -1)
        baseline[..., 3:6] = 0.0
    elif case in {"oblique_impact", "rolling_sliding"}:
        name = "gt_initial_velocity_extrapolation"
        baseline[..., :3] = pos0 + gt_v0 * t
        baseline[..., 3:6] = gt_v0.expand(-1, T, -1, -1)
    return name, baseline


def _masked_rmse(pred: torch.Tensor, gt: torch.Tensor, mask_bt: torch.Tensor,
                 point_mask: torch.Tensor, channels: slice) -> float:
    pred_c = pred[..., channels]
    gt_c = gt[..., channels]
    mask = mask_bt.unsqueeze(-1) & point_mask.unsqueeze(1)
    sq = (pred_c - gt_c).square().sum(dim=-1)
    denom = (mask.float().sum() * pred_c.shape[-1]).clamp_min(1)
    return float(torch.sqrt((sq * mask.float()).sum() / denom).item())


def _centroid_error(pred: torch.Tensor, gt: torch.Tensor, point_obj_idx: torch.Tensor,
                    point_mask: torch.Tensor, frame_mask: torch.Tensor) -> float:
    total = pred.new_tensor(0.0)
    count = pred.new_tensor(0.0)
    for b in range(pred.shape[0]):
        frames = frame_mask[b].nonzero(as_tuple=True)[0]
        obj_ids = torch.unique(point_obj_idx[b][point_mask[b]])
        for obj_id in obj_ids.tolist():
            obj_mask = (point_obj_idx[b] == obj_id) & point_mask[b]
            if not obj_mask.any():
                continue
            pred_c = pred[b, frames][:, obj_mask, :3].mean(dim=1)
            gt_c = gt[b, frames][:, obj_mask, :3].mean(dim=1)
            total = total + torch.linalg.vector_norm(pred_c - gt_c, dim=-1).sum()
            count = count + frames.numel()
    return float((total / count.clamp_min(1)).item())


def _floor_metrics(pred: torch.Tensor, batch: dict[str, Any], axis: str,
                   point_mask: torch.Tensor, frame_mask: torch.Tensor) -> dict[str, float]:
    g_axis = axis_index(axis)
    floor = batch["c_floor"].to(pred.device).float().view(-1, 1, 1)
    depth = (floor - pred[..., g_axis].float()).clamp_min(0.0)
    mask = frame_mask.unsqueeze(-1) & point_mask.unsqueeze(1)
    valid_depth = depth[mask]
    if valid_depth.numel() == 0:
        return {"floor_penetration_depth_mean": 0.0, "floor_penetration_depth_max": 0.0, "floor_penetration_rate": 0.0}
    return {
        "floor_penetration_depth_mean": float(valid_depth.mean().item()),
        "floor_penetration_depth_max": float(valid_depth.max().item()),
        "floor_penetration_rate": float((valid_depth > 0).float().mean().item()),
    }


def compute_eval_metrics(name: str, pred: torch.Tensor, gt: torch.Tensor,
                         batch: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    point_mask, frame_mask = get_masks(batch, pred.device)
    axis = guess_coordinate_axis(batch)["axis_guess"]
    pos_stats = valid_position_stats(gt[..., :3], point_mask, frame_mask)
    vel_stats = valid_position_stats(gt[..., 3:6], point_mask, frame_mask)
    pos_rmse = masked_position_rmse(pred[..., :3], gt[..., :3], point_mask, frame_mask)
    vel_rmse = torch.tensor(_masked_rmse(pred, gt, frame_mask, point_mask, slice(3, 6)), device=pred.device)
    contact = _contact_mask_from_heuristics(batch, args, axis)
    contact_window = expand_contact_window(contact, args.contact_window_radius)
    global_rmse = float(pos_rmse.item())
    contact_rmse = _masked_rmse(pred, gt, contact_window, point_mask, slice(0, 3)) if contact_window.any() else None
    contact_ratio = (contact_rmse / max(global_rmse, 1e-8)) if contact_rmse is not None else None
    metrics = {
        "normalized_position_rmse": float((pos_rmse / pos_stats["bbox_diag"]).item()),
        "normalized_velocity_rmse": float((vel_rmse / vel_stats["bbox_diag"]).item()),
        "position_rmse": global_rmse,
        "velocity_rmse": float(vel_rmse.item()),
        "chamfer": float(sampled_chamfer(pred[..., :3], gt[..., :3], point_mask, frame_mask, args.max_chamfer_points).item()),
        "corresponding_point_l2": _masked_rmse(pred, gt, frame_mask, point_mask, slice(0, 3)),
        "centroid_error": _centroid_error(pred, gt, batch["point_obj_idx"].to(pred.device), point_mask, frame_mask),
        "contact_window_rmse": contact_rmse,
        "contact_window_rmse_over_global_rmse": contact_ratio,
        "contact_velocity_inconsistency": _masked_rmse(pred, gt, contact_window, point_mask, slice(3, 6)) if contact_window.any() else None,
        "contact_frames": int(contact.sum().item()),
        "contact_window_frames": int(contact_window.sum().item()),
        "projection_continuity_preview": "unavailable: no camera/projection metadata exposed by current Stage 1 sample dict",
    }
    metrics.update(local_geometry_metrics(
        pred_pos=pred[..., :3],
        gt_pos=gt[..., :3],
        point_obj_idx=batch["point_obj_idx"].to(pred.device),
        point_mask=point_mask,
        frame_mask=frame_mask,
        k=args.knn_k,
        collapse_threshold=args.collapse_ratio_threshold,
    ))
    metrics.update(_floor_metrics(pred, batch, axis, point_mask, frame_mask))
    return {name: metrics}


def save_metrics_files(metrics: dict[str, Any], out_dir: str) -> None:
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(tensor_to_jsonable(metrics), f, indent=2, sort_keys=True)
    rows: list[dict[str, Any]] = []
    scalarize("", tensor_to_jsonable(metrics), rows)
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def save_case_visuals(case_dir: str, snapshot_path: str, args: argparse.Namespace) -> None:
    if not args.save_visualizations:
        return
    os.makedirs("/tmp/matplotlib", exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    from scripts.pv_simulator.visualize import visualize_point_cloud_motion

    def _normalize_traj(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        return arr[0] if arr.ndim == 4 else arr

    data = np.load(snapshot_path)
    point_obj_idx = data["point_obj_idx"]
    if point_obj_idx.ndim == 2:
        point_obj_idx = point_obj_idx[0]
    gt = _normalize_traj(data["gt"])
    vis_dir = os.path.join(case_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    for key in ["gt", "ae_recon", "simdit_pred", "naive_baseline"]:
        if key not in data:
            continue
        traj = _normalize_traj(data[key])
        visualize_point_cloud_motion(
            point_states=traj,
            point_obj_idx=point_obj_idx,
            output_path=os.path.join(vis_dir, f"{key}.mp4"),
            fps=args.vis_fps,
            views=["birdseye", "side", "iso"],
            max_points_per_object=args.vis_max_points_per_object,
            reference_point_states=gt if key != "gt" else None,
            auto_select_keypoints=8 if key != "gt" else 0,
        )


def load_pipeline(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    if args.skip_stage1_pred:
        return None
    return SimulationPipeline.from_pretrained(
        ckpt_dir=args.ckpt_dir,
        ae_ckpt_dir=args.ae_ckpt_dir,
        device=str(device),
        dtype=dtype,
        max_objects=args.max_objects,
    )


def run_case(case: str, sample_idx: int, dataset, args: argparse.Namespace,
             out_dir: str, device: torch.device, dtype: torch.dtype,
             pipeline, ae) -> dict[str, Any]:
    sample = dataset[sample_idx]
    batch = collate_one(sample, args)
    batch["_sample_indices"] = [sample_idx]
    features = compute_sample_features(batch, args)
    batch_d = move_batch_to_device(batch, device, dtype)
    gt = batch_d["x_s_raw"]
    case_dir = os.path.join(out_dir, case)
    os.makedirs(case_dir, exist_ok=True)

    with torch.no_grad():
        z = encode_state(ae, gt)
        ae_recon = decode_state(ae, z, gt.shape[1])

    baseline_name, naive = make_naive_baseline(batch_d, args, case)
    simdit_pred = None
    if pipeline is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed + int(sample_idx))
        T_latent = (gt.shape[1] - 1) // 4 + 1
        point_mask = batch_d.get("point_mask")
        result = pipeline(
            c_floor=batch_d["c_floor"],
            c_id=batch_d["c_id"],
            c_mat=batch_d["c_mat"],
            c_mass=batch_d["c_mass"],
            c_static=batch_d["c_static"],
            c_force_raw=batch_d["c_force_raw"],
            x_s_init=gt[:, :1],
            point_obj_idx=batch_d["point_obj_idx"],
            T=T_latent,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            point_mask=point_mask,
            x_s_target=gt,
            show_progress=False,
        )
        simdit_pred = result["x_s"].to(device=device, dtype=torch.float32)

    snapshot = {
        "gt": gt.float().cpu().numpy(),
        "ae_recon": ae_recon.float().cpu().numpy(),
        "naive_baseline": naive.float().cpu().numpy(),
        "point_obj_idx": batch_d["point_obj_idx"].cpu().numpy(),
    }
    if simdit_pred is not None:
        snapshot["simdit_pred"] = simdit_pred.float().cpu().numpy()
    np.save(os.path.join(case_dir, "gt.npy"), snapshot["gt"])
    np.save(os.path.join(case_dir, "ae_recon.npy"), snapshot["ae_recon"])
    np.save(os.path.join(case_dir, "naive_baseline.npy"), snapshot["naive_baseline"])
    if "simdit_pred" in snapshot:
        np.save(os.path.join(case_dir, "simdit_pred.npy"), snapshot["simdit_pred"])
    snapshot_path = os.path.join(case_dir, "case_snapshot.npz")
    np.savez_compressed(snapshot_path, **snapshot)

    metrics: dict[str, Any] = {
        "case": case,
        "case_title": CASE_TITLES[case],
        "sample_idx": sample_idx,
        "heuristic_features": features,
        "heuristic_limitations": HEURISTIC_LIMITATIONS,
        "naive_baseline_name": baseline_name,
        "ae_reconstruction": compute_eval_metrics("ae_reconstruction", ae_recon.float(), gt.float(), batch_d, args)["ae_reconstruction"],
        "naive_baseline": compute_eval_metrics("naive_baseline", naive.float(), gt.float(), batch_d, args)["naive_baseline"],
        "simdit_prediction": None,
    }
    if simdit_pred is not None:
        metrics["simdit_prediction"] = compute_eval_metrics("simdit_prediction", simdit_pred.float(), gt.float(), batch_d, args)["simdit_prediction"]
    save_metrics_files(metrics, case_dir)
    save_case_visuals(case_dir, snapshot_path, args)
    return metrics


def write_readme(out_dir: str, summary: dict[str, Any], args: argparse.Namespace) -> None:
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write("# Stage 1 Six-Case Diagnostics\n\n")
        f.write(f"{HEURISTIC_LIMITATIONS}\n\n")
        f.write("This run evaluates current Stage 1 behavior only. It does not add Contact-Aware Sampling, Rigid-Residual, PBD, XPBD, Warp, Violation Feedback, or Stage 2 code.\n\n")
        f.write("## Selected Cases\n\n")
        for case, info in summary["selected_cases"].items():
            f.write(f"- {CASE_TITLES[case]} (`{case}`): sample `{info.get('sample_idx')}`, status `{info.get('status')}`\n")
        f.write("\n## Outputs\n\n")
        f.write("Each case directory contains `gt.npy`, `ae_recon.npy`, `naive_baseline.npy`, optional `simdit_pred.npy`, `case_snapshot.npz`, `metrics.json`, and `metrics.csv`.\n")
        if args.save_visualizations:
            f.write("Visualization videos are stored under each case's `visualizations/` directory.\n")


def main() -> None:
    args = parse_args()
    set_reproducible_seed(args.seed)
    out_dir = make_output_dir(args)
    save_config(args, out_dir)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    print(f"Stage 1 six-case output_dir: {out_dir}")

    dataset = build_dataset(args)
    selected, scanned_rows = select_case_samples(dataset, args)
    distribution_summary = summarize_distribution(scanned_rows)
    with open(os.path.join(out_dir, "candidate_distribution_summary.json"), "w") as f:
        json.dump(tensor_to_jsonable(distribution_summary), f, indent=2, sort_keys=True)
    with open(os.path.join(out_dir, "candidate_sample_features.json"), "w") as f:
        json.dump(tensor_to_jsonable(scanned_rows), f, indent=2, sort_keys=True)

    ae_args = argparse.Namespace(ae_ckpt_dir=args.ae_ckpt_dir)
    ae = load_ae(ae_args, device, dtype)
    pipeline = load_pipeline(args, device, dtype)

    summary: dict[str, Any] = {
        "output_dir": out_dir,
        "selected_cases": {},
        "heuristic_limitations": HEURISTIC_LIMITATIONS,
        "stage1_prediction_enabled": pipeline is not None,
    }
    all_metrics = {}
    for case in args.cases:
        sample_idx = selected.get(case)
        if sample_idx is None:
            summary["selected_cases"][case] = {
                "status": "not_found",
                "sample_idx": None,
            }
            continue
        print(f"Running {case} on sample {sample_idx}")
        metrics = run_case(case, sample_idx, dataset, args, out_dir, device, dtype, pipeline, ae)
        all_metrics[case] = metrics
        summary["selected_cases"][case] = {
            "status": "completed",
            "sample_idx": sample_idx,
            "simdit_prediction_available": metrics["simdit_prediction"] is not None,
        }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(tensor_to_jsonable(summary), f, indent=2, sort_keys=True)
    with open(os.path.join(out_dir, "all_case_metrics.json"), "w") as f:
        json.dump(tensor_to_jsonable(all_metrics), f, indent=2, sort_keys=True)
    write_readme(out_dir, summary, args)
    print(f"Saved six-case diagnostics to {out_dir}")


if __name__ == "__main__":
    main()

"""Audit Stage 1 dataset distribution for six-case diagnostics.

The MOVI/custom simulation datasets do not consistently expose semantic labels
for free fall, bounce, impact, rolling, or collision density. This script uses
simple trajectory heuristics and records their limitations in the output.

No model architecture, Stage 2, PBD, XPBD, Warp, Rigid-Residual, Contact-Aware
Sampling, or Violation Feedback code is introduced here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
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

from run_phase0_sanity import (  # noqa: E402
    build_dataset,
    get_masks,
    guess_coordinate_axis,
    sim_collate_fn,
    sim_collate_fn_padded,
    tensor_to_jsonable,
)


CASE_NAMES = [
    "free_fall",
    "vertical_bounce",
    "oblique_impact",
    "rolling_sliding",
    "multi_object_floor",
    "collision_heavy",
]

HEURISTIC_LIMITATIONS = (
    "Exact scenario labels are not assumed. Contact is inferred from force/contact "
    "channels when nonzero, otherwise from points near the floor and vertical "
    "velocity sign changes near the floor. Oblique impact, rolling/sliding, and "
    "collision-heavy labels are therefore approximate and should be used for "
    "diagnostic sampling rather than final evaluation claims."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Stage 1 diagnostic-case distribution.")
    parser.add_argument("--dataset_type", type=str, default="movi", choices=["movi", "simulation"])
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--ann_path", type=str, default=None)
    parser.add_argument("--output_root", type=str, default="experiments/stage1_diagnostics")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=256)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--padded_batch", action="store_true")
    parser.add_argument("--max_T_raw", type=int, default=21)
    parser.add_argument("--max_objects", type=int, default=5)
    parser.add_argument("--max_points_per_object", type=int, default=200)
    parser.add_argument("--center_on_contact", action="store_true")
    parser.add_argument("--floor_eps", type=float, default=0.08)
    parser.add_argument("--contact_ratio_heavy", type=float, default=0.25)
    parser.add_argument("--horizontal_speed_threshold", type=float, default=0.25)
    parser.add_argument("--horizontal_displacement_threshold", type=float, default=0.25)
    parser.add_argument("--vertical_motion_threshold", type=float, default=0.2)
    parser.add_argument("--bounce_velocity_threshold", type=float, default=0.05)
    args = parser.parse_args()
    if args.dataset_type == "simulation" and args.ann_path is None:
        parser.error("--ann_path is required when --dataset_type simulation")
    if args.max_samples < 1:
        parser.error("--max_samples must be >= 1")
    if args.stride < 1:
        parser.error("--stride must be >= 1")
    return args


def make_output_dir(args: argparse.Namespace, suffix: str = "distribution_audit") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{stamp}_{suffix}"
    out_dir = os.path.join(args.output_root, run_name)
    os.makedirs(out_dir, exist_ok=False)
    return out_dir


def collate_one(sample: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.padded_batch:
        return sim_collate_fn_padded(
            [sample],
            max_T_raw=args.max_T_raw,
            max_objects=args.max_objects,
            max_points_per_object=args.max_points_per_object,
        )
    return sim_collate_fn([sample])


def _axis_index(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[axis]


def _valid_obj_counts(point_obj_idx: torch.Tensor, point_mask: torch.Tensor) -> dict[int, int]:
    counts: dict[int, int] = {}
    ids = torch.unique(point_obj_idx[point_mask])
    for obj_id in ids.tolist():
        counts[int(obj_id)] = int(((point_obj_idx == obj_id) & point_mask).sum().item())
    return counts


def compute_sample_features(batch: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Compute heuristic features for one batched sample with B=1."""
    x = batch["x_s_raw"].float()
    point_mask, frame_mask = get_masks(batch, x.device)
    pos = x[..., :3]
    vel = x[..., 3:6]
    axis_guess = guess_coordinate_axis(batch)["axis_guess"]
    g_axis = _axis_index(axis_guess)
    tangential_axes = [i for i in range(3) if i != g_axis]

    b = 0
    valid_points = point_mask[b]
    valid_frames = frame_mask[b]
    point_obj_idx = batch["point_obj_idx"][b].long()
    floor = float(batch["c_floor"][b].float().item())
    pos_b = pos[b]
    vel_b = vel[b]

    obj_counts = _valid_obj_counts(point_obj_idx, valid_points)
    n_objects = len(obj_counts)
    valid_pos = pos_b[:, valid_points]
    valid_vel = vel_b[:, valid_points]
    frame_valid_pos = valid_pos[valid_frames]
    frame_valid_vel = valid_vel[valid_frames]

    min_height = frame_valid_pos[..., g_axis].min(dim=1).values
    near_floor = min_height <= floor + args.floor_eps

    c_force_raw = batch.get("c_force_raw")
    force_contact = torch.zeros_like(valid_frames)
    force_available = False
    if isinstance(c_force_raw, torch.Tensor):
        force_norm = torch.linalg.vector_norm(c_force_raw[b, :, :, :3].float(), dim=-1)
        contact_norm = torch.linalg.vector_norm(c_force_raw[b, :, :, 3:6].float(), dim=-1)
        force_contact = ((force_norm > 1e-8) | (contact_norm > 1e-8)).any(dim=-1) & valid_frames
        force_available = bool(force_contact.any().item())

    mean_v = torch.zeros(pos_b.shape[0], 3)
    mean_v[valid_frames] = valid_vel.mean(dim=1)
    vertical_v = mean_v[:, g_axis]
    sign_change = torch.zeros_like(valid_frames)
    if valid_frames.sum() > 1:
        prev = vertical_v[:-1]
        nxt = vertical_v[1:]
        changed = (prev < -args.bounce_velocity_threshold) & (nxt > args.bounce_velocity_threshold)
        sign_change[1:] = changed
    vertical_sign_change_near_floor = sign_change & near_floor

    contact = (force_contact | near_floor | vertical_sign_change_near_floor) & valid_frames
    contact_ratio = float(contact.float().sum().item() / max(float(valid_frames.sum().item()), 1.0))
    near_floor_ratio = float((near_floor & valid_frames).float().sum().item() / max(float(valid_frames.sum().item()), 1.0))

    valid_frame_indices = valid_frames.nonzero(as_tuple=True)[0]
    first_frame = int(valid_frame_indices[0].item()) if valid_frame_indices.numel() else 0
    last_frame = int(valid_frame_indices[-1].item()) if valid_frame_indices.numel() else 0
    centroid0 = pos_b[first_frame, valid_points].mean(dim=0)
    centroid_last = pos_b[last_frame, valid_points].mean(dim=0)
    displacement = centroid_last - centroid0
    vertical_displacement = float(displacement[g_axis].item())
    horizontal_displacement = float(torch.linalg.vector_norm(displacement[tangential_axes]).item())

    horizontal_speed = torch.linalg.vector_norm(mean_v[:, tangential_axes], dim=-1)
    contact_horizontal_speed = horizontal_speed[contact].mean() if contact.any() else horizontal_speed.new_tensor(0.0)
    persistent_floor_motion = (
        near_floor_ratio > 0.4
        and horizontal_displacement > args.horizontal_displacement_threshold
        and float(horizontal_speed[valid_frames].mean().item()) > args.horizontal_speed_threshold * 0.5
    )

    cases = {
        "free_fall": (
            contact_ratio < 0.05
            and vertical_displacement < -args.vertical_motion_threshold
        ),
        "vertical_bounce": bool(vertical_sign_change_near_floor.any().item())
        and float(contact_horizontal_speed.item()) < args.horizontal_speed_threshold,
        "oblique_impact": contact_ratio > 0.0
        and float(contact_horizontal_speed.item()) >= args.horizontal_speed_threshold,
        "rolling_sliding": bool(persistent_floor_motion),
        "multi_object_floor": n_objects > 1 and near_floor_ratio > 0.0,
        "collision_heavy": contact_ratio >= args.contact_ratio_heavy,
    }

    c_mat = batch["c_mat"][b].float()
    c_mass = batch["c_mass"][b].float()
    n_obj_meta = int(batch["n_objects"][b].item()) if isinstance(batch.get("n_objects"), torch.Tensor) else n_objects
    c_mat_valid = c_mat[:n_obj_meta]
    c_mass_valid = c_mass[:n_obj_meta]
    return {
        "sample_idx": int(batch.get("_sample_indices", [0])[0]),
        "axis_guess": axis_guess,
        "n_objects": int(n_objects),
        "valid_points": int(valid_points.sum().item()),
        "valid_frames": int(valid_frames.sum().item()),
        "contact_frame_ratio": contact_ratio,
        "near_floor_frame_ratio": near_floor_ratio,
        "force_contact_available": force_available,
        "vertical_sign_change_near_floor": bool(vertical_sign_change_near_floor.any().item()),
        "horizontal_displacement": horizontal_displacement,
        "vertical_displacement": vertical_displacement,
        "contact_horizontal_speed": float(contact_horizontal_speed.item()),
        "mean_horizontal_speed": float(horizontal_speed[valid_frames].mean().item()),
        "cases": cases,
        "friction_values": c_mat_valid[:, 0].cpu().tolist() if c_mat_valid.numel() else [],
        "restitution_values": c_mat_valid[:, 1].cpu().tolist() if c_mat_valid.numel() else [],
        "mass_values": c_mass_valid.cpu().tolist() if c_mass_valid.numel() else [],
        "valid_points_per_object": obj_counts,
        "heuristic_limitations": HEURISTIC_LIMITATIONS,
    }


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _std(values: list[float]) -> float | None:
    return float(np.std(values)) if values else None


def summarize_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = max(len(rows), 1)
    case_counts = {case: sum(1 for row in rows if row["cases"][case]) for case in CASE_NAMES}
    obj_counts = Counter(int(row["n_objects"]) for row in rows)
    friction = [v for row in rows for v in row["friction_values"]]
    restitution = [v for row in rows for v in row["restitution_values"]]
    mass = [v for row in rows for v in row["mass_values"]]
    return {
        "num_samples": len(rows),
        "heuristic_limitations": HEURISTIC_LIMITATIONS,
        "case_counts": case_counts,
        "case_ratios": {case: case_counts[case] / total for case in CASE_NAMES},
        "contact_frame_ratio_mean": _mean([row["contact_frame_ratio"] for row in rows]),
        "rolling_sliding_sample_ratio": case_counts["rolling_sliding"] / total,
        "oblique_impact_sample_ratio": case_counts["oblique_impact"] / total,
        "collision_heavy_sample_ratio": case_counts["collision_heavy"] / total,
        "multi_object_sample_ratio": sum(1 for row in rows if row["n_objects"] > 1) / total,
        "force_contact_availability_ratio": sum(1 for row in rows if row["force_contact_available"]) / total,
        "object_count_distribution": dict(sorted(obj_counts.items())),
        "friction_distribution": {"mean": _mean(friction), "std": _std(friction), "min": min(friction) if friction else None, "max": max(friction) if friction else None},
        "restitution_distribution": {"mean": _mean(restitution), "std": _std(restitution), "min": min(restitution) if restitution else None, "max": max(restitution) if restitution else None},
        "mass_distribution": {"mean": _mean(mass), "std": _std(mass), "min": min(mass) if mass else None, "max": max(mass) if mass else None},
    }


def save_distribution(rows: list[dict[str, Any]], summary: dict[str, Any], out_dir: str) -> None:
    with open(os.path.join(out_dir, "distribution_summary.json"), "w") as f:
        json.dump(tensor_to_jsonable(summary), f, indent=2, sort_keys=True)
    with open(os.path.join(out_dir, "sample_features.json"), "w") as f:
        json.dump(tensor_to_jsonable(rows), f, indent=2, sort_keys=True)

    fieldnames = [
        "sample_idx",
        "n_objects",
        "valid_points",
        "valid_frames",
        "axis_guess",
        "contact_frame_ratio",
        "near_floor_frame_ratio",
        "force_contact_available",
        "horizontal_displacement",
        "vertical_displacement",
        "contact_horizontal_speed",
        "mean_horizontal_speed",
    ] + CASE_NAMES
    with open(os.path.join(out_dir, "sample_features.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in fieldnames if key not in CASE_NAMES}
            flat.update({case: row["cases"][case] for case in CASE_NAMES})
            writer.writerow(flat)

    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write("# Stage 1 Dataset Distribution Audit\n\n")
        f.write(f"{HEURISTIC_LIMITATIONS}\n\n")
        f.write(f"- Samples scanned: {summary['num_samples']}\n")
        for case in CASE_NAMES:
            f.write(f"- {case}: {summary['case_counts'][case]} ({summary['case_ratios'][case]:.3f})\n")
        f.write(f"- Force/contact availability ratio: {summary['force_contact_availability_ratio']:.3f}\n")


def main() -> None:
    args = parse_args()
    out_dir = make_output_dir(args)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    dataset = build_dataset(args)
    rows: list[dict[str, Any]] = []
    max_idx = min(len(dataset), args.sample_start + args.max_samples * args.stride)
    print(f"Scanning dataset samples {args.sample_start}:{max_idx}:{args.stride}")
    for sample_idx in range(args.sample_start, max_idx, args.stride):
        sample = dataset[sample_idx]
        batch = collate_one(sample, args)
        batch["_sample_indices"] = [sample_idx]
        rows.append(compute_sample_features(batch, args))
        if len(rows) >= args.max_samples:
            break

    summary = summarize_distribution(rows)
    save_distribution(rows, summary, out_dir)
    print(f"Saved distribution audit to {out_dir}")


if __name__ == "__main__":
    main()

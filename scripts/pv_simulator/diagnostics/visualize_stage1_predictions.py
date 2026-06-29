"""Visualize Stage 1 diagnostic prediction snapshots.

This is a thin wrapper around the existing point-cloud renderer. It accepts the
npz snapshots written by ``run_stage1_cases.py`` and renders comparable GT,
AE-reconstruction, SimDiT-prediction, and naive-baseline videos when available.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

os.makedirs("/tmp/matplotlib", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

CURRENT_FILE = os.path.abspath(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.pv_simulator.visualize import visualize_point_cloud_motion  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Stage 1 diagnostic snapshot videos.")
    parser.add_argument("--snapshot_npz", type=str, required=True,
                        help="Snapshot written by run_stage1_cases.py.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Defaults to a 'visualizations' folder next to the npz.")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--views", nargs="+", default=["birdseye", "side", "iso"],
                        choices=["birdseye", "side", "front", "iso"])
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--max_points_per_object", type=int, default=120)
    parser.add_argument("--auto_select_keypoints", type=int, default=8)
    return parser.parse_args()


def _normalize_traj(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 4:
        return arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected trajectory shape (T,N,6) or (1,T,N,6), got {arr.shape}")
    return arr


def main() -> None:
    args = parse_args()
    data = np.load(args.snapshot_npz)
    out_dir = args.output_dir or os.path.join(os.path.dirname(args.snapshot_npz), "visualizations")
    os.makedirs(out_dir, exist_ok=True)

    if "point_obj_idx" not in data:
        raise KeyError("snapshot_npz must contain point_obj_idx")
    point_obj_idx = np.asarray(data["point_obj_idx"])
    if point_obj_idx.ndim == 2:
        point_obj_idx = point_obj_idx[0]

    if "gt" not in data:
        raise KeyError("snapshot_npz must contain gt")
    gt = _normalize_traj(data["gt"])
    rendered = []
    for key, label in [
        ("gt", "gt"),
        ("ae_recon", "ae_recon"),
        ("simdit_pred", "simdit_pred"),
        ("naive_baseline", "naive_baseline"),
    ]:
        if key not in data:
            continue
        traj = _normalize_traj(data[key])
        out_path = os.path.join(out_dir, f"{label}.mp4")
        visualize_point_cloud_motion(
            point_states=traj,
            point_obj_idx=point_obj_idx,
            output_path=out_path,
            fps=args.fps,
            views=args.views,
            dpi=args.dpi,
            max_points_per_object=args.max_points_per_object,
            reference_point_states=gt if key != "gt" else None,
            auto_select_keypoints=args.auto_select_keypoints if key != "gt" else 0,
        )
        rendered.append(out_path)

    print("Rendered:")
    for path in rendered:
        print(f"  {path}")


if __name__ == "__main__":
    main()

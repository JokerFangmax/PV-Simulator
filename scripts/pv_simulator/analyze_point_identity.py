"""Analyze point-identity ambiguity in Stage 1 predictions.

This script tests whether a collapsing point cloud is caused more by:

1. Bad geometry generation: even as a set, predicted points do not match GT.
2. Identity mismatch: the predicted point set is roughly correct, but point
   indices no longer correspond to the same semantic points as GT.

For each evaluated sample, it runs the normal Stage 1 rollout and reports:

- index_rmse:
    Direct pointwise RMSE using the original point ordering.
- rigid_index_rmse:
    RMSE after best-fit rigid alignment from prediction -> GT, still using the
    original point ordering. This removes global pose error.
- best_match_rmse:
    RMSE after rigid alignment plus Hungarian matching. This measures how well
    the predicted point *set* matches GT when point identities are allowed to
    permute.

Interpretation:
- If index_rmse >> best_match_rmse, the model is generating a reasonable set of
  points but losing point identity across samples.
- If all three errors are large, the issue is deeper than permutation and the
  geometry itself is wrong.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

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
from videox_fun.pipeline.pipeline_simulation import SimulationPipeline

try:
    import wandb
except ImportError:  # pragma: no cover - wandb is optional
    wandb = None


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze point identity ambiguity in Stage 1 predictions")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Path to Stage 1 checkpoint directory (for example outputs/stage1/.../final).")
    parser.add_argument("--ae_ckpt_dir", type=str, required=True,
                        help="Path to the frozen Stage 0 AE checkpoint.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data_dir", type=str,
                       help="MOVI sample directory containing point_cloud_states.pkl + metadata.json.")
    group.add_argument("--shard_root", type=str,
                       help="Sharded MOVI dataset root (dataset root or webdataset root).")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Single sample index used when --sample_indices is not set.")
    parser.add_argument("--sample_indices", type=int, nargs="*", default=None,
                        help="Optional explicit list of sample indices to evaluate from --shard_root.")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of consecutive samples to evaluate starting from --sample_idx "
                             "when --sample_indices is not provided.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save predictions, metrics, and videos.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--views", type=str, nargs="+",
                        default=["birdseye", "side", "iso"],
                        choices=["birdseye", "side", "front", "iso"])
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--auto_select_keypoints", type=int, default=6)
    parser.add_argument("--max_objects", type=int, default=5)
    parser.add_argument("--d_state", type=int, default=64)
    parser.add_argument("--d_sim", type=int, default=512)
    parser.add_argument("--sim_ffn_dim", type=int, default=2048)
    parser.add_argument("--sim_num_heads", type=int, default=8)
    parser.add_argument("--sim_num_layers", type=int, default=10)
    parser.add_argument("--anchor_mode", type=str, default="local",
                        choices=["local"],
                        help="Anchor mode used by the checkpoint-compatible inference path. The earlier canonical_pca option is disabled because it was unstable across samples.")
    parser.add_argument("--report_to", type=str, default="wandb", choices=["none", "wandb"])
    parser.add_argument("--wandb_project", type=str, default="pv-simulator-analysis")
    parser.add_argument("--wandb_run_name", type=str, default="point-identity-analysis")
    return parser.parse_args()


def _kabsch_align(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    if pred.shape[0] < 3:
        pred_center = pred.mean(axis=0, keepdims=True)
        target_center = target.mean(axis=0, keepdims=True)
        return pred - pred_center + target_center

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
    return pred_c @ rot + target_center


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _pairwise_dist_rmse(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] < 2:
        return 0.0
    da = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=-1)
    db = np.linalg.norm(b[:, None, :] - b[None, :, :], axis=-1)
    tri = np.triu_indices(a.shape[0], k=1)
    return float(np.sqrt(np.mean((da[tri] - db[tri]) ** 2)))


def _best_match_rmse(pred_aligned: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray]:
    cost = np.linalg.norm(pred_aligned[:, None, :] - target[None, :, :], axis=-1)
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_pred = pred_aligned[row_ind]
    matched_target = target[col_ind]
    return _rmse(matched_pred, matched_target), col_ind.astype(np.int64)


def _compute_frame_metrics(pred: np.ndarray, gt: np.ndarray, point_obj_idx: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, List[float]] = {
        "index_rmse": [],
        "rigid_index_rmse": [],
        "best_match_rmse": [],
        "pairwise_dist_rmse": [],
        "assignment_gap": [],
    }

    unique_obj_ids = np.unique(point_obj_idx)
    for obj_id in unique_obj_ids:
        obj_mask = point_obj_idx == obj_id
        pred_obj = pred[obj_mask]
        gt_obj = gt[obj_mask]
        pred_aligned = _kabsch_align(pred_obj, gt_obj)
        index_rmse = _rmse(pred_obj, gt_obj)
        rigid_rmse = _rmse(pred_aligned, gt_obj)
        best_match_rmse, _ = _best_match_rmse(pred_aligned, gt_obj)
        pairwise_rmse = _pairwise_dist_rmse(pred_obj, gt_obj)

        metrics["index_rmse"].append(index_rmse)
        metrics["rigid_index_rmse"].append(rigid_rmse)
        metrics["best_match_rmse"].append(best_match_rmse)
        metrics["pairwise_dist_rmse"].append(pairwise_rmse)
        metrics["assignment_gap"].append(rigid_rmse - best_match_rmse)

    return {key: float(np.mean(values)) if values else 0.0 for key, values in metrics.items()}


def _summarize_sequence(pred: np.ndarray, gt: np.ndarray, point_obj_idx: np.ndarray) -> Dict[str, object]:
    per_frame_all = []
    per_frame_future = []
    for t in range(pred.shape[0]):
        frame_metrics = _compute_frame_metrics(pred[t, :, :3], gt[t, :, :3], point_obj_idx)
        frame_metrics["frame_idx"] = t
        per_frame_all.append(frame_metrics)
        if t > 0:
            per_frame_future.append(frame_metrics)

    def _mean_metric(rows: List[Dict[str, float]], key: str) -> float:
        if not rows:
            return 0.0
        return float(np.mean([row[key] for row in rows]))

    metric_keys = [
        "index_rmse",
        "rigid_index_rmse",
        "best_match_rmse",
        "pairwise_dist_rmse",
        "assignment_gap",
    ]
    summary = {}
    for key in metric_keys:
        summary[f"{key}_all_frames"] = _mean_metric(per_frame_all, key)
        summary[f"{key}_future_frames"] = _mean_metric(per_frame_future, key)

    summary["point_state_mae_all"] = float(np.mean(np.abs(pred - gt)))
    summary["point_state_mse_all"] = float(np.mean((pred - gt) ** 2))
    return {
        "summary": summary,
        "per_frame": per_frame_all,
    }


def _load_sample(args, sample_index: int):
    if args.data_dir is not None:
        sample = load_movi_sample(args.data_dir)
        sample_ref = args.data_dir
    else:
        sample = load_movi_shard_sample(args.shard_root, sample_index)
        sample_ref = f"{args.shard_root}#{sample_index}"
    return sample, sample_ref


def _infer_sample(args, pipeline: SimulationPipeline, sample, sample_seed: int) -> np.ndarray:
    device = torch.device(args.device)
    generator = torch.Generator(device=args.device).manual_seed(sample_seed)
    result = pipeline(
        c_floor=sample["c_floor"],
        c_id=sample["c_id"],
        c_mat=sample["c_mat"],
        c_mass=sample["c_mass"],
        c_static=sample["c_static"],
        c_force_raw=sample["c_force_raw"],
        x_s_init=sample["x_s_init"],
        point_obj_idx=sample["point_obj_idx_batched"],
        T=sample["T_latent"],
        num_inference_steps=args.num_inference_steps,
        generator=generator,
    )
    return result["x_s"][0].float().cpu().numpy()


def _maybe_init_wandb(args, config: Dict[str, object]):
    if args.report_to != "wandb":
        return None
    if wandb is None:
        raise ImportError("wandb is not installed but --report_to wandb was requested.")
    return wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=config)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    device = torch.device(args.device)

    if args.sample_indices:
        sample_indices = list(args.sample_indices)
    else:
        sample_indices = list(range(args.sample_idx, args.sample_idx + args.num_samples))

    run = _maybe_init_wandb(args, {
        "ckpt_dir": args.ckpt_dir,
        "ae_ckpt_dir": args.ae_ckpt_dir,
        "sample_indices": sample_indices,
        "num_inference_steps": args.num_inference_steps,
        "anchor_mode": args.anchor_mode,
        "dtype": args.dtype,
    })

    pipeline = SimulationPipeline.from_pretrained(
        ckpt_dir=args.ckpt_dir,
        ae_ckpt_dir=args.ae_ckpt_dir,
        device=args.device,
        dtype=dtype,
        max_objects=args.max_objects,
        d_state=args.d_state,
        d_sim=args.d_sim,
        sim_ffn_dim=args.sim_ffn_dim,
        sim_num_heads=args.sim_num_heads,
        sim_num_layers=args.sim_num_layers,
        anchor_mode=args.anchor_mode,
    )

    sample_summaries = []
    summary_rows_for_mean = []

    for sample_offset, sample_index in enumerate(sample_indices):
        sample, sample_ref = _load_sample(args, sample_index)
        pred = _infer_sample(args, pipeline, sample, sample_seed=args.seed + sample_offset)
        gt = sample["point_states"].numpy().astype(np.float32)
        point_obj_idx = sample["point_obj_idx"].numpy().astype(np.int64)

        sample_dir = os.path.join(args.output_dir, f"sample_{sample_index:05d}")
        os.makedirs(sample_dir, exist_ok=True)
        np.save(os.path.join(sample_dir, "pred.npy"), pred)
        np.save(os.path.join(sample_dir, "gt.npy"), gt)
        np.save(os.path.join(sample_dir, "point_obj_idx.npy"), point_obj_idx)

        metrics = _summarize_sequence(pred=pred, gt=gt, point_obj_idx=point_obj_idx)
        metrics["summary"].update({
            "sample_idx": int(sample_index),
            "sample_ref": sample_ref,
            "num_points": int(gt.shape[1]),
            "T_raw": int(gt.shape[0]),
            "num_objects": int(np.max(point_obj_idx)) + 1,
        })
        with open(os.path.join(sample_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        video_path = os.path.join(sample_dir, "pred_vs_gt_keypoints.mp4")
        visualize_point_cloud_motion(
            point_states=pred,
            point_obj_idx=point_obj_idx,
            output_path=video_path,
            fps=args.fps,
            views=args.views,
            dpi=args.dpi,
            reference_point_states=gt,
            auto_select_keypoints=args.auto_select_keypoints,
            keypoint_history=True,
        )

        sample_summary = metrics["summary"]
        sample_summaries.append(sample_summary)
        summary_rows_for_mean.append(sample_summary)

        if run is not None:
            log_row = {f"sample/{sample_index}/{k}": v for k, v in sample_summary.items() if isinstance(v, (int, float))}
            log_row["sample/sample_idx"] = sample_index
            run.log(log_row)
            run.log({
                f"sample/{sample_index}/pred_vs_gt_keypoints": wandb.Video(video_path, fps=args.fps, format="mp4")
            })

    aggregate = {}
    numeric_keys = [key for key, value in sample_summaries[0].items() if isinstance(value, (int, float))]
    for key in numeric_keys:
        aggregate[f"mean/{key}"] = float(np.mean([row[key] for row in summary_rows_for_mean]))
        aggregate[f"std/{key}"] = float(np.std([row[key] for row in summary_rows_for_mean]))

    experiment_summary = {
        "config": {
            "ckpt_dir": args.ckpt_dir,
            "ae_ckpt_dir": args.ae_ckpt_dir,
            "sample_indices": sample_indices,
            "num_inference_steps": args.num_inference_steps,
            "anchor_mode": args.anchor_mode,
            "dtype": args.dtype,
        },
        "aggregate": aggregate,
        "samples": sample_summaries,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(experiment_summary, f, indent=2)

    if run is not None:
        run.summary.update(aggregate)
        run.summary["output_dir"] = args.output_dir
        run.finish()

    print(json.dumps(experiment_summary, indent=2))


if __name__ == "__main__":
    main()

"""Stage 1 inference script for PV-Simulator.

Loads a sample from the MOVI-AB dataset (or custom npz), runs the trained
SimTransformer to generate a predicted trajectory from pure noise, and
renders side-by-side MP4/GIF animations of the ground truth vs prediction.

Usage:
    # From a MOVI-AB sample directory (ground truth + inference):
    python scripts/pv_simulator/infer_stage1.py \
        --ckpt_dir outputs/stage1_test/final \
        --ae_ckpt_dir outputs/ae/final \
        --data_dir datasets/movi_ab_10k/00000 \
        --output_dir outputs/stage1_test/infer \
        --num_inference_steps 50

    # From a saved numpy array:
    python scripts/pv_simulator/infer_stage1.py \
        --ckpt_dir outputs/stage1_test/final \
        --ae_ckpt_dir outputs/ae/final \
        --point_states_npy /path/to/states.npy \
        --point_obj_idx_npy /path/to/obj_idx.npy \
        --output_dir outputs/infer
"""

import argparse
import json
import os
import pickle
import sys

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

from videox_fun.pipeline.pipeline_simulation import SimulationPipeline


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_movi_sample(data_dir: str, temporal_compression_ratio: int = 4):
    """Load a MOVI-AB sample directory → tensors ready for the pipeline."""
    pkl_path = os.path.join(data_dir, 'point_cloud_states.pkl')
    meta_path = os.path.join(data_dir, 'metadata.json')

    with open(pkl_path, 'rb') as f:
        pkl = pickle.load(f)
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    # Point states — clip to largest valid T_raw = 4k+1
    point_states = pkl['point_states'].astype(np.float32)   # (T_avail, N, 6)
    T_avail = point_states.shape[0]
    r = temporal_compression_ratio
    k = (T_avail - 1) // r
    T_raw = k * r + 1
    assert T_raw >= r + 1, f"Too few frames: T_avail={T_avail}"
    point_states = point_states[:T_raw]                       # (T_raw, N, 6)
    T_raw, N, _ = point_states.shape

    # point_obj_idx
    instances = pkl['instances']
    n_objects = len(instances)
    point_obj_idx = np.zeros(N, dtype=np.int64)
    for obj_i, inst in enumerate(instances):
        start, end = inst['point_range']
        point_obj_idx[start:end] = obj_i

    # Per-object properties
    meta_instances = meta['instances']
    c_mat = np.zeros((n_objects, 2), dtype=np.float32)
    c_mass = np.zeros(n_objects, dtype=np.float32)
    c_init_state = np.zeros((n_objects, 6), dtype=np.float32)
    for obj_i, mi in enumerate(meta_instances):
        c_mat[obj_i, 0] = mi.get('friction', 0.5)
        c_mat[obj_i, 1] = mi.get('restitution', 0.5)
        c_mass[obj_i] = mi.get('mass', 1.0)
        c_init_state[obj_i, :3] = mi['positions'][0]
        c_init_state[obj_i, 3:6] = mi['velocities'][0]

    T_latent = k + 1

    return {
        'point_states': torch.from_numpy(point_states),               # (T_raw, N, 6)
        'point_obj_idx': torch.from_numpy(point_obj_idx),             # (N,)
        'c_floor': torch.tensor([0.0]),                                # (1,)
        'c_id': torch.arange(n_objects, dtype=torch.long).unsqueeze(0),  # (1, n_obj)
        'c_mat': torch.from_numpy(c_mat).unsqueeze(0),                # (1, n_obj, 2)
        'c_mass': torch.from_numpy(c_mass).unsqueeze(0),              # (1, n_obj)
        'c_static': torch.zeros(1, n_objects, dtype=torch.long),      # (1, n_obj)
        'c_force_raw': torch.zeros(1, T_raw, N, 6),                   # (1, T_raw, N, 6)
        'x_s_init': torch.from_numpy(point_states[:1]).unsqueeze(0),  # (1, 1, N, 6)
        'point_obj_idx_batched': torch.from_numpy(point_obj_idx).unsqueeze(0),  # (1, N)
        'T_latent': T_latent,
        'T_raw': T_raw,
        'n_objects': n_objects,
        'N': N,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 inference for PV-Simulator")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Path to Stage 1 checkpoint directory (e.g. outputs/stage1/final)")
    # Input data
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data_dir", type=str,
                       help="MOVI-AB sample directory (contains point_cloud_states.pkl + metadata.json)")
    group.add_argument("--point_states_npy", type=str,
                       help="Path to .npy file with ground truth point states (T_raw, N, 6)")
    parser.add_argument("--point_obj_idx_npy", type=str, default=None,
                        help="Path to .npy file with point-to-object mapping (N,). "
                             "Required when --point_states_npy is used with multiple objects.")
    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/infer",
                        help="Directory to save output animations")
    # Inference options
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    # AE checkpoint (from Stage 0)
    parser.add_argument("--ae_ckpt_dir", type=str, required=True,
                        help="Path to Stage 0 CausalAE checkpoint directory")
    # Model architecture (must match the checkpoint)
    parser.add_argument("--d_state", type=int, default=64)
    parser.add_argument("--d_sim", type=int, default=512)
    parser.add_argument("--sim_ffn_dim", type=int, default=2048)
    parser.add_argument("--sim_num_heads", type=int, default=8)
    parser.add_argument("--sim_num_layers", type=int, default=10)
    parser.add_argument("--max_objects", type=int, default=5)
    # Visualization
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--views", type=str, nargs='+',
                        default=["birdseye", "side", "iso"],
                        choices=["birdseye", "side", "front", "iso"],
                        help="Views to render side by side. Default: birdseye side iso")
    parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    # --- Load sample ---
    if args.data_dir is not None:
        sample = load_movi_sample(args.data_dir)
    else:
        point_states_np = np.load(args.point_states_npy).astype(np.float32)
        if args.point_obj_idx_npy is not None:
            point_obj_idx_np = np.load(args.point_obj_idx_npy).astype(np.int64)
        else:
            point_obj_idx_np = np.zeros(point_states_np.shape[1], dtype=np.int64)

        T_raw, N, _ = point_states_np.shape
        r = 4
        k = (T_raw - 1) // r
        assert (T_raw - 1) % r == 0, f"T_raw={T_raw} is not 4k+1"
        n_objects = int(point_obj_idx_np.max()) + 1

        sample = {
            'point_states': torch.from_numpy(point_states_np),
            'point_obj_idx': torch.from_numpy(point_obj_idx_np),
            'c_floor': torch.tensor([0.0]),
            'c_id': torch.arange(n_objects, dtype=torch.long).unsqueeze(0),
            'c_mat': torch.full((1, n_objects, 2), 0.5),
            'c_mass': torch.ones(1, n_objects),
            'c_static': torch.zeros(1, n_objects, dtype=torch.long),
            'c_force_raw': torch.zeros(1, T_raw, N, 6),
            'x_s_init': torch.from_numpy(point_states_np[:1]).unsqueeze(0),
            'point_obj_idx_batched': torch.from_numpy(point_obj_idx_np).unsqueeze(0),
            'T_latent': k + 1,
            'T_raw': T_raw,
            'n_objects': n_objects,
            'N': N,
        }

    T_latent = sample['T_latent']
    T_raw = sample['T_raw']
    point_obj_idx_np = sample['point_obj_idx'].numpy()

    print(f"Sample: T_raw={T_raw}, T_latent={T_latent}, N={sample['N']}, "
          f"n_objects={sample['n_objects']}")

    # --- Load pipeline ---
    print(f"Loading checkpoint from {args.ckpt_dir} ...")
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
    )

    # --- Run inference ---
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    print(f"Running inference ({args.num_inference_steps} steps) ...")
    result = pipeline(
        c_floor=sample['c_floor'],
        c_id=sample['c_id'],
        c_mat=sample['c_mat'],
        c_mass=sample['c_mass'],
        c_static=sample['c_static'],
        c_force_raw=sample['c_force_raw'],
        x_s_init=sample['x_s_init'],
        point_obj_idx=sample['point_obj_idx_batched'],
        T=T_latent,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
    )

    # x_s_pred: (1, T_raw, N, 6) — take first (and only) batch item
    x_s_pred = result['x_s'][0].float().cpu().numpy()   # (T_raw, N, 6)
    x_s_gt = sample['point_states'].numpy()              # (T_raw, N, 6)

    # --- Visualize ---
    from scripts.pv_simulator.visualize import visualize_point_cloud_motion

    os.makedirs(args.output_dir, exist_ok=True)
    gt_path = os.path.join(args.output_dir, "gt.mp4")
    pred_path = os.path.join(args.output_dir, "pred.mp4")

    print("Rendering ground truth ...")
    visualize_point_cloud_motion(
        point_states=x_s_gt,
        point_obj_idx=point_obj_idx_np,
        output_path=gt_path,
        fps=args.fps,
        views=args.views,
        dpi=args.dpi,
    )

    print("Rendering prediction ...")
    visualize_point_cloud_motion(
        point_states=x_s_pred,
        point_obj_idx=point_obj_idx_np,
        output_path=pred_path,
        fps=args.fps,
        views=args.views,
        dpi=args.dpi,
    )

    print(f"\nDone. Outputs saved to {args.output_dir}/")
    print(f"  Ground truth : {gt_path}")
    print(f"  Prediction   : {pred_path}")


if __name__ == "__main__":
    main()

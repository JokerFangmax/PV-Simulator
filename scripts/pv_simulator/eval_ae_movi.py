"""Evaluate a CausalAE checkpoint on MOVI-AB data (real physics trajectories).

Splits each sample's x_s_raw (T_raw, N, 6) into pos(:3) and vel(3:), runs each
through the AE independently (matching Stage 1 usage), and reports reconstruction
MSE/MAE — overall and split by pos/vel.

Usage:
    python scripts/pv_simulator/eval_ae_movi.py \
        outputs/stage0/diag/stage_mlp/causal-mae-long-fp32-50k/final \
        --data_root datasets/movi_ab_10k
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

current_file_path = os.path.abspath(__file__)
for _root in [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]:
    if _root not in sys.path:
        sys.path.insert(0, _root)

from videox_fun.models.sim_ae import CausalAE
from videox_fun.data.dataset_simulation import MoviSimulationDataset


@torch.no_grad()
def eval_on_movi(ckpt_path: str, data_root: str, device: str = "cuda",
                 max_objects: int = 16, max_samples: int = None):
    ae = CausalAE.load(ckpt_path, map_location=device)
    ae = ae.to(device)
    ae.eval()

    dataset = MoviSimulationDataset(data_root=data_root, max_objects=max_objects)
    n = len(dataset) if max_samples is None else min(len(dataset), max_samples)

    # Accumulators (element-wise sums + counts for exact mean across variable shapes)
    stats = {}
    for key in ["all", "pos", "vel"]:
        stats[key] = {"se_sum": 0.0, "ae_sum": 0.0, "count": 0}

    # Per-channel (x/y/z) for pos and vel
    per_channel = {"pos": [[0.0, 0.0, 0] for _ in range(3)],  # [se, ae, count]
                   "vel": [[0.0, 0.0, 0] for _ in range(3)]}

    for idx in range(n):
        sample = dataset[idx]
        x = sample["x_s_raw"].to(device).unsqueeze(0).float()  # (1, T_raw, N, 6)
        T_raw = x.shape[1]

        pos = x[..., :3]
        vel = x[..., 3:6]

        pos_hat, _ = ae(pos)
        vel_hat, _ = ae(vel)

        # If AE outputs slightly different T (shouldn't, but safeguard)
        pos_hat = pos_hat[:, :T_raw]
        vel_hat = vel_hat[:, :T_raw]

        for key, a, b in [("pos", pos, pos_hat), ("vel", vel, vel_hat)]:
            se = (a - b).pow(2).sum().item()
            ae_sum = (a - b).abs().sum().item()
            cnt = a.numel()
            stats[key]["se_sum"] += se
            stats[key]["ae_sum"] += ae_sum
            stats[key]["count"] += cnt
            stats["all"]["se_sum"] += se
            stats["all"]["ae_sum"] += ae_sum
            stats["all"]["count"] += cnt

            # Per-channel
            for c in range(3):
                diff = (a[..., c] - b[..., c])
                per_channel[key][c][0] += diff.pow(2).sum().item()
                per_channel[key][c][1] += diff.abs().sum().item()
                per_channel[key][c][2] += diff.numel()

        if (idx + 1) % 20 == 0 or idx + 1 == n:
            print(f"  [{idx+1}/{n}] running MAE(all)={stats['all']['ae_sum']/stats['all']['count']:.6f}")

    print(f"\n=== MOVI-AB Reconstruction: {ckpt_path} ===")
    print(f"Samples evaluated: {n}")
    print(f"{'split':<8} {'MSE':>12} {'MAE':>12}")
    for key in ["all", "pos", "vel"]:
        s = stats[key]
        mse = s["se_sum"] / s["count"]
        mae = s["ae_sum"] / s["count"]
        print(f"{key:<8} {mse:>12.6e} {mae:>12.6e}")

    print(f"\nPer-channel breakdown:")
    print(f"{'split':<6} {'ch':<4} {'MSE':>12} {'MAE':>12}")
    for key in ["pos", "vel"]:
        for c, name in enumerate(["x", "y", "z"]):
            se, ae_, cnt = per_channel[key][c]
            print(f"{key:<6} {name:<4} {se/cnt:>12.6e} {ae_/cnt:>12.6e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="AE checkpoint dir (e.g. .../final)")
    parser.add_argument("--data_root", type=str, default="datasets/movi_ab_10k")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_objects", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional cap on number of samples (default: all)")
    args = parser.parse_args()

    eval_on_movi(args.checkpoint, args.data_root, args.device,
                 max_objects=args.max_objects, max_samples=args.max_samples)


if __name__ == "__main__":
    main()

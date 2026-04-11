"""Evaluate CausalAE checkpoints on synthetic trajectories.

Reports MSE, MMD, and interpolation consistency — overall, per-category, and
per-generator.

Usage:
    python scripts/pv_simulator/eval_ae.py outputs/stage0/exp-a/final outputs/stage0/exp-b/final
    python scripts/pv_simulator/eval_ae.py outputs/stage0/exp-a/checkpoint-10000 --json
"""

import argparse
import json
import os
import random
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
from scripts.pv_simulator.train_stage0 import TRAJECTORY_GENERATORS, mmd_imq

# ---------------------------------------------------------------------------
# Category grouping (mirrors comments in train_stage0.py TRAJECTORY_GENERATORS)
# ---------------------------------------------------------------------------
GENERATOR_CATEGORIES = {
    # Smooth continuous — positions/velocities
    "gen_smooth_fourier":     "smooth",
    "gen_ballistic":          "smooth",
    "gen_sinusoidal":         "smooth",
    "gen_damped_oscillation": "smooth",
    "gen_circular":           "smooth",
    "gen_constant_velocity":  "smooth",
    "gen_brownian":           "smooth",
    "gen_exponential_decay":  "smooth",
    # Sharp / discontinuous — contact forces, collisions
    "gen_bounce":             "sharp",
    "gen_smooth_then_sharp":  "sharp",
    "gen_sudden_stop":        "sharp",
    "gen_piecewise_linear":   "sharp",
    "gen_piecewise_constant": "sharp",
    # Sparse / intermittent — force vectors, contact points
    "gen_sparse_impulse":     "sparse",
    "gen_sparse_constant":    "sparse",
    "gen_step_function":      "sparse",
    "gen_zero":               "sparse",
    # Other
    "gen_stationary":         "other",
    "gen_gaussian_noise":     "other",
}

CATEGORY_ORDER = ["smooth", "sharp", "sparse", "other"]


METRIC_NAMES = ["mse", "mmd", "interp"]


@torch.no_grad()
def eval_checkpoint(ckpt_path: str, device: str = "cpu", n_eval: int = 512, seed: int = 0):
    """Evaluate a CausalAE checkpoint.

    Returns:
        (overall_metrics, per_generator, per_category)
        - overall_metrics: dict {metric: value} for mse/mmd/interp
        - per_generator: dict {gen_name: {metric: value}}
        - per_category: dict {category: {metric: value}}
    """
    random.seed(seed)
    torch.manual_seed(seed)
    ae = CausalAE.load(ckpt_path, map_location=device)
    ae = ae.to(device)
    ae.eval()
    d_latent = ae.d_latent

    per_generator = {}
    batch_size = max(1, n_eval // 64)

    for gen_fn, _ in TRAJECTORY_GENERATORS:
        name = gen_fn.__name__
        totals = {m: 0.0 for m in METRIC_NAMES}
        for k in range(1, 6):  # T_raw from 5 to 21
            T_raw = 4 * k + 1
            x = gen_fn(batch_size, T_raw, 16, 3, device)
            x_hat, z = ae(x)

            # MSE
            totals["mse"] += F.mse_loss(x_hat.float(), x.float()).item()

            # MMD
            z_flat = z.reshape(-1, d_latent).float()
            p = torch.randn_like(z_flat)
            totals["mmd"] += mmd_imq(z_flat, p).item()

            # Interpolation consistency
            perm = torch.randperm(batch_size, device=x.device)
            x2 = x[perm]
            alpha = torch.rand(batch_size, 1, 1, 1, device=x.device)
            x_interp = alpha * x + (1.0 - alpha) * x2
            _, z_interp = ae(x_interp)
            _, z2 = ae(x2)
            z_target = alpha * z + (1.0 - alpha) * z2
            totals["interp"] += F.mse_loss(z_interp.float(), z_target.float()).item()

        per_generator[name] = {m: totals[m] / 5 for m in METRIC_NAMES}

    # Per-category aggregation
    per_category = {}
    for cat in CATEGORY_ORDER:
        cat_gens = [name for name, c in GENERATOR_CATEGORIES.items() if c == cat]
        cat_data = [per_generator[g] for g in cat_gens if g in per_generator]
        if cat_data:
            per_category[cat] = {
                m: sum(d[m] for d in cat_data) / len(cat_data) for m in METRIC_NAMES
            }
        else:
            per_category[cat] = {m: 0.0 for m in METRIC_NAMES}

    overall = {
        m: sum(d[m] for d in per_generator.values()) / len(per_generator)
        for m in METRIC_NAMES
    }
    return overall, per_generator, per_category


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint directories to evaluate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_eval", type=int, default=256)
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    results_list = []

    for ckpt in args.checkpoints:
        if not os.path.exists(ckpt):
            if not args.json:
                print(f"{ckpt}: NOT FOUND")
            continue
        try:
            overall, per_gen, per_cat = eval_checkpoint(
                ckpt, device=args.device, n_eval=args.n_eval
            )
            result = {
                "checkpoint": ckpt,
                "overall": overall,
                "per_category": per_cat,
                "per_generator": per_gen,
            }
            results_list.append(result)
        except Exception as e:
            if not args.json:
                print(f"{ckpt}: ERROR: {e}")

    if args.json:
        print(json.dumps(results_list, indent=2))
        return

    # Table output — one section per metric
    for metric in METRIC_NAMES:
        print(f"\n=== {metric.upper()} ===")
        header = (f"{'Checkpoint':<40} {'Overall':>8}  {'smooth':>8}  {'sharp':>8}  "
                  f"{'sparse':>8}  {'other':>8}")
        print(header)
        print("-" * len(header))

        for r in results_list:
            cat = r["per_category"]
            print(f"{r['checkpoint']:<40} {r['overall'][metric]:>8.4f}  "
                  f"{cat.get('smooth', {}).get(metric, 0):>8.4f}  "
                  f"{cat.get('sharp', {}).get(metric, 0):>8.4f}  "
                  f"{cat.get('sparse', {}).get(metric, 0):>8.4f}  "
                  f"{cat.get('other', {}).get(metric, 0):>8.4f}")

    # Worst generators by MSE
    if results_list:
        print(f"\n=== Top-3 worst generators (MSE) ===")
        for r in results_list:
            worst = sorted(r["per_generator"].items(), key=lambda x: x[1]["mse"], reverse=True)[:3]
            worst_str = ", ".join(f"{n.replace('gen_','')[:14]}={v['mse']:.4f}" for n, v in worst)
            print(f"{r['checkpoint']:<40} {worst_str}")

    if len(results_list) > 1:
        best = min(results_list, key=lambda r: r["overall"]["mse"])
        print(f"\nBest by MSE: {best['checkpoint']} (MSE={best['overall']['mse']:.4f}, "
              f"MMD={best['overall']['mmd']:.4f}, interp={best['overall']['interp']:.4f})")


if __name__ == "__main__":
    main()

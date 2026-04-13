"""Plot GT vs reconstruction for all trajectory generators.

Loads the trained CausalAE, generates one sample per generator, encodes+decodes,
and plots the first component (x-axis) of the first sample over time.

Usage:
    conda run -n asr python scripts/pv_simulator/plot_ae_reconstruction.py \
        --ckpt outputs/stage0/default/final \
        --out outputs/stage0/ae_reconstruction.png
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from scripts.pv_simulator.train_stage0 import TRAJECTORY_GENERATORS
from videox_fun.models.sim_ae import CausalAE


# 4 broad categories
CATEGORY_MAP = {
    "gen_smooth_fourier": "smooth",
    "gen_ballistic": "smooth",
    "gen_sinusoidal": "smooth",
    "gen_damped_oscillation": "smooth",
    "gen_circular": "smooth",
    "gen_constant_velocity": "smooth",
    "gen_brownian": "smooth",
    "gen_exponential_decay": "smooth",
    "gen_bounce": "sharp",
    "gen_smooth_then_sharp": "sharp",
    "gen_sudden_stop": "sharp",
    "gen_piecewise_linear": "sharp",
    "gen_piecewise_constant": "sharp",
    "gen_sparse_impulse": "sparse",
    "gen_sparse_constant": "sparse",
    "gen_step_function": "sparse",
    "gen_zero": "sparse",
    "gen_stationary": "other",
    "gen_gaussian_noise": "other",
}

CATEGORY_COLORS = {
    "smooth": "#1f77b4",
    "sharp": "#d62728",
    "sparse": "#2ca02c",
    "other": "#9467bd",
}


def build_reconstruction_figure(ae, device, t_raw=21, n_points=16,
                                c_in=3, component=0, seed=123):
    """Build a reconstruction figure with GT vs AE output for every generator.

    Args:
        ae: Loaded CausalAE (already on `device`, eval mode set by caller).
        device: torch device the model lives on.
        t_raw: Number of raw frames (must be 4k+1).
        n_points: Number of point trajectories per sample.
        c_in: Input channel count (typically 3).
        component: Which channel to plot (0..c_in-1).
        seed: Seed base so figures are reproducible across runs.

    Returns:
        (fig, overall_mse) — matplotlib Figure (caller must close) and the
        average MSE across all generators (for logging).
    """
    gens = [(fn, w) for fn, w in TRAJECTORY_GENERATORS]
    gens.sort(key=lambda x: (
        ["smooth", "sharp", "sparse", "other"].index(CATEGORY_MAP[x[0].__name__]),
        x[0].__name__,
    ))

    n = len(gens)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.5 * nrows), sharex=True)
    axes = axes.flatten()

    t_idx = torch.arange(t_raw).cpu().numpy()

    mse_sum = 0.0
    mse_count = 0

    for i, (gen_fn, _) in enumerate(gens):
        ax = axes[i]
        name = gen_fn.__name__
        cat = CATEGORY_MAP[name]
        color = CATEGORY_COLORS[cat]

        torch.manual_seed(seed + i)
        with torch.no_grad():
            x = gen_fn(1, t_raw, n_points, c_in, device)  # (1, T_raw, N, c_in)
            x_hat, _ = ae(x)

        gt = x[0, :, 0, component].float().cpu().numpy()
        rec = x_hat[0, :, 0, component].float().cpu().numpy()

        mse = float(((gt - rec) ** 2).mean())
        mse_sum += mse
        mse_count += 1

        ax.plot(t_idx, gt, "-o", color=color, markersize=4, linewidth=1.8,
                label="GT", alpha=0.9)
        ax.plot(t_idx, rec, "--x", color="black", markersize=5, linewidth=1.3,
                label="AE rec", alpha=0.85)

        short_name = name.replace("gen_", "")
        ax.set_title(f"[{cat}] {short_name}  (MSE={mse:.3f})",
                     fontsize=10, color=color)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=8, loc="best")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"CausalAE reconstruction  (c_mid={ae.c_mid}, d_latent={ae.d_latent}, "
        f"T_raw={t_raw}, component={component})",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()

    overall_mse = mse_sum / max(1, mse_count)
    return fig, overall_mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="outputs/stage0/default/final")
    parser.add_argument("--out", type=str,
                        default="outputs/stage0/ae_reconstruction.png")
    parser.add_argument("--t_raw", type=int, default=21,
                        help="Number of raw frames (must be 4k+1)")
    parser.add_argument("--n_points", type=int, default=16,
                        help="Number of independent point trajectories per sample")
    parser.add_argument("--c_in", type=int, default=3)
    parser.add_argument("--component", type=int, default=0,
                        help="Which component (0..c_in-1) to plot")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading AE from {args.ckpt}")
    ae = CausalAE.load(args.ckpt)
    ae = ae.to(device).eval()
    print(f"  c_mid={ae.c_mid}, d_latent={ae.d_latent}")

    fig, overall_mse = build_reconstruction_figure(
        ae, device,
        t_raw=args.t_raw,
        n_points=args.n_points,
        c_in=args.c_in,
        component=args.component,
        seed=args.seed,
    )
    print(f"  overall MSE: {overall_mse:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()

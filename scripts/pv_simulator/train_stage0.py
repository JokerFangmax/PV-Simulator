"""Stage 0: Train Causal Autoencoder for PV-Simulator.

Trains a CausalAE to compress (T_raw, 3) trajectories to (T, 16) latent
with 4x temporal reduction. Uses synthesized random data (no physics data
needed). After training, AE weights are frozen and reused in Stages 1+2.

Losses: MSE reconstruction + MMD regularization (WAE) + temporal smoothness
        + linear interpolation consistency.

Usage:
    python scripts/pv_simulator/train_stage0.py \
        --output_dir outputs/ae \
        --max_train_steps 50000 \
        --batch_size 256

    # Multi-GPU:
    accelerate launch --num_processes=4 scripts/pv_simulator/train_stage0.py \
        --output_dir outputs/ae \
        --max_train_steps 50000 \
        --batch_size 256
"""

import argparse
import logging
import math
import os
import random
import sys

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm

# Add project root to path
current_file_path = os.path.abspath(__file__)
for _root in [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]:
    if _root not in sys.path:
        sys.path.insert(0, _root)

from videox_fun.models.sim_ae import CausalAE

logger = get_logger(__name__, log_level="INFO")


# ---------------------------------------------------------------------------
# Synthetic trajectory generators
# ---------------------------------------------------------------------------
# Each generator returns (B, T_raw, N_pts, c_in) on the given device.
# They cover the full spectrum of signals the AE will encounter:
#   - Smooth continuous: positions, velocities in physical simulations
#   - Sharp/discontinuous: contact forces, collisions
#   - Sparse: intermittent forces, contact points
#   - Stationary: objects at rest
#   - Mixed: real scenes combine all of the above

def _time_grid(B, T_raw, N, c_in, device):
    """Normalized time grid [0, 1] → (B, T_raw, 1, 1) broadcast-ready."""
    t = torch.linspace(0, 1, T_raw, device=device).view(1, T_raw, 1, 1)
    return t.expand(B, T_raw, 1, 1)


def gen_gaussian_noise(B, T_raw, N, c_in, device):
    """Pure i.i.d. Gaussian noise — baseline, stress-tests reconstruction."""
    std = torch.empty(B, 1, 1, 1, device=device).uniform_(0.5, 10.0)
    return torch.randn(B, T_raw, N, c_in, device=device) * std


def gen_stationary(B, T_raw, N, c_in, device):
    """Near-constant signal with tiny jitter — objects at rest."""
    offset = torch.randn(B, 1, N, c_in, device=device) * 3.0
    jitter = torch.randn(B, T_raw, N, c_in, device=device) * 0.01
    return offset + jitter


def gen_constant_velocity(B, T_raw, N, c_in, device):
    """Linear trajectories x(t) = x0 + v*t — uniform motion."""
    t = _time_grid(B, T_raw, N, c_in, device)
    x0 = torch.randn(B, 1, N, c_in, device=device) * 3.0
    v = torch.randn(B, 1, N, c_in, device=device) * 5.0
    return x0 + v * t


def gen_ballistic(B, T_raw, N, c_in, device):
    """Parabolic trajectories x(t) = x0 + v0*t + 0.5*a*t^2 — gravity/throws."""
    t = _time_grid(B, T_raw, N, c_in, device)
    x0 = torch.randn(B, 1, N, c_in, device=device) * 2.0
    v0 = torch.randn(B, 1, N, c_in, device=device) * 4.0
    a = torch.randn(B, 1, N, c_in, device=device) * 6.0
    return x0 + v0 * t + 0.5 * a * t ** 2


def gen_sinusoidal(B, T_raw, N, c_in, device):
    """Sinusoidal oscillation — springs, vibrations, periodic motion."""
    t = _time_grid(B, T_raw, N, c_in, device)
    amp = torch.empty(B, 1, N, c_in, device=device).uniform_(0.5, 5.0)
    freq = torch.empty(B, 1, N, c_in, device=device).uniform_(0.5, 8.0)
    phase = torch.empty(B, 1, N, c_in, device=device).uniform_(0, 2 * math.pi)
    offset = torch.randn(B, 1, N, c_in, device=device) * 2.0
    return offset + amp * torch.sin(2 * math.pi * freq * t + phase)


def gen_damped_oscillation(B, T_raw, N, c_in, device):
    """Exponentially damped sinusoid — friction, energy dissipation."""
    t = _time_grid(B, T_raw, N, c_in, device)
    amp = torch.empty(B, 1, N, c_in, device=device).uniform_(1.0, 8.0)
    freq = torch.empty(B, 1, N, c_in, device=device).uniform_(1.0, 6.0)
    phase = torch.empty(B, 1, N, c_in, device=device).uniform_(0, 2 * math.pi)
    decay = torch.empty(B, 1, N, c_in, device=device).uniform_(1.0, 8.0)
    offset = torch.randn(B, 1, N, c_in, device=device) * 1.0
    return offset + amp * torch.exp(-decay * t) * torch.sin(2 * math.pi * freq * t + phase)


def gen_smooth_fourier(B, T_raw, N, c_in, device):
    """Random smooth curves via low-frequency Fourier basis — general smooth motion."""
    t = _time_grid(B, T_raw, N, c_in, device)  # (B, T, 1, 1)
    n_harmonics = random.randint(2, 6)
    x = torch.zeros(B, T_raw, N, c_in, device=device)
    for k in range(1, n_harmonics + 1):
        amp = torch.randn(B, 1, N, c_in, device=device) * (3.0 / k)
        phase = torch.empty(B, 1, N, c_in, device=device).uniform_(0, 2 * math.pi)
        x = x + amp * torch.sin(2 * math.pi * k * t + phase)
    return x


def gen_circular(B, T_raw, N, c_in, device):
    """Circular/spiral motion — rotation, orbits. Pairs channels for rotation."""
    t = _time_grid(B, T_raw, N, c_in, device)
    radius = torch.empty(B, 1, N, 1, device=device).uniform_(0.5, 5.0)
    freq = torch.empty(B, 1, N, 1, device=device).uniform_(0.5, 4.0)
    phase = torch.empty(B, 1, N, 1, device=device).uniform_(0, 2 * math.pi)
    center = torch.randn(B, 1, N, c_in, device=device) * 2.0
    # Radial growth for spirals
    growth = torch.empty(B, 1, N, 1, device=device).uniform_(-1.0, 1.0)
    r = radius + growth * t.expand(B, T_raw, N, 1)
    angle = 2 * math.pi * freq * t.expand(B, T_raw, N, 1) + phase
    x = torch.zeros(B, T_raw, N, c_in, device=device)
    # Fill channels in pairs (cos, sin); if odd channel left, use cos
    for ch in range(0, c_in, 2):
        x[..., ch] = (r * torch.cos(angle + ch * 0.5))[..., 0]
        if ch + 1 < c_in:
            x[..., ch + 1] = (r * torch.sin(angle + ch * 0.5))[..., 0]
    return center + x


def gen_brownian(B, T_raw, N, c_in, device):
    """Brownian motion / random walk — cumulative small steps."""
    step_std = torch.empty(B, 1, 1, 1, device=device).uniform_(0.1, 2.0)
    steps = torch.randn(B, T_raw, N, c_in, device=device) * step_std
    steps[:, 0] = torch.randn(B, N, c_in, device=device) * 2.0  # random start
    return steps.cumsum(dim=1)


def gen_exponential_decay(B, T_raw, N, c_in, device):
    """Exponential decay/growth — energy dissipation, temperature cooling."""
    t = _time_grid(B, T_raw, N, c_in, device)
    amp = torch.randn(B, 1, N, c_in, device=device) * 5.0
    rate = torch.empty(B, 1, N, c_in, device=device).uniform_(0.5, 5.0)
    offset = torch.randn(B, 1, N, c_in, device=device) * 1.0
    # Randomly choose decay vs growth
    sign = torch.sign(torch.randn(B, 1, N, c_in, device=device))
    return offset + amp * torch.exp(sign * rate * t)


def gen_piecewise_constant(B, T_raw, N, c_in, device):
    """Piecewise constant — force on/off, discrete state changes."""
    n_segments = random.randint(2, 5)
    x = torch.zeros(B, T_raw, N, c_in, device=device)
    # Random change points
    change_points = sorted(random.sample(range(1, T_raw), min(n_segments - 1, T_raw - 1)))
    change_points = [0] + change_points + [T_raw]
    for i in range(len(change_points) - 1):
        t_start, t_end = change_points[i], change_points[i + 1]
        val = torch.randn(B, 1, N, c_in, device=device) * 4.0
        x[:, t_start:t_end] = val
    return x


def gen_piecewise_linear(B, T_raw, N, c_in, device):
    """Piecewise linear — velocity changes at discrete points (collisions)."""
    n_segments = random.randint(2, 5)
    change_points = sorted(random.sample(range(1, T_raw), min(n_segments - 1, T_raw - 1)))
    change_points = [0] + change_points + [T_raw]

    x = torch.zeros(B, T_raw, N, c_in, device=device)
    # Values at each change point
    vals = [torch.randn(B, 1, N, c_in, device=device) * 4.0
            for _ in range(len(change_points))]

    for i in range(len(change_points) - 1):
        t_s, t_e = change_points[i], change_points[i + 1]
        length = t_e - t_s
        if length == 0:
            continue
        alpha = torch.linspace(0, 1, length, device=device).view(1, length, 1, 1)
        x[:, t_s:t_e] = (1 - alpha) * vals[i] + alpha * vals[i + 1]
    return x


def gen_bounce(B, T_raw, N, c_in, device):
    """Bounce / collision pattern — smooth parabolas with sudden velocity reversals."""
    t = torch.linspace(0, 1, T_raw, device=device)
    # Generate 2-4 bounce events
    n_bounces = random.randint(1, 3)
    bounce_times = sorted([random.uniform(0.15, 0.85) for _ in range(n_bounces)])

    x0 = torch.randn(B, 1, N, c_in, device=device) * 2.0
    v0 = torch.randn(B, 1, N, c_in, device=device) * 4.0
    a = torch.randn(B, 1, N, c_in, device=device) * 8.0
    restitution = torch.empty(B, 1, N, c_in, device=device).uniform_(0.4, 0.95)

    x = torch.zeros(B, T_raw, N, c_in, device=device)
    seg_start = 0
    cur_x0, cur_v0 = x0, v0

    for bounce_t in bounce_times + [1.0]:
        seg_end = int(bounce_t * (T_raw - 1)) + 1
        seg_end = min(seg_end, T_raw)
        if seg_start >= seg_end:
            continue
        dt = t[seg_start:seg_end].view(1, -1, 1, 1) - t[seg_start].item()
        x[:, seg_start:seg_end] = cur_x0 + cur_v0 * dt + 0.5 * a * dt ** 2
        # Compute velocity at end of segment for next segment
        seg_dur = t[min(seg_end, T_raw) - 1].item() - t[seg_start].item()
        cur_v0 = -(cur_v0 + a * seg_dur) * restitution  # reverse + dampen
        cur_x0 = x[:, seg_end - 1:seg_end]
        seg_start = seg_end

    return x


def gen_sparse_impulse(B, T_raw, N, c_in, device):
    """Sparse impulses — mostly zero with occasional short bursts. Models contact forces."""
    x = torch.zeros(B, T_raw, N, c_in, device=device)
    # 1-3 impulse windows per trajectory
    n_impulses = random.randint(1, 3)
    for _ in range(n_impulses):
        # Random start and duration (1-3 frames)
        dur = random.randint(1, min(3, T_raw))
        start = random.randint(0, T_raw - dur)
        # Only affect a random subset of points (~30-80%)
        point_mask = torch.rand(B, 1, N, 1, device=device) < random.uniform(0.3, 0.8)
        impulse_val = torch.randn(B, dur, N, c_in, device=device) * 5.0
        x[:, start:start + dur] += impulse_val * point_mask
    return x


def gen_step_function(B, T_raw, N, c_in, device):
    """Step function — abrupt on/off transitions. Models contact begin/end."""
    x = torch.zeros(B, T_raw, N, c_in, device=device)
    # Random step time
    step_t = random.randint(1, T_raw - 1)
    val_before = torch.randn(B, 1, N, c_in, device=device) * 3.0
    val_after = torch.randn(B, 1, N, c_in, device=device) * 3.0
    # Randomly choose: zero→nonzero, nonzero→zero, or nonzero→nonzero
    pattern = random.choice(["zero_to_val", "val_to_zero", "val_to_val"])
    if pattern == "zero_to_val":
        x[:, step_t:] = val_after
    elif pattern == "val_to_zero":
        x[:, :step_t] = val_before
    else:
        x[:, :step_t] = val_before
        x[:, step_t:] = val_after
    return x


def gen_sparse_constant(B, T_raw, N, c_in, device):
    """Sparse constant segments — mostly zero with sustained non-zero windows.

    Models sustained contact forces / contact points that persist for multiple frames."""
    x = torch.zeros(B, T_raw, N, c_in, device=device)
    n_windows = random.randint(1, 3)
    for _ in range(n_windows):
        dur = random.randint(2, max(2, T_raw // 2))
        start = random.randint(0, T_raw - dur)
        val = torch.randn(B, 1, N, c_in, device=device) * 4.0
        # Apply to random subset of points
        point_mask = torch.rand(B, 1, N, 1, device=device) < random.uniform(0.2, 0.7)
        x[:, start:start + dur] += val * point_mask
    return x


def gen_smooth_then_sharp(B, T_raw, N, c_in, device):
    """Smooth trajectory with a sharp discontinuity — collision event mid-trajectory."""
    t = _time_grid(B, T_raw, N, c_in, device)
    # Smooth part
    x0 = torch.randn(B, 1, N, c_in, device=device) * 2.0
    v = torch.randn(B, 1, N, c_in, device=device) * 4.0
    x = x0 + v * t

    # Sharp jump at a random time
    jump_t = random.randint(T_raw // 4, 3 * T_raw // 4)
    jump_val = torch.randn(B, 1, N, c_in, device=device) * 5.0
    v2 = torch.randn(B, 1, N, c_in, device=device) * 4.0  # new velocity after jump
    t2 = _time_grid(B, T_raw - jump_t, N, c_in, device)
    x[:, jump_t:] = x[:, jump_t:jump_t + 1] + jump_val + v2 * t2
    return x


def gen_zero(B, T_raw, N, c_in, device):
    """Pure zeros — the trivial case, common for inactive force channels."""
    return torch.zeros(B, T_raw, N, c_in, device=device)


# Registry: (generator_fn, weight)
# Weights reflect frequency of each pattern in physics simulation data.
TRAJECTORY_GENERATORS = [
    # Smooth continuous — most common for positions/velocities
    (gen_smooth_fourier,       0.12),   # general smooth curves
    (gen_ballistic,            0.08),   # gravity, projectiles
    (gen_sinusoidal,           0.06),   # oscillations, springs
    (gen_damped_oscillation,   0.05),   # friction, damping
    (gen_circular,             0.04),   # rotation, orbits
    (gen_constant_velocity,    0.06),   # uniform linear motion
    (gen_brownian,             0.05),   # stochastic drift
    (gen_exponential_decay,    0.03),   # energy dissipation

    # Sharp / discontinuous — contact forces, collisions
    (gen_bounce,               0.07),   # bounce with velocity reversal
    (gen_smooth_then_sharp,    0.05),   # collision mid-trajectory
    (gen_piecewise_linear,     0.05),   # velocity changes at discrete points
    (gen_piecewise_constant,   0.04),   # discrete state changes

    # Sparse / intermittent — force vectors, contact points
    (gen_sparse_impulse,       0.08),   # short force bursts
    (gen_sparse_constant,      0.06),   # sustained force windows
    (gen_step_function,        0.05),   # abrupt on/off
    (gen_zero,                 0.03),   # inactive channels

    # Other
    (gen_stationary,           0.04),   # objects at rest
    (gen_gaussian_noise,       0.04),   # pure noise baseline
]


def generate_synthetic_batch(B, T_raw, N, c_in, device):
    """Generate a diverse batch of synthetic trajectories.

    Each sample in the batch is drawn from a randomly chosen generator
    according to the weight distribution. This ensures the AE sees all
    signal types during training.

    Args:
        B: Batch size.
        T_raw: Number of raw temporal frames (must be 4k+1).
        N: Number of points (independent trajectories).
        c_in: Channels per point (typically 3).
        device: Torch device.

    Returns:
        (B, T_raw, N, c_in) tensor of synthetic trajectories.
    """
    generators, weights = zip(*TRAJECTORY_GENERATORS)
    total_w = sum(weights)
    probs = [w / total_w for w in weights]

    # Sample generator indices for each item in the batch
    indices = random.choices(range(len(generators)), weights=probs, k=B)

    # Group by generator for efficient batched generation
    from collections import Counter
    counts = Counter(indices)

    parts = []
    order = []  # track original positions for reassembly
    for gen_idx, count in counts.items():
        gen_fn = generators[gen_idx]
        part = gen_fn(count, T_raw, N, c_in, device)
        parts.append(part)
        order.extend([(gen_idx, i) for i in range(count)])

    # Concatenate all parts
    x = torch.cat(parts, dim=0)  # (B, T_raw, N, c_in)

    # Shuffle to restore original sampling order
    # (groups are contiguous above; shuffle to mix them in the batch)
    perm = torch.randperm(B, device=device)
    x = x[perm]

    return x


# ---------------------------------------------------------------------------
# MMD with inverse multiquadric (IMQ) kernel — standard WAE-MMD choice
# ---------------------------------------------------------------------------

def _imq_kernel(x, y, scales=None):
    """Inverse multiquadric kernel: k(x,y) = C / (C + ||x-y||^2).

    Args:
        x: (N, D) samples from one distribution.
        y: (M, D) samples from another distribution.
        scales: List of C values. If None, uses a default range.
    Returns:
        Kernel matrix (N, M).
    """
    if scales is None:
        d = x.shape[1]
        # Heuristic bandwidth set: {0.1, 0.2, 0.5, 1, 2, 5, 10} * d
        scales = [0.1 * d, 0.2 * d, 0.5 * d, float(d), 2.0 * d, 5.0 * d, 10.0 * d]

    # Pairwise squared distances
    xx = (x * x).sum(dim=1, keepdim=True)  # (N, 1)
    yy = (y * y).sum(dim=1, keepdim=True)  # (M, 1)
    dists = xx + yy.t() - 2.0 * x @ y.t()  # (N, M)
    dists = dists.clamp(min=0.0)

    k = torch.zeros_like(dists)
    for c in scales:
        k = k + c / (c + dists)
    return k


def mmd_imq(z, p):
    """Compute MMD^2 between empirical samples z and prior samples p.

    Args:
        z: (N, D) — encoded latent samples.
        p: (N, D) — prior samples (e.g. from N(0, I)).
    Returns:
        Scalar MMD^2 estimate.
    """
    n = z.shape[0]
    # Subsample for efficiency when n is large
    max_n = 2048
    if n > max_n:
        idx = torch.randperm(n, device=z.device)[:max_n]
        z = z[idx]
        p = p[idx]
        n = max_n

    k_zz = _imq_kernel(z, z)
    k_pp = _imq_kernel(p, p)
    k_zp = _imq_kernel(z, p)

    # Unbiased estimator: exclude diagonal for k_zz and k_pp
    diag_zz = k_zz.diag()
    diag_pp = k_pp.diag()
    mmd2 = ((k_zz.sum() - diag_zz.sum()) / (n * (n - 1))
             + (k_pp.sum() - diag_pp.sum()) / (n * (n - 1))
             - 2.0 * k_zp.mean())
    return mmd2


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 0: Train Causal AE")
    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/ae")
    # Model
    parser.add_argument("--c_in", type=int, default=3)
    parser.add_argument("--c_mid", type=int, default=64)
    parser.add_argument("--d_latent", type=int, default=16)
    # Data generation
    parser.add_argument("--n_pts", type=int, default=64,
                        help="Number of independent point trajectories per sample")
    parser.add_argument("--min_k", type=int, default=1,
                        help="Min k for T_raw = 4k+1 (k=1 → T_raw=5)")
    parser.add_argument("--max_k", type=int, default=5,
                        help="Max k for T_raw = 4k+1 (k=5 → T_raw=21)")
    # Training
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_train_steps", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                        choices=["cosine", "constant", "linear"])
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    # Loss weights
    parser.add_argument("--lambda_mmd", type=float, default=0.1)
    parser.add_argument("--lambda_smooth", type=float, default=0.01)
    parser.add_argument("--lambda_interp", type=float, default=0.1)
    # Logging
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--checkpointing_steps", type=int, default=5000)
    parser.add_argument("--report_to", type=str, default="wandb",
                        choices=["tensorboard", "wandb", "none"])
    parser.add_argument("--wandb_project", type=str, default="pv-sim-ae")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    # Mixed precision
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs"),
    )
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to if args.report_to != "none" else None,
        project_config=project_config,
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # --- Model ---
    ae = CausalAE(c_in=args.c_in, c_mid=args.c_mid, d_latent=args.d_latent)
    n_params = sum(p.numel() for p in ae.parameters())
    logger.info(f"CausalAE params: {n_params:,}")

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(ae.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_train_steps, eta_min=0.0,
    ) if args.lr_scheduler == "cosine" else None

    # Prepare with accelerator
    ae, optimizer = accelerator.prepare(ae, optimizer)
    if lr_scheduler is not None:
        lr_scheduler = accelerator.prepare(lr_scheduler)

    weight_dtype = torch.float32
    if args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        weight_dtype = torch.float16

    # --- Trackers ---
    if accelerator.is_main_process:
        init_kwargs = {}
        if args.report_to == "wandb":
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        if args.report_to != "none":
            accelerator.init_trackers(args.wandb_project, config=vars(args),
                                      init_kwargs=init_kwargs)

    # --- Training loop ---
    progress_bar = tqdm(
        range(args.max_train_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    ae.train()
    global_step = 0
    rng = random.Random(args.seed + accelerator.process_index)

    for step in range(args.max_train_steps):
        # --- Generate synthetic data ---
        k = rng.randint(args.min_k, args.max_k)
        T_raw = 4 * k + 1
        # Shape: (B, T_raw, N_pts, c_in) — diverse trajectory types
        x = generate_synthetic_batch(
            args.batch_size, T_raw, args.n_pts, args.c_in,
            device=accelerator.device,
        )

        # --- Forward ---
        with torch.cuda.amp.autocast(dtype=weight_dtype):
            x_hat, z = ae(x)  # x_hat: (B, T_raw, N, 3), z: (B, T, N, 16)

            # 1. MSE reconstruction loss
            loss_mse = F.mse_loss(x_hat, x)

            # 2. MMD regularization (WAE-MMD with IMQ kernel)
            z_flat = z.reshape(-1, args.d_latent).float()
            p = torch.randn_like(z_flat)
            loss_mmd = mmd_imq(z_flat, p)

            # 3. Temporal smoothness
            loss_smooth = (z[:, 1:] - z[:, :-1]).pow(2).mean()

            # 4. Linear interpolation consistency
            alpha = torch.rand(args.batch_size, 1, 1, 1, device=x.device)
            perm = torch.randperm(args.batch_size, device=x.device)
            x2 = x[perm]
            x_interp = alpha * x + (1.0 - alpha) * x2
            _, z_interp = ae(x_interp)
            with torch.no_grad():
                _, z2 = ae(x2)
            z_target = alpha * z.detach() + (1.0 - alpha) * z2
            loss_interp = F.mse_loss(z_interp, z_target)

            loss = (loss_mse
                    + args.lambda_mmd * loss_mmd
                    + args.lambda_smooth * loss_smooth
                    + args.lambda_interp * loss_interp)

        # --- Backward ---
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        if lr_scheduler is not None:
            lr_scheduler.step()

        global_step += 1
        progress_bar.update(1)

        # --- Logging ---
        if global_step % args.logging_steps == 0:
            current_lr = (lr_scheduler.get_last_lr()[0]
                          if lr_scheduler is not None else args.lr)
            log_dict = {
                "loss": loss.item(),
                "loss_mse": loss_mse.item(),
                "loss_mmd": loss_mmd.item(),
                "loss_smooth": loss_smooth.item(),
                "loss_interp": loss_interp.item(),
                "lr": current_lr,
            }
            accelerator.log(log_dict, step=global_step)
            logger.info(
                f"Step {global_step}: loss={loss.item():.4f} "
                f"mse={loss_mse.item():.4f} mmd={loss_mmd.item():.4f} "
                f"smooth={loss_smooth.item():.4f} interp={loss_interp.item():.4f}"
            )

        # --- Checkpoint ---
        if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
            save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            unwrapped = accelerator.unwrap_model(ae)
            unwrapped.save(save_dir)
            logger.info(f"Saved checkpoint to {save_dir}")

    # --- Save final ---
    if accelerator.is_main_process:
        save_dir = os.path.join(args.output_dir, "final")
        unwrapped = accelerator.unwrap_model(ae)
        unwrapped.save(save_dir)
        logger.info(f"Saved final model to {save_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()

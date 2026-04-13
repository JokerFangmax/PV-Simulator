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

from torch.optim.lr_scheduler import LambdaLR

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
#
# Time convention: T_raw frames captured at FPS=12 Hz.
# All generators use REAL time in seconds: t = frame_index / FPS.
# Physical scales for a table-top rigid-body simulation (MOVI-AB style):
#   positions:     ~[-5, 5] m
#   velocities:    ~[-5, 5] m/s
#   accelerations: ~[-15, 15] m/s²  (gravity ≈ 9.8 m/s²)
#   frequencies:   0.1 – 10 Hz

FPS = 12.0   # capture frame rate (frames per second)


def _time_grid(B, T_raw, N, c_in, device):
    """Real time in seconds at FPS=12: shape (B, T_raw, 1, 1), broadcast-ready.

    t[i] = i / FPS  →  t_max = (T_raw-1)/FPS  (e.g. 1.67 s for T_raw=21)
    """
    t = torch.arange(T_raw, device=device).float() / FPS
    return t.view(1, T_raw, 1, 1).expand(B, T_raw, 1, 1)


def gen_gaussian_noise(B, T_raw, N, c_in, device):
    """Pure i.i.d. Gaussian noise — baseline, stress-tests reconstruction.

    Range: ~[-15, 15]
      std ~ U(0.01, 7.5), worst case std=7.5 → 2·std = 15.
    """
    std = torch.empty(B, 1, 1, 1, device=device).uniform_(0.01, 7.5)
    return torch.randn(B, T_raw, N, c_in, device=device) * std


def gen_stationary(B, T_raw, N, c_in, device):
    """Near-constant signal with tiny sensor jitter — objects at rest.

    Range: ~[-15, 15]
      offset = N(0,7.5) → ±15; jitter = N(0,0.01) → ±0.02 (negligible).
    """
    offset = torch.randn(B, 1, N, c_in, device=device) * 7.5   # ±7.5 m position
    jitter = torch.randn(B, T_raw, N, c_in, device=device) * 0.01  # 1 cm jitter
    return offset + jitter


def gen_constant_velocity(B, T_raw, N, c_in, device):
    """Linear trajectories x(t) = x0 + v*t — uniform motion.

    Range: ~[-27, 27]
      x0 = N(0,5) → ±10; v·t_max = N(0,5)·1.67 → ±16.7; total ±26.7.
    """
    t = _time_grid(B, T_raw, N, c_in, device)
    x0 = torch.randn(B, 1, N, c_in, device=device) * 5.0   # ±5 m
    v = torch.randn(B, 1, N, c_in, device=device) * 5.0    # ±5 m/s
    return x0 + v * t


def gen_ballistic(B, T_raw, N, c_in, device):
    """Parabolic trajectories x(t) = x0 + v0*t + 0.5*a*t² — gravity/throws.

    Range: ~[-41, 41]
      x0=N(0,5)→±10; v0·t_max→±16.7; 0.5·a·t_max²=0.5·N(0,5)·1.67²→±13.9;
      total ±40.6.
    """
    t = _time_grid(B, T_raw, N, c_in, device)
    x0 = torch.randn(B, 1, N, c_in, device=device) * 5.0   # ±5 m
    v0 = torch.randn(B, 1, N, c_in, device=device) * 5.0   # ±5 m/s
    a  = torch.randn(B, 1, N, c_in, device=device) * 5.0   # ±5 m/s² (gravity ~9.8)
    return x0 + v0 * t + 0.5 * a * t ** 2


def gen_sinusoidal(B, T_raw, N, c_in, device):
    """Sinusoidal oscillation — springs, vibrations, periodic motion.

    Range: ~[-20, 20]
      offset = N(0,5) → ±10; amp·sin = N(0,5)·[-1,1] → ±10; total ±20.
    """
    t = _time_grid(B, T_raw, N, c_in, device)
    amp = torch.randn(B, 1, N, c_in, device=device) * 5.0   # ±5 m
    freq = torch.empty(B, 1, N, c_in, device=device).uniform_(0.1, 10.0)  # 0.1–10 Hz
    phase = torch.empty(B, 1, N, c_in, device=device).uniform_(0, 2 * math.pi)
    offset = torch.randn(B, 1, N, c_in, device=device) * 5.0   # ±5 m
    return offset + amp * torch.sin(2 * math.pi * freq * t + phase)


def gen_damped_oscillation(B, T_raw, N, c_in, device):
    """Exponentially damped sinusoid — friction, energy dissipation.

    Range: ~[-20, 20]
      offset = N(0,5) → ±10; amp·exp(−decay·t)·sin ≤ |amp| at t=0 → ±10;
      exp decay only reduces amplitude over time; total ±20.
    """
    t = _time_grid(B, T_raw, N, c_in, device)
    amp = torch.randn(B, 1, N, c_in, device=device) * 5.0   # ±5 m
    freq = torch.empty(B, 1, N, c_in, device=device).uniform_(0.1, 10.0)  # 0.1–10 Hz
    phase = torch.empty(B, 1, N, c_in, device=device).uniform_(0, 2 * math.pi)
    decay = torch.empty(B, 1, N, c_in, device=device).uniform_(0.5, 6.0) # half-life 0.12–1.4 s
    offset = torch.randn(B, 1, N, c_in, device=device) * 5.0   # ±5 m
    return offset + amp * torch.exp(-decay * t) * torch.sin(2 * math.pi * freq * t + phase)


def gen_smooth_fourier(B, T_raw, N, c_in, device):
    """Random smooth curves via low-frequency Fourier basis — general smooth motion.

    Harmonics at k Hz (k=1..n_harmonics), amplitude falls as 1/k.

    Range: ~[-24, 24]  (n_harmonics up to 5)
      base = N(0,5) → ±10; k=1: N(0,3)→±6; k=2: N(0,1.5)→±3; k=3: N(0,1)→±2;
      k=4: N(0,0.75)→±1.5; k=5: N(0,0.6)→±1.2; total ±23.7.
    """
    t = _time_grid(B, T_raw, N, c_in, device)  # real seconds
    n_harmonics = random.randint(2, 5)
    x = torch.randn(B, 1, N, c_in, device=device) * 5.0   # ±5 m
    for k in range(1, n_harmonics + 1):
        amp = torch.randn(B, 1, N, c_in, device=device) * (3 / k)  # 3 m / k
        phase = torch.empty(B, 1, N, c_in, device=device).uniform_(0, 2 * math.pi)
        x = x + amp * torch.sin(2 * math.pi * k * t + phase)
    return x


def gen_circular(B, T_raw, N, c_in, device):
    """Circular/spiral motion — rotation, orbits. Pairs channels for rotation.

    Range: ~[-26, 26]
      radius = N(0,3) → ±6; growth·t_max = N(0,3)·1.67 → ±10; r = radius+growth·t → ±16;
      r·cos/sin → ±16; center = N(0,5) → ±10; total ±26.
    """
    t = _time_grid(B, T_raw, N, c_in, device)  # real seconds
    radius = torch.randn(B, 1, N, 1, device=device) * 3   # ±3 m
    freq   = torch.empty(B, 1, N, 1, device=device).uniform_(0.1, 10.0)    # 0.3–2 Hz
    phase  = torch.empty(B, 1, N, 1, device=device).uniform_(0, 2 * math.pi)
    center = torch.randn(B, 1, N, c_in, device=device) * 5              # ±5 m
    growth = torch.randn(B, 1, N, 1, device=device) * 3   # ±3 m/s radial
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
    """Brownian motion / random walk — cumulative small steps.

    step_std is per frame (1/FPS s), so diffusion = step_std * sqrt(FPS) m/sqrt(s).

    Range: ~[-20, 20]
      start = N(0,8.5) → ±17; step_std ~ U(0.005, 0.3), worst case 0.3;
      cumulative drift after 20 steps: std = 0.3·√20 ≈ 1.34 → 2·std ≈ 2.7;
      total ±19.7.
    """
    step_std = torch.empty(B, 1, 1, 1, device=device).uniform_(0.005, 0.3)  # 5–300 mm/frame
    steps = torch.randn(B, T_raw, N, c_in, device=device) * step_std
    steps[:, 0] = torch.randn(B, N, c_in, device=device) * 8.5   # ±8.5 m start
    return steps.cumsum(dim=1)


def gen_exponential_decay(B, T_raw, N, c_in, device):
    """Exponential decay/growth — energy dissipation, temperature cooling.

    Range: ~[-25, 25]
      offset = N(0,0.3) → ±0.6; amp = N(0,1) → ±2;
      growth case (sign=+1): exp(rate·t_max) ≤ exp(1.5·1.67) = exp(2.5) ≈ 12.2,
        amp·exp → ±24.4; total ±25.
      decay case (sign=−1): exp(−rate·t) ≤ 1 → amp·exp ≤ ±2; total ±2.6.
    """
    t = _time_grid(B, T_raw, N, c_in, device)
    amp  = torch.randn(B, 1, N, c_in, device=device) * 1.0       # ±1.0 m amplitude
    rate = torch.empty(B, 1, N, c_in, device=device).uniform_(0.3, 1.5)  # 1/s
    offset = torch.randn(B, 1, N, c_in, device=device) * 0.3
    sign = torch.sign(torch.randn(B, 1, N, c_in, device=device))
    return offset + amp * torch.exp(sign * rate * t)


def gen_piecewise_constant(B, T_raw, N, c_in, device):
    """Piecewise constant — force on/off, discrete state changes.

    Range: ~[-15, 15]
      Each segment: val = N(0,7.5) → ±15; no accumulation across segments.
    """
    n_segments = random.randint(2, 5)
    x = torch.zeros(B, T_raw, N, c_in, device=device)
    # Random change points
    change_points = sorted(random.sample(range(1, T_raw), min(n_segments - 1, T_raw - 1)))
    change_points = [0] + change_points + [T_raw]
    for i in range(len(change_points) - 1):
        t_start, t_end = change_points[i], change_points[i + 1]
        val = torch.randn(B, 1, N, c_in, device=device) * 7.5  # ±7.5 (m or N)
        x[:, t_start:t_end] = val
    return x


def gen_piecewise_linear(B, T_raw, N, c_in, device):
    """Piecewise linear — velocity changes at discrete points (collisions).

    Range: ~[-15, 15]
      vals at change points = N(0,7.5) → ±15; linear interpolation stays within
      the convex hull of endpoint values; no accumulation.
    """
    n_segments = random.randint(2, 5)
    change_points = sorted(random.sample(range(1, T_raw), min(n_segments - 1, T_raw - 1)))
    change_points = [0] + change_points + [T_raw]

    x = torch.zeros(B, T_raw, N, c_in, device=device)
    # Values at each change point
    vals = [torch.randn(B, 1, N, c_in, device=device) * 5.0
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
    """Bounce / collision pattern — smooth parabolas with sudden velocity reversals.

    Uses real time (seconds) so v0 is in m/s and a is in m/s².

    Range: ~[-41, 41]
      First segment is ballistic: x0=N(0,5)→±10, v0·t→±16.7, 0.5·a·t²→±13.9;
      total ±40.6. Post-bounce velocity is multiplied by restitution ∈ [0.4, 0.95]
      and reversed, so subsequent segments are damped; position resets to the
      previous segment end and does not accumulate unboundedly.
    """
    t = torch.arange(T_raw, device=device).float() / FPS   # real seconds
    n_bounces = random.randint(1, 3)
    bounce_times = sorted([random.uniform(0.15, 0.85) for _ in range(n_bounces)])

    x0 = torch.randn(B, 1, N, c_in, device=device) * 5   # 5 m
    v0 = torch.randn(B, 1, N, c_in, device=device) * 5.0   # ±5 m/s
    a  = torch.randn(B, 1, N, c_in, device=device) * 5.0   # ±5 m/s²
    restitution = torch.empty(B, 1, N, c_in, device=device).uniform_(0.4, 0.95)

    x = torch.zeros(B, T_raw, N, c_in, device=device)
    seg_start = 0
    cur_x0, cur_v0 = x0, v0

    t_max = t[-1].item()
    for bounce_t in bounce_times + [1.0]:
        seg_end = int(bounce_t * (T_raw - 1)) + 1
        seg_end = min(seg_end, T_raw)
        if seg_start >= seg_end:
            continue
        dt = t[seg_start:seg_end].view(1, -1, 1, 1) - t[seg_start].item()
        x[:, seg_start:seg_end] = cur_x0 + cur_v0 * dt + 0.5 * a * dt ** 2
        # Compute velocity at end of segment for next segment
        seg_dur = t[min(seg_end, T_raw) - 1].item() - t[seg_start].item()
        cur_v0 = -(cur_v0 + a * seg_dur) * restitution
        cur_x0 = x[:, seg_end - 1:seg_end]
        seg_start = seg_end

    return x


def gen_sparse_impulse(B, T_raw, N, c_in, device):
    """Sparse impulses — mostly zero with occasional short bursts. Models contact forces.

    Range: ~[-7.5, 7.5] typical; ~[-22.5, 22.5] worst case
      impulse_val = N(0,3.75) → ±7.5 per impulse; up to 3 additive impulses if they
      overlap at the same frame → ±22.5; background is 0.
    """
    x = torch.zeros(B, T_raw, N, c_in, device=device)
    # 1-3 impulse windows per trajectory
    n_impulses = random.randint(1, 3)
    for _ in range(n_impulses):
        # Random start and duration (1-3 frames)
        dur = random.randint(1, min(3, T_raw))
        start = random.randint(0, T_raw - dur)
        # Only affect a random subset of points (~30-80%)
        point_mask = torch.rand(B, 1, N, 1, device=device) < random.uniform(0.3, 0.8)
        impulse_val = torch.randn(B, dur, N, c_in, device=device) * 3.75  # ±3.75 N
        x[:, start:start + dur] += impulse_val * point_mask
    return x


def gen_step_function(B, T_raw, N, c_in, device):
    """Step function — abrupt on/off transitions. Models contact begin/end.

    Range: ~[-15, 15]
      val_before / val_after = N(0,7.5) → ±15; background is 0.
    """
    x = torch.zeros(B, T_raw, N, c_in, device=device)
    # Random step time
    step_t = random.randint(1, T_raw - 1)
    val_before = torch.randn(B, 1, N, c_in, device=device) * 7.5
    val_after  = torch.randn(B, 1, N, c_in, device=device) * 7.5
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

    Models sustained contact forces / contact points that persist for multiple frames.

    Range: ~[-7.5, 7.5] typical; ~[-22.5, 22.5] worst case
      val = N(0,3.75) → ±7.5 per window; up to 3 windows added together → ±22.5;
      background is 0.
    """
    x = torch.zeros(B, T_raw, N, c_in, device=device)
    n_windows = random.randint(1, 3)
    for _ in range(n_windows):
        dur = random.randint(2, max(2, T_raw // 2))
        start = random.randint(0, T_raw - dur)
        val = torch.randn(B, 1, N, c_in, device=device) * 3.75
        point_mask = torch.rand(B, 1, N, 1, device=device) < random.uniform(0.2, 0.7)
        x[:, start:start + dur] += val * point_mask
    return x


def gen_smooth_then_sharp(B, T_raw, N, c_in, device):
    """Smooth trajectory with a sharp discontinuity — collision event mid-trajectory.

    Range: ~[-35, 35]
      Pre-jump: x0=N(0,3)→±6, v·t_max→±10; value at jump_t ≈ ±15.
      Post-jump adds jump_val=N(0,3)→±6 and v2·t2_max→±10;
      total ±(15 + 10 + 10) ≈ ±35.
    """
    t = _time_grid(B, T_raw, N, c_in, device)   # real seconds
    x0 = torch.randn(B, 1, N, c_in, device=device) * 5.0    # ±5 m
    v  = torch.randn(B, 1, N, c_in, device=device) * 3.0    # ±3 m/s
    x  = x0 + v * t

    # Sharp jump at a random time
    jump_t = random.randint(T_raw // 4, 3 * T_raw // 4)
    jump_val = torch.randn(B, 1, N, c_in, device=device) * 5.0  # ±5 m impulse
    v2 = torch.randn(B, 1, N, c_in, device=device) * 3.0        # ±3 m/s new velocity
    t2 = _time_grid(B, T_raw - jump_t, N, c_in, device)         # real seconds from jump
    x[:, jump_t:] = x[:, jump_t:jump_t + 1] + jump_val + v2 * t2
    return x


def gen_zero(B, T_raw, N, c_in, device):
    """Pure zeros — the trivial case, common for inactive force channels.

    Range: [0, 0]
    """
    return torch.zeros(B, T_raw, N, c_in, device=device)


# Base generators suitable as the "moving" phase in gen_sudden_stop.
# Excludes sparse/zero generators — we want clearly visible motion before the stop.
_SUDDEN_STOP_BASE_GENERATORS = [
    gen_smooth_fourier,
    gen_ballistic,
    gen_sinusoidal,
    gen_damped_oscillation,
    gen_circular,
    gen_constant_velocity,
    gen_brownian,
    gen_bounce,
    gen_piecewise_linear,
]


def gen_sudden_stop(B, T_raw, N, c_in, device):
    """Normal motion followed by an abrupt freeze at a random time.

    Models an object that suddenly stops (e.g. wall collision, catching, landing).
    A smooth/physics base trajectory is generated, then all values are held constant
    at the frozen frame for the remainder of the sequence.

    Range: inherited from the base generator (~[-41, 41] worst case from ballistic).
      stop_t ~ U[T_raw//4, 3*T_raw//4]; post-stop signal = constant.
    """
    base_fn = random.choice(_SUDDEN_STOP_BASE_GENERATORS)
    x = base_fn(B, T_raw, N, c_in, device)

    # Freeze at stop_t: x[:, stop_t:] = x[:, stop_t-1:stop_t]
    stop_t = random.randint(T_raw // 4, 3 * T_raw // 4)
    x[:, stop_t:] = x[:, stop_t - 1 : stop_t]
    return x


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
    (gen_sudden_stop,          0.05),   # normal motion then abrupt freeze
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
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Checkpoint dir to resume from (loads model weights; optimizer/scheduler reset)")
    parser.add_argument("--start_step", type=int, default=0,
                        help="Step to treat as the starting point (used with --resume_from). "
                             "--max_train_steps is the TOTAL target, so only "
                             "(max_train_steps - start_step) more steps will run.")
    # Model
    parser.add_argument("--c_in", type=int, default=3)
    parser.add_argument("--c_mid", type=int, default=64)
    parser.add_argument("--d_latent", type=int, default=16)
    parser.add_argument("--n_res_blocks", type=int, default=1,
                        help="ResidualBlock1d stacks per level (depth knob). "
                             "Default 1 reproduces the original AE architecture.")
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
                        choices=["cosine", "constant", "linear", "cosine_floor"])
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_total_steps", type=int, default=None,
                        help="Total steps for the LR schedule cycle. Defaults to max_train_steps. "
                             "Set this to the FULL planned training length when doing a partial run "
                             "that will later be resumed, so the schedule matches across both runs.")
    parser.add_argument("--lr_decay_steps", type=int, default=None,
                        help="(cosine_floor only) Steps over which cosine decays from lr to "
                             "lr*lr_min_ratio. After this, LR holds at the floor. "
                             "Defaults to lr_total_steps (i.e. standard cosine).")
    parser.add_argument("--lr_min_ratio", type=float, default=0.01,
                        help="(cosine_floor only) Floor LR as a fraction of peak LR. "
                             "E.g. 0.01 means floor = lr * 0.01.")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Max gradient norm (0 to disable)")
    parser.add_argument("--seed", type=int, default=42)
    # Loss weights
    parser.add_argument("--lambda_mmd", type=float, default=0.1)
    parser.add_argument("--lambda_smooth", type=float, default=0.0)
    parser.add_argument("--lambda_interp", type=float, default=0.1)
    parser.add_argument("--loss_type", type=str, default="mse",
                        choices=["mse", "mae", "huber"],
                        help="Reconstruction loss used as the training objective. "
                             "MSE and MAE are always logged; huber uses smooth_l1 with "
                             "beta=--huber_delta and is logged as loss_huber.")
    parser.add_argument("--huber_delta", type=float, default=0.1,
                        help="Beta for smooth L1 (Huber) — quadratic below |x|<delta, "
                             "linear above. Only used when --loss_type huber.")
    # Logging
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--checkpointing_steps", type=int, default=5000)
    parser.add_argument("--report_to", type=str, default="wandb",
                        choices=["tensorboard", "wandb", "none"])
    parser.add_argument("--wandb_project", type=str, default="pv-sim-ae")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    # Validation
    parser.add_argument("--val_every", type=int, default=0,
                        help="Run a validation step every N training steps (0 disables). "
                             "Logs val/mse, val/mmd, val/interp, and a reconstruction image to wandb.")
    parser.add_argument("--val_batch_size", type=int, default=256,
                        help="Batch size for the validation synthetic batch.")
    parser.add_argument("--val_t_raw", type=int, default=21,
                        help="T_raw for validation (fixed, must be 4k+1).")
    parser.add_argument("--val_n_points", type=int, default=16,
                        help="N points per sample for the reconstruction figure.")
    parser.add_argument("--val_seed", type=int, default=123,
                        help="Seed for the reconstruction figure generators.")
    # Mixed precision
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_validation(ae, accelerator, args, global_step, weight_dtype):
    """Evaluate on a fresh synthetic batch and log metrics + reconstruction image.

    Called only from the main process. Assumes `ae` is the accelerator-wrapped
    model and that `accelerator.is_main_process` is True.
    """
    unwrapped = accelerator.unwrap_model(ae)
    was_training = unwrapped.training
    unwrapped.eval()

    device = accelerator.device
    T_raw = args.val_t_raw
    B = args.val_batch_size

    with torch.amp.autocast("cuda", dtype=weight_dtype):
        x = generate_synthetic_batch(B, T_raw, args.n_pts, args.c_in, device=device)
        x_hat, z = unwrapped(x)
        loss_mse = F.mse_loss(x_hat, x)
        loss_mae = F.l1_loss(x_hat, x)
        loss_huber = F.smooth_l1_loss(x_hat, x, beta=args.huber_delta)
        z_flat = z.reshape(-1, args.d_latent).float()
        p = torch.randn_like(z_flat)
        loss_mmd = mmd_imq(z_flat, p)
        alpha = torch.rand(B, 1, 1, 1, device=device)
        perm = torch.randperm(B, device=device)
        x2 = x[perm]
        x_interp = alpha * x + (1.0 - alpha) * x2
        _, z_interp = unwrapped(x_interp)
        _, z2 = unwrapped(x2)
        z_target = alpha * z + (1.0 - alpha) * z2
        loss_interp = F.mse_loss(z_interp, z_target)

    val_metrics = {
        "val/mse":    float(loss_mse.item()),
        "val/mae":    float(loss_mae.item()),
        "val/huber":  float(loss_huber.item()),
        "val/mmd":    float(loss_mmd.item()),
        "val/interp": float(loss_interp.item()),
    }

    # Build reconstruction figure and log as wandb image.
    try:
        from scripts.pv_simulator.plot_ae_reconstruction import build_reconstruction_figure
        import matplotlib.pyplot as plt

        fig, fig_mse = build_reconstruction_figure(
            unwrapped, device,
            t_raw=args.val_t_raw,
            n_points=args.val_n_points,
            c_in=args.c_in,
            component=0,
            seed=args.val_seed,
        )
        val_metrics["val/recon_figure_mse"] = float(fig_mse)

        if args.report_to == "wandb":
            try:
                import wandb
                tracker = accelerator.get_tracker("wandb", unwrap=True)
                if tracker is not None:
                    tracker.log(
                        {"val/reconstruction": wandb.Image(fig),
                         **val_metrics},
                        step=global_step,
                    )
            except Exception as e:
                logger.warning(f"[val] wandb image log failed: {e}")
                accelerator.log(val_metrics, step=global_step)
        else:
            accelerator.log(val_metrics, step=global_step)
        plt.close(fig)
    except Exception as e:
        logger.warning(f"[val] reconstruction figure failed: {e}")
        accelerator.log(val_metrics, step=global_step)
    logger.info(
        f"[val @ {global_step}] mse={val_metrics['val/mse']:.4f} "
        f"mae={val_metrics['val/mae']:.4f} "
        f"mmd={val_metrics['val/mmd']:.4f} interp={val_metrics['val/interp']:.4f}"
    )

    if was_training:
        unwrapped.train()


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
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        ae = CausalAE.load(args.resume_from)
    else:
        ae = CausalAE(c_in=args.c_in, c_mid=args.c_mid, d_latent=args.d_latent,
                      n_res_blocks=args.n_res_blocks)
    n_params = sum(p.numel() for p in ae.parameters())
    logger.info(f"CausalAE params: {n_params:,}")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(ae.parameters(), lr=args.lr, weight_decay=1e-4)

    # Prepare with accelerator before creating scheduler (scheduler must reference prepared optimizer)
    ae, optimizer = accelerator.prepare(ae, optimizer)

    # --- Scheduler (created after prepare so it references the wrapped optimizer) ---
    # lr_lambda receives a *relative* step (0-indexed from resume point).
    # We convert to *absolute* step by adding start_step so the cosine curve
    # continues from the right position rather than restarting from the peak.
    # lr_total_steps sets the cosine cycle length independently of max_train_steps,
    # so a screening run (max=20k) and its continuation (max=50k) share the same curve.
    def _make_scheduler(optimizer):
        warmup = args.lr_warmup_steps
        total = args.lr_total_steps if args.lr_total_steps is not None else args.max_train_steps
        start = args.start_step
        sched = args.lr_scheduler
        if sched == "cosine":
            def lr_lambda(rel_step):
                step = rel_step + start  # absolute step
                if step < warmup:
                    return float(step) / max(1, warmup)
                progress = float(step - warmup) / max(1, total - warmup)
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        elif sched == "cosine_floor":
            decay_steps = args.lr_decay_steps if args.lr_decay_steps is not None else total
            min_ratio = args.lr_min_ratio
            def lr_lambda(rel_step):
                step = rel_step + start
                if step < warmup:
                    return float(step) / max(1, warmup)
                if step >= decay_steps:
                    return min_ratio
                progress = float(step - warmup) / max(1, decay_steps - warmup)
                return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        elif sched == "linear":
            def lr_lambda(rel_step):
                step = rel_step + start
                if step < warmup:
                    return float(step) / max(1, warmup)
                return max(0.0, float(total - step) / max(1, total - warmup))
        else:  # constant
            def lr_lambda(rel_step):
                step = rel_step + start
                if step < warmup:
                    return float(step) / max(1, warmup)
                return 1.0
        return LambdaLR(optimizer, lr_lambda)

    lr_scheduler = accelerator.prepare(_make_scheduler(optimizer))

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

    remaining_steps = args.max_train_steps - args.start_step
    logger.info(f"Training from step {args.start_step} to {args.max_train_steps} "
                f"({remaining_steps} steps remaining)")

    # --- Training loop ---
    progress_bar = tqdm(
        range(remaining_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    ae.train()
    global_step = args.start_step
    rng = random.Random(args.seed + accelerator.process_index)

    # EMA accumulators for smooth logging (alpha=0.98 ≈ 50-step window)
    # Initialized on first step to avoid zero-bias at resume.
    _ema_alpha = 0.98
    ema = {}

    def _update_ema(key, val):
        if key not in ema:
            ema[key] = val
        else:
            ema[key] = _ema_alpha * ema[key] + (1 - _ema_alpha) * val

    for step in range(remaining_steps):
        # --- Generate synthetic data ---
        k = rng.randint(args.min_k, args.max_k)
        T_raw = 4 * k + 1
        # Shape: (B, T_raw, N_pts, c_in) — diverse trajectory types
        x = generate_synthetic_batch(
            args.batch_size, T_raw, args.n_pts, args.c_in,
            device=accelerator.device,
        )

        # --- Forward ---
        with torch.amp.autocast("cuda", dtype=weight_dtype):
            x_hat, z = ae(x)  # x_hat: (B, T_raw, N, 3), z: (B, T, N, 16)

            # 1. Reconstruction — MSE, MAE, and Huber are all computed each
            #    step so any of the three can be compared post-hoc.
            #    Optimizer sees whichever --loss_type names.
            loss_mse   = F.mse_loss(x_hat, x)
            loss_mae   = F.l1_loss(x_hat, x)
            loss_huber = F.smooth_l1_loss(x_hat, x, beta=args.huber_delta)
            loss_recon = {
                "mse":   loss_mse,
                "mae":   loss_mae,
                "huber": loss_huber,
            }[args.loss_type]

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

            loss = (loss_recon
                    + args.lambda_mmd * loss_mmd
                    + args.lambda_smooth * loss_smooth
                    + args.lambda_interp * loss_interp)

        # --- Backward ---
        accelerator.backward(loss)
        if args.grad_clip > 0:
            accelerator.clip_grad_norm_(ae.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        global_step += 1
        progress_bar.update(1)

        # Update EMA every step
        _update_ema("loss",        loss.item())
        _update_ema("loss_mse",    loss_mse.item())
        _update_ema("loss_mae",    loss_mae.item())
        _update_ema("loss_huber",  loss_huber.item())
        _update_ema("loss_mmd",    loss_mmd.item())
        _update_ema("loss_smooth", loss_smooth.item())
        _update_ema("loss_interp", loss_interp.item())

        # --- Logging ---
        if global_step % args.logging_steps == 0:
            current_lr = lr_scheduler.get_last_lr()[0]
            log_dict = {
                "loss":        ema["loss"],
                "loss_mse":    ema["loss_mse"],
                "loss_mae":    ema["loss_mae"],
                "loss_huber":  ema["loss_huber"],
                "loss_mmd":    ema["loss_mmd"],
                "loss_smooth": ema["loss_smooth"],
                "loss_interp": ema["loss_interp"],
                "lr": current_lr,
            }
            accelerator.log(log_dict, step=global_step)
            logger.info(
                f"Step {global_step}: loss={ema['loss']:.4f} "
                f"mse={ema['loss_mse']:.4f} mae={ema['loss_mae']:.4f} "
                f"mmd={ema['loss_mmd']:.4f} "
                f"smooth={ema['loss_smooth']:.4f} interp={ema['loss_interp']:.4f}"
            )

        # --- Checkpoint ---
        if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
            save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            unwrapped = accelerator.unwrap_model(ae)
            unwrapped.save(save_dir)
            logger.info(f"Saved checkpoint to {save_dir}")

        # --- Validation ---
        if (args.val_every > 0
                and global_step % args.val_every == 0
                and accelerator.is_main_process):
            run_validation(ae, accelerator, args, global_step, weight_dtype)

    # --- Save final ---
    if accelerator.is_main_process:
        save_dir = os.path.join(args.output_dir, "final")
        unwrapped = accelerator.unwrap_model(ae)
        unwrapped.save(save_dir)
        logger.info(f"Saved final model to {save_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()

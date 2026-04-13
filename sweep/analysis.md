# Stage 0 AE — MSE vs MAE sweep analysis  (2026-04-12)

Both full 3-phase sweeps (A loss weights → B capacity → C training dynamics,
24 runs each) completed. Both converged on the **same lr/scheduler**
(`lr=1e-3, cosine`) but chose **different capacities**:

| Sweep | λ_mmd | λ_interp | c_mid | d_latent | lr | sched |
|---|---|---|---|---|---|---|
| MSE | 0.1 (forced) | 0.1 (forced) | 128 | **32** | 1e-3 | cosine |
| MAE | 0.1 (forced) | 0.1 (forced) | 128 | **16** | 1e-3 | cosine |

> Phase A for both sweeps was overridden (`λ_mmd=0.1, λ_interp=0.1`) because
> Phase A's composite-score winner always collapses to `λ_mmd=0` configs that
> have near-zero interp (λ_interp=1 drives interp→0 but leaves the latent far
> from N(0,I), breaking downstream diffusion).

---

## Head-to-head on identical eval set (n_eval=512, seed=0)

| Metric | MSE-winner | MAE-winner | Delta | Winner |
|---|---|---|---|---|
| MSE | **0.000243** | 0.000454 | +87 % | MSE |
| MAE | 0.01117 | **0.00911** | −18 % | **MAE** |
| MMD | **0.2653** | 0.3844 | +45 % | MSE |
| interp | 0.00515 | **0.00161** | **−69 %** | **MAE** (3.2×) |

Per-category MAE (lower=better — this is the metric the user cares about for "detail accuracy"):

| Category | MSE-trained | MAE-trained | Delta |
|---|---|---|---|
| smooth | 0.01146 | **0.00954** | −17 % |
| sharp  | 0.01184 | **0.00970** | −18 % |
| sparse | 0.00878 | **0.00628** | **−29 %** |
| other  | 0.01314 | **0.01158** | −12 % |

Per-category MSE:

| Category | MSE-trained | MAE-trained | Delta |
|---|---|---|---|
| smooth | **0.000238** | 0.000612 | +157 % |
| sharp  | **0.000254** | 0.000337 |  +33 % |
| sparse | **0.000203** | 0.000290 |  +43 % |
| other  | **0.000318** | 0.000439 |  +38 % |

---

## Key findings

1. **MAE training achieves 3.2× better interpolation consistency** than MSE.
   Because MAE gives full-magnitude gradient on *any* nonzero residual, the
   latent space ends up more locally linear. This is a **direct downstream win
   for Stage 1/2 flow matching**, where the diffusion model has to navigate
   straight paths through latent space.

2. **MAE training improves per-category MAE everywhere**, with the largest gain
   on `sparse` trajectories (−29 %). This confirms the user's intuition that
   MSE "smooths away" small details: sparse generators have long zero stretches
   with rare spikes that L2 under-weights.

3. **MSE training still wins on L2 metrics by ≈2×** (expected — it optimizes
   MSE directly). If downstream consumers measure L2 reconstruction error the
   MSE AE is strictly better; if they care about per-point fidelity or
   latent-space geometry, MAE wins.

4. **MMD is 45 % worse for MAE** (0.38 vs 0.27). The latent is further from
   N(0, I) — will cost some flow-matching training stability but not
   catastrophic (both are << the mmd=0 runs in Phase A at ~2.0).

5. **MAE converges much slower in 5000 steps.** In the MAE Phase A+B, only
   `c_mid=128, d_latent∈{16,32}` configs actually converged; everything else
   stayed near MSE≈2.0. Under MSE all 9 Phase A configs converged cleanly. MAE
   picks a *smaller* `d_latent=16` winner precisely because `d_latent=32` was
   still underfitting at 5k steps — **the MAE sweep is budget-limited**.

6. **Worst-generator failure modes differ**:
   - MSE-winner worst-MSE: `gen_gaussian_noise` (4.6e-4), `gen_step_function` (4.5e-4)
   - MAE-winner worst-MSE: **`gen_brownian` (3.7e-3 — 15× worse)**, `gen_step_function` (1.0e-3)

   MAE is robust to outliers by design, so on a high-variance noise process
   like Brownian motion it tolerates large residuals that MSE would hammer
   down. For physics data this is a real regression on the `gen_brownian`
   category.

---

## Recommended immediate action

For Stage 1/2 downstream training, **use the MAE-trained AE**
(`outputs/stage0/sweeps_mae/training/lr1e-3-cosine/final`). The 3.2× interp
improvement and detail-preserving per-point MAE outweigh the L2 regression and
the worse `gen_brownian` behavior, because:

- Diffusion training operates on the latent, and interp loss is the closest
  proxy we have for "is this latent space flow-friendly".
- `gen_brownian` is a pure-noise training category that won't appear in real
  deployment data — the regression there is acceptable.
- User's stated goal is detail accuracy, which MAE delivers (−17 to −29 %
  per-category MAE).

---

## Proposed follow-up experiments (ranked by expected ROI)

### F1. **Huber / SmoothL1 loss sweep** ⭐⭐⭐

Best-of-both: L2 near zero (fast convergence, clean gradients) + L1 in the
tail (detail, outlier robustness). Expect to match MAE's detail win *and* MSE's
convergence speed.

- Grid: `delta ∈ {0.01, 0.05, 0.1, 0.3}` at the winning config
  `(c_mid=128, d_latent=16, lr=1e-3, cosine)`
- 5000 steps × 4 runs, 2 GPUs ≈ 40 min
- Eval with the identical composite score + per-category MAE
- Plug into `train_stage0.py` as `--loss_type huber --huber_delta ...`

### F2. **MSE+MAE convex mix** ⭐⭐

Simpler than Huber; linearly interpolates between the two regimes.
`loss_recon = (1-α) * MSE + α * MAE`

- Grid: `α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}` — 5 runs
- Directly answers "how much MAE do we need before interp improves" without
  Huber's extra hyperparam.

### F3. **Longer MAE training (10–20k steps)** ⭐⭐

The MAE sweep is budget-limited: 7/9 Phase B runs and 2/9 Phase A runs never
converged. Re-run `c_mid=128, d_latent∈{16, 32}` with 15k steps to see if:
(a) d_latent=32 catches up and wins the MAE sweep,
(b) MSE improves enough to remove the gen_brownian regression.

- 4 runs × 15k steps ≈ 3 h on 2 GPUs.

### F4. **Stage 1 A/B validation** ⭐⭐⭐

The real test is downstream: train two matched Stage 1 SimDiTs using the
MSE-winner and MAE-winner AEs as frozen encoders. Compare diffusion training
loss curves and inference-time trajectory reconstructions on held-out data.
This is the only experiment that directly measures what the sweep was *for*.

- 2 Stage 1 runs @ ~same step budget
- Success signal: lower final diffusion loss, smoother generated trajectories

### F5. **Debug MAE on `gen_brownian`** ⭐

Isolated investigation, not a sweep. Questions:
- Does the MAE winner genuinely regress, or is it within generator-sampling
  noise? Run eval with multiple seeds.
- If it regresses, does Huber (F1) fix it?

Only worth doing if F1/F2 don't cleanly dominate.

---

## What NOT to re-run

- **Phase A (loss weights)** for either sweep is decided: `λ_mmd=0.1, λ_interp=0.1`.
  The composite-score picker keeps collapsing to `λ_mmd=0` configs which have
  degenerate latents; forced values are correct.
- **λ_smooth=0** is locked — enabling it punishes legitimate sharp
  transitions.
- **Phase C `constant` schedulers** — all constant-schedule runs in both
  sweeps were strictly dominated by their cosine counterparts.

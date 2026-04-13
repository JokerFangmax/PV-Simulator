# Stage 0 AE — Sweep Tracker  (loss_type=mse, updated: 2026-04-12 17:10:14)


## Plan

- **Phase A** (loss weights, `pv-ae-stage-f-loss`): 9 runs, λ_mmd × λ_interp.
- **Phase B** (width, `pv-ae-stage-f-capacity`): 9 runs, c_mid × d_latent (uses winner from A).
- **Phase C** (training dynamics, `pv-ae-stage-f-training`): 6 runs, lr × scheduler (uses winner from A+B).
- **Phase D** (depth, `pv-ae-stage-f-depth`): sweep n_res_blocks at the A+B+C winner.
- Each run: 5000 steps, bs=256, bf16, **loss_type=mse** (recon objective).
- Validation logs BOTH MSE and MAE every 1000 steps regardless of training objective.
- λ_smooth pinned to 0 per user feedback.

## Stage F — Huber loss sweep  (status: finished)

> Stage F: Huber loss sweep (delta ∈ {0.01, 0.05, 0.1, 0.3}), 15k steps

| Run | λ_mmd | λ_interp | c_mid | d_latent | n_res | lr | sched | status | **score** | MSE | MAE | MMD | interp | smooth_cat | sharp_cat | sparse_cat | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| huber-d0.01 | 0.1 | 0.1 | 128 | 16 | 1 | 1e-3 | cosine | ✅ done | **3.273** | 0.0001 | 0.0080 | 0.2810 | 0.0046 | 0.0087 | 0.0086 | 0.0048 | 0 |
| huber-d0.05 | 0.1 | 0.1 | 128 | 16 | 1 | 1e-3 | cosine | ✅ done | **3.348** | 0.0001 | 0.0077 | 0.2699 | 0.0053 | 0.0081 | 0.0082 | 0.0054 | 1 |
| huber-d0.1 | 0.1 | 0.1 | 128 | 16 | 1 | 1e-3 | cosine | ✅ done | **3.317** | 0.0001 | 0.0073 | 0.2566 | 0.0057 | 0.0075 | 0.0077 | 0.0054 | 0 |
| huber-d0.3 | 0.1 | 0.1 | 128 | 16 | 1 | 1e-3 | cosine | ✅ done | **3.460** | 0.0001 | 0.0082 | 0.2388 | 0.0062 | 0.0083 | 0.0084 | 0.0068 | 1 |

**Winner**: `huber-d0.01` — score=3.273 (MSE=0.0001, MAE=0.0080, MMD=0.2810, interp=0.0046)

*`score` = mae/mae_min + mmd/mmd_min + interp/interp_min (lower is better, ≈3.0 means tied on all axes with the per-metric best). `smooth/sharp/sparse_cat` show per-category MAE.*

## Summary

- `huber-d0.01` (delta=0.01): MSE=0.000138, MAE=0.0080, MMD=0.2810, interp=0.0046, score=3.273
- `huber-d0.05` (delta=0.05): MSE=0.000120, MAE=0.0077, MMD=0.2699, interp=0.0053, score=3.348
- `huber-d0.1` (delta=0.1): MSE=0.000107, MAE=0.0073, MMD=0.2566, interp=0.0057, score=3.317
- `huber-d0.3` (delta=0.3): MSE=0.000128, MAE=0.0082, MMD=0.2388, interp=0.0062, score=3.460
- Best by composite: `huber-d0.01` (delta=0.01)
- Stage F complete.

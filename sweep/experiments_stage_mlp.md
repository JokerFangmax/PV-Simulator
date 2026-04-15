# Stage 0 AE — Sweep Tracker  (loss_type=mse, updated: 2026-04-14 00:58:00)


## Plan

- **Phase A** (loss weights, `pv-ae-stage-mlp-loss`): 9 runs, λ_mmd × λ_interp.
- **Phase B** (width, `pv-ae-stage-mlp-capacity`): 9 runs, c_mid × d_latent (uses winner from A).
- **Phase C** (training dynamics, `pv-ae-stage-mlp-training`): 6 runs, lr × scheduler (uses winner from A+B).
- **Phase D** (depth, `pv-ae-stage-mlp-depth`): sweep n_res_blocks at the A+B+C winner.
- Each run: 5000 steps, bs=256, bf16, **loss_type=mse** (recon objective).
- Validation logs BOTH MSE and MAE every 1000 steps regardless of training objective.
- λ_smooth pinned to 0 per user feedback.

## Stage MLP — MLP-based AE capacity sweep  (status: finished)

> Stage MLP: MLP-based AE capacity sweep (hidden_dim ∈ {64,128,256} × n_hidden_layers ∈ {1,2}), MAE loss, 15k steps

| Run | λ_mmd | λ_interp | c_mid | d_latent | n_res | lr | sched | status | **score** | MSE | MAE | MMD | interp | smooth_cat | sharp_cat | sparse_cat | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h64-L1-d16 | 0 | 0 | 64 | 16 | 1 | 1e-3 | cosine | ✅ done | **16.354** | 0.0010 | 0.0187 | 1.7985 | 1.2383 | 0.0204 | 0.0196 | 0.0092 | 0 |
| h64-L2-d16 | 0 | 0 | 64 | 16 | 1 | 1e-3 | cosine | ✅ done | **39.282** | 0.0034 | 0.0336 | 1.9582 | 3.1845 | 0.0363 | 0.0340 | 0.0174 | 1 |
| h128-L1-d16 | 0 | 0 | 128 | 16 | 1 | 1e-3 | cosine | ✅ done | **16.780** | 0.0009 | 0.0180 | 1.8255 | 1.2775 | 0.0188 | 0.0193 | 0.0089 | 0 |
| h128-L2-d16 | 0 | 0 | 128 | 16 | 1 | 1e-3 | cosine | ✅ done | **14.140** | 0.0040 | 0.0335 | 1.7953 | 0.9698 | 0.0355 | 0.0344 | 0.0171 | 1 |
| h256-L1-d16 | 0 | 0 | 256 | 16 | 1 | 1e-3 | cosine | ✅ done | **7.866** | 0.0014 | 0.0188 | 1.6846 | 0.4937 | 0.0193 | 0.0198 | 0.0092 | 0 |
| h256-L2-d16 | 0 | 0 | 256 | 16 | 1 | 1e-3 | cosine | ✅ done | **3.703** | 0.0052 | 0.0307 | 1.3536 | 0.0885 | 0.0317 | 0.0320 | 0.0132 | 1 |

**Winner**: `h256-L2-d16` — score=3.703 (MSE=0.0052, MAE=0.0307, MMD=1.3536, interp=0.0885)

*`score` = mae/mae_min + mmd/mmd_min + interp/interp_min (lower is better, ≈3.0 means tied on all axes with the per-metric best). `smooth/sharp/sparse_cat` show per-category MAE.*

## Summary

- `h64-L1-d16` (hidden=64, layers=1): MSE=0.000955, MAE=0.0187, MMD=1.7985, interp=1.2383, score=16.354
- `h64-L2-d16` (hidden=64, layers=2): MSE=0.003411, MAE=0.0336, MMD=1.9582, interp=3.1845, score=39.282
- `h128-L1-d16` (hidden=128, layers=1): MSE=0.000942, MAE=0.0180, MMD=1.8255, interp=1.2775, score=16.780
- `h128-L2-d16` (hidden=128, layers=2): MSE=0.003988, MAE=0.0335, MMD=1.7953, interp=0.9698, score=14.140
- `h256-L1-d16` (hidden=256, layers=1): MSE=0.001420, MAE=0.0188, MMD=1.6846, interp=0.4937, score=7.866
- `h256-L2-d16` (hidden=256, layers=2): MSE=0.005219, MAE=0.0307, MMD=1.3536, interp=0.0885, score=3.703
- Best by composite: `h256-L2-d16` (hidden=256, layers=2)
- Stage MLP complete.

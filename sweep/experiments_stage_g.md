# Stage 0 AE — Sweep Tracker  (loss_type=mse, updated: 2026-04-13 05:01:23)


## Plan

- **Phase A** (loss weights, `pv-ae-stage-g-loss`): 9 runs, λ_mmd × λ_interp.
- **Phase B** (width, `pv-ae-stage-g-capacity`): 9 runs, c_mid × d_latent (uses winner from A).
- **Phase C** (training dynamics, `pv-ae-stage-g-training`): 6 runs, lr × scheduler (uses winner from A+B).
- **Phase D** (depth, `pv-ae-stage-g-depth`): sweep n_res_blocks at the A+B+C winner.
- Each run: 5000 steps, bs=256, bf16, **loss_type=mse** (recon objective).
- Validation logs BOTH MSE and MAE every 1000 steps regardless of training objective.
- λ_smooth pinned to 0 per user feedback.

## Stage G — cosine_floor LR schedule  (status: finished)

> Stage G: cosine_floor LR sweep (decay_steps ∈ {3k,5k,7k} × min_ratio ∈ {0.01,0.1}), MAE loss, 15k steps

| Run | λ_mmd | λ_interp | c_mid | d_latent | n_res | lr | sched | status | **score** | MSE | MAE | MMD | interp | smooth_cat | sharp_cat | sparse_cat | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decay3000-floor0.01 | 0.1 | 0.1 | 128 | 16 | 1 | 1e-3 | cosine_floor | ✅ done | **3.364** | 0.0002 | 0.0084 | 0.3855 | 0.0010 | 0.0088 | 0.0088 | 0.0055 | 0 |
| decay3000-floor0.1 | 0.1 | 0.1 | 128 | 16 | 1 | 1e-3 | cosine_floor | ✅ done | **5.870** | 0.0009 | 0.0210 | 0.3249 | 0.0022 | 0.0224 | 0.0228 | 0.0139 | 1 |
| decay5000-floor0.01 | 0.1 | 0.1 | 128 | 16 | 1 | 1e-3 | cosine_floor | ✅ done | **3.875** | 0.0002 | 0.0079 | 0.3285 | 0.0018 | 0.0080 | 0.0082 | 0.0062 | 0 |
| decay5000-floor0.1 | 0.1 | 0.1 | 128 | 16 | 1 | 1e-3 | cosine_floor | ✅ done | **5.769** | 0.0005 | 0.0166 | 0.3055 | 0.0028 | 0.0170 | 0.0177 | 0.0123 | 1 |
| decay7000-floor0.01 | 0.1 | 0.1 | 128 | 16 | 1 | 1e-3 | cosine_floor | ✅ done | **4.581** | 0.0002 | 0.0084 | 0.3086 | 0.0026 | 0.0088 | 0.0088 | 0.0059 | 0 |
| decay7000-floor0.1 | 0.1 | 0.1 | 128 | 16 | 1 | 1e-3 | cosine_floor | ✅ done | **6.175** | 0.0005 | 0.0159 | 0.2959 | 0.0033 | 0.0165 | 0.0165 | 0.0123 | 1 |

**Winner**: `decay3000-floor0.01` — score=3.364 (MSE=0.0002, MAE=0.0084, MMD=0.3855, interp=0.0010)

*`score` = mae/mae_min + mmd/mmd_min + interp/interp_min (lower is better, ≈3.0 means tied on all axes with the per-metric best). `smooth/sharp/sparse_cat` show per-category MAE.*

## Summary

- `decay3000-floor0.01` (decay=3000, floor=0.01): MSE=0.000237, MAE=0.0084, MMD=0.3855, interp=0.0010, score=3.364
- `decay3000-floor0.1` (decay=3000, floor=0.1): MSE=0.000947, MAE=0.0210, MMD=0.3249, interp=0.0022, score=5.870
- `decay5000-floor0.01` (decay=5000, floor=0.01): MSE=0.000161, MAE=0.0079, MMD=0.3285, interp=0.0018, score=3.875
- `decay5000-floor0.1` (decay=5000, floor=0.1): MSE=0.000548, MAE=0.0166, MMD=0.3055, interp=0.0028, score=5.769
- `decay7000-floor0.01` (decay=7000, floor=0.01): MSE=0.000165, MAE=0.0084, MMD=0.3086, interp=0.0026, score=4.581
- `decay7000-floor0.1` (decay=7000, floor=0.1): MSE=0.000492, MAE=0.0159, MMD=0.2959, interp=0.0033, score=6.175
- Best by composite: `decay3000-floor0.01` (decay=3000, floor=0.01)
- Stage G complete.

# Stage 0 AE — Sweep Tracker  (loss_type=mae, updated: 2026-04-12 09:38:01)


## Plan

- **Phase A** (loss weights, `pv-ae-sweep-mae-loss`): 9 runs, λ_mmd × λ_interp.
- **Phase B** (model capacity, `pv-ae-sweep-mae-capacity`): 9 runs, c_mid × d_latent (uses winner from A).
- **Phase C** (training dynamics, `pv-ae-sweep-mae-training`): 6 runs, lr × scheduler (uses winner from A+B).
- Each run: 5000 steps, bs=256, bf16, **loss_type=mae** (recon objective).
- Validation logs BOTH MSE and MAE every 1000 steps regardless of training objective.
- λ_smooth pinned to 0 per user feedback.

## Phase A — Loss weights  (status: finished)

| Run | λ_mmd | λ_interp | c_mid | d_latent | lr | sched | status | **score** | MSE | MAE | MMD | interp | smooth_cat | sharp_cat | sparse_cat | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mmd0-int0 | 0 | 0 | 64 | 16 | 3e-4 | cosine | ✅ done | **18.827** | 2.5555 | 0.4212 | 1.5993 | 0.0033 | 0.4176 | 0.3681 | 0.2380 | 0 |
| mmd0-int0.1 | 0 | 0.1 | 64 | 16 | 3e-4 | cosine | ✅ done | **12.625** | 2.3709 | 0.4088 | 1.6011 | 0.0013 | 0.4019 | 0.3594 | 0.2346 | 1 |
| mmd0-int1 | 0 | 1 | 64 | 16 | 3e-4 | cosine | ✅ done | **9.537** | 2.0276 | 0.3828 | 1.6360 | 0.0003 | 0.3688 | 0.3388 | 0.2268 | 0 |
| mmd0.1-int0 | 0.1 | 0 | 64 | 16 | 3e-4 | cosine | ✅ done | **14.641** | 2.6373 | 0.4323 | 0.4506 | 0.0036 | 0.4309 | 0.3772 | 0.2427 | 1 |
| mmd0.1-int0.1 | 0.1 | 0.1 | 64 | 16 | 3e-4 | cosine | ✅ done | **10.071** | 2.5897 | 0.4278 | 0.5167 | 0.0021 | 0.4255 | 0.3730 | 0.2414 | 0 |
| mmd0.1-int1 | 0.1 | 1 | 64 | 16 | 3e-4 | cosine | ✅ done | **8.785** | 1.7393 | 0.3060 | 1.1562 | 0.0008 | 0.3029 | 0.2681 | 0.1775 | 1 |
| mmd1-int0 | 1 | 0 | 64 | 16 | 3e-4 | cosine | ✅ done | **58.307** | 2.1989 | 0.4111 | 0.2245 | 0.0180 | 0.4054 | 0.3651 | 0.2416 | 0 |
| mmd1-int0.1 | 1 | 0.1 | 64 | 16 | 3e-4 | cosine | ✅ done | **34.593** | 2.2371 | 0.4127 | 0.2371 | 0.0104 | 0.4063 | 0.3662 | 0.2423 | 1 |
| mmd1-int1 | 1 | 1 | 64 | 16 | 3e-4 | cosine | ✅ done | **15.047** | 2.4487 | 0.4282 | 0.2367 | 0.0041 | 0.4228 | 0.3774 | 0.2476 | 0 |

**Winner**: `mmd0.1-int1` — score=8.785 (MSE=1.7393, MAE=0.3060, MMD=1.1562, interp=0.0008)

*`score` = mae/mae_min + mmd/mmd_min + interp/interp_min (lower is better, ≈3.0 means tied on all axes with the per-metric best). `smooth/sharp/sparse_cat` show per-category MAE.*

## Phase B — Model capacity  (status: finished)

| Run | λ_mmd | λ_interp | c_mid | d_latent | lr | sched | status | **score** | MSE | MAE | MMD | interp | smooth_cat | sharp_cat | sparse_cat | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cmid32-dlat8 | 0.1 | 0.1 | 32 | 8 | 3e-4 | cosine | ✅ done | **39.388** | 2.1258 | 0.4970 | 0.5296 | 0.0012 | 0.4939 | 0.4313 | 0.2992 | 0 |
| cmid32-dlat16 | 0.1 | 0.1 | 32 | 16 | 3e-4 | cosine | ✅ done | **37.677** | 2.4076 | 0.4630 | 0.5003 | 0.0021 | 0.4569 | 0.4066 | 0.2745 | 1 |
| cmid32-dlat32 | 0.1 | 0.1 | 32 | 32 | 3e-4 | cosine | ✅ done | **35.550** | 2.6316 | 0.4333 | 0.4662 | 0.0023 | 0.4305 | 0.3781 | 0.2464 | 0 |
| cmid64-dlat8 | 0.1 | 0.1 | 64 | 8 | 3e-4 | cosine | ✅ done | **34.876** | 2.6506 | 0.4367 | 0.5115 | 0.0012 | 0.4351 | 0.3799 | 0.2464 | 1 |
| cmid64-dlat16 | 0.1 | 0.1 | 64 | 16 | 3e-4 | cosine | ✅ done | **35.004** | 2.5897 | 0.4278 | 0.5167 | 0.0021 | 0.4255 | 0.3730 | 0.2414 | 0 |
| cmid64-dlat32 | 0.1 | 0.1 | 64 | 32 | 3e-4 | cosine | ✅ done | **34.161** | 2.5354 | 0.4215 | 0.4424 | 0.0018 | 0.4190 | 0.3676 | 0.2379 | 1 |
| cmid128-dlat8 | 0.1 | 0.1 | 128 | 8 | 3e-4 | cosine | ✅ done | **24.207** | 1.6282 | 0.2925 | 0.4342 | 0.0015 | 0.2945 | 0.2426 | 0.1600 | 0 |
| cmid128-dlat16 | 0.1 | 0.1 | 128 | 16 | 3e-4 | cosine | ✅ done | **3.314** | 0.0013 | 0.0148 | 0.4917 | 0.0011 | 0.0154 | 0.0157 | 0.0097 | 1 |
| cmid128-dlat32 | 0.1 | 0.1 | 128 | 32 | 3e-4 | cosine | ✅ done | **3.082** | 0.0007 | 0.0134 | 0.4697 | 0.0010 | 0.0140 | 0.0143 | 0.0090 | 0 |

**Winner**: `cmid128-dlat16` — score=3.314 (MSE=0.0013, MAE=0.0148, MMD=0.4917, interp=0.0011)

*`score` = mae/mae_min + mmd/mmd_min + interp/interp_min (lower is better, ≈3.0 means tied on all axes with the per-metric best). `smooth/sharp/sparse_cat` show per-category MAE.*

## Phase C — Training dynamics  (status: finished)

| Run | λ_mmd | λ_interp | c_mid | d_latent | lr | sched | status | **score** | MSE | MAE | MMD | interp | smooth_cat | sharp_cat | sparse_cat | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr1e-4-cosine | 0.1 | 0.1 | 128 | 16 | 1e-4 | cosine | ✅ done | **49.781** | 2.3665 | 0.4189 | 0.5811 | 0.0025 | 0.4119 | 0.3697 | 0.2448 | 0 |
| lr1e-4-constant | 0.1 | 0.1 | 128 | 16 | 1e-4 | constant | ✅ done | **20.512** | 0.3884 | 0.1598 | 0.5205 | 0.0017 | 0.1461 | 0.1581 | 0.1163 | 1 |
| lr3e-4-cosine | 0.1 | 0.1 | 128 | 16 | 3e-4 | cosine | ✅ done | **3.961** | 0.0013 | 0.0148 | 0.4917 | 0.0011 | 0.0154 | 0.0157 | 0.0097 | 0 |
| lr3e-4-constant | 0.1 | 0.1 | 128 | 16 | 3e-4 | constant | ✅ done | **13.835** | 0.0297 | 0.1043 | 0.4605 | 0.0013 | 0.1189 | 0.1170 | 0.0487 | 1 |
| lr1e-3-cosine | 0.1 | 0.1 | 128 | 16 | 1e-3 | cosine | ✅ done | **3.244** | 0.0003 | 0.0091 | 0.3921 | 0.0013 | 0.0094 | 0.0096 | 0.0061 | 0 |
| lr1e-3-constant | 0.1 | 0.1 | 128 | 16 | 1e-3 | constant | ✅ done | **21.622** | 0.0539 | 0.1682 | 0.3685 | 0.0024 | 0.1706 | 0.1756 | 0.1308 | 1 |

**Winner**: `lr1e-3-cosine` — score=3.244 (MSE=0.0003, MAE=0.0091, MMD=0.3921, interp=0.0013)

*`score` = mae/mae_min + mmd/mmd_min + interp/interp_min (lower is better, ≈3.0 means tied on all axes with the per-metric best). `smooth/sharp/sparse_cat` show per-category MAE.*

## Summary

- Phase A winner: `mmd0.1-int1` (λ_mmd=0.1, λ_interp=1.0) score=8.785  [MSE=1.7393, MMD=1.1562, interp=0.0008]
- Phase B+C loss weights forced by user: λ_mmd=0.1, λ_interp=0.1.
- Phase B winner (smallest within +10% of best composite score): `cmid128-dlat16` (c_mid=128, d_latent=16) score=3.314  [MSE=0.0013, MMD=0.4917, interp=0.0011]
- Phase C winner: `lr1e-3-cosine` (lr=0.001, sched=cosine) score=3.244  [MSE=0.0003, MMD=0.3921, interp=0.0013]
- **Recommended Stage 0 config**: λ_mmd=0.1, λ_interp=0.1, λ_smooth=0, c_mid=128, d_latent=16, lr=0.001, scheduler=cosine
- Sweep complete.

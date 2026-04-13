# Stage 0 AE — Sweep Tracker  (updated: 2026-04-12 05:44:50)


## Plan

- **Phase A** (loss weights, `pv-ae-sweep-loss`): 9 runs, λ_mmd × λ_interp.
- **Phase B** (model capacity, `pv-ae-sweep-capacity`): 9 runs, c_mid × d_latent (uses winner from A).
- **Phase C** (training dynamics, `pv-ae-sweep-training`): 6 runs, lr × scheduler (uses winner from A+B).
- Each run: 5000 steps, bs=256, bf16, 2 GPUs (CUDA_VISIBLE_DEVICES 0/1) in parallel.
- λ_smooth pinned to 0 per user feedback.

## Phase A — Loss weights  (status: finished (reloaded))

| Run | λ_mmd | λ_interp | c_mid | d_latent | lr | sched | status | **score** | MSE | MMD | interp | smooth_cat | sharp_cat | sparse_cat | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mmd0-int0 | 0 | 0 | 64 | 16 | 3e-4 | cosine | ✅ done | **22.085** | 0.0004 | 1.8699 | 0.0013 | 0.0004 | 0.0004 | 0.0003 |  |
| mmd0-int0.1 | 0 | 0.1 | 64 | 16 | 3e-4 | cosine | ✅ done | **13.709** | 0.0004 | 1.9384 | 0.0004 | 0.0004 | 0.0005 | 0.0003 |  |
| mmd0-int1 | 0 | 1 | 64 | 16 | 3e-4 | cosine | ✅ done | **12.152** | 0.0004 | 2.0587 | 0.0001 | 0.0004 | 0.0005 | 0.0003 |  |
| mmd0.1-int0 | 0.1 | 0 | 64 | 16 | 3e-4 | cosine | ✅ done | **124.075** | 0.0018 | 0.3582 | 0.0133 | 0.0017 | 0.0019 | 0.0014 |  |
| mmd0.1-int0.1 | 0.1 | 0.1 | 64 | 16 | 3e-4 | cosine | ✅ done | **73.318** | 0.0016 | 0.4135 | 0.0076 | 0.0015 | 0.0017 | 0.0012 |  |
| mmd0.1-int1 | 0.1 | 1 | 64 | 16 | 3e-4 | cosine | ✅ done | **15.106** | 0.0005 | 1.7809 | 0.0006 | 0.0005 | 0.0006 | 0.0004 |  |
| mmd1-int0 | 1 | 0 | 64 | 16 | 3e-4 | cosine | ✅ done | **398.313** | 0.0047 | 0.2040 | 0.0435 | 0.0045 | 0.0046 | 0.0039 |  |
| mmd1-int0.1 | 1 | 0.1 | 64 | 16 | 3e-4 | cosine | ✅ done | **170.144** | 0.0038 | 0.2260 | 0.0180 | 0.0036 | 0.0037 | 0.0034 |  |
| mmd1-int1 | 1 | 1 | 64 | 16 | 3e-4 | cosine | ✅ done | **66.297** | 0.0034 | 1.1029 | 0.0059 | 0.0028 | 0.0028 | 0.0041 |  |

**Winner**: `mmd0-int1` — score=12.152 (MSE=0.0004, MMD=2.0587, interp=0.0001)

*`score` = mse/mse_min + mmd/mmd_min + interp/interp_min (lower is better, ≈3.0 means tied on all axes with the per-metric best).*

## Phase B — Model capacity  (status: finished)

| Run | λ_mmd | λ_interp | c_mid | d_latent | lr | sched | status | **score** | MSE | MMD | interp | smooth_cat | sharp_cat | sparse_cat | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cmid32-dlat8 | 0.1 | 0.1 | 32 | 8 | 3e-4 | cosine | ✅ done | **1805.712** | 1.1235 | 1.0292 | 0.0180 | 0.8689 | 1.0048 | 0.5936 | 0 |
| cmid32-dlat16 | 0.1 | 0.1 | 32 | 16 | 3e-4 | cosine | ✅ done | **6.457** | 0.0024 | 0.4758 | 0.0052 | 0.0023 | 0.0025 | 0.0016 | 1 |
| cmid32-dlat32 | 0.1 | 0.1 | 32 | 32 | 3e-4 | cosine | ✅ done | **4.585** | 0.0015 | 0.3832 | 0.0043 | 0.0014 | 0.0015 | 0.0012 | 0 |
| cmid64-dlat8 | 0.1 | 0.1 | 64 | 8 | 3e-4 | cosine | ✅ done | **1757.079** | 1.0939 | 0.6323 | 0.0177 | 0.9213 | 0.9045 | 0.4728 | 1 |
| cmid64-dlat16 | 0.1 | 0.1 | 64 | 16 | 3e-4 | cosine | ✅ done | **5.582** | 0.0016 | 0.4135 | 0.0076 | 0.0015 | 0.0017 | 0.0012 | 0 |
| cmid64-dlat32 | 0.1 | 0.1 | 64 | 32 | 3e-4 | cosine | ✅ done | **3.997** | 0.0012 | 0.3417 | 0.0047 | 0.0011 | 0.0013 | 0.0009 | 1 |
| cmid128-dlat8 | 0.1 | 0.1 | 128 | 8 | 3e-4 | cosine | ✅ done | **1568.480** | 0.9785 | 0.3294 | 0.0049 | 0.8260 | 0.7573 | 0.2653 | 0 |
| cmid128-dlat16 | 0.1 | 0.1 | 128 | 16 | 3e-4 | cosine | ✅ done | **4.149** | 0.0009 | 0.3870 | 0.0069 | 0.0008 | 0.0009 | 0.0008 | 1 |
| cmid128-dlat32 | 0.1 | 0.1 | 128 | 32 | 3e-4 | cosine | ✅ done | **3.269** | 0.0006 | 0.3304 | 0.0054 | 0.0006 | 0.0007 | 0.0005 | 0 |

**Winner**: `cmid128-dlat32` — score=3.269 (MSE=0.0006, MMD=0.3304, interp=0.0054)

*`score` = mse/mse_min + mmd/mmd_min + interp/interp_min (lower is better, ≈3.0 means tied on all axes with the per-metric best).*

## Phase C — Training dynamics  (status: finished)

| Run | λ_mmd | λ_interp | c_mid | d_latent | lr | sched | status | **score** | MSE | MMD | interp | smooth_cat | sharp_cat | sparse_cat | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr1e-4-cosine | 0.1 | 0.1 | 128 | 32 | 1e-4 | cosine | ✅ done | **10.399** | 0.0016 | 0.3982 | 0.0076 | 0.0015 | 0.0018 | 0.0012 | 0 |
| lr1e-4-constant | 0.1 | 0.1 | 128 | 32 | 1e-4 | constant | ✅ done | **26.841** | 0.0057 | 0.3810 | 0.0061 | 0.0061 | 0.0060 | 0.0039 | 1 |
| lr3e-4-cosine | 0.1 | 0.1 | 128 | 32 | 3e-4 | cosine | ✅ done | **5.360** | 0.0006 | 0.3304 | 0.0054 | 0.0006 | 0.0007 | 0.0005 | 0 |
| lr3e-4-constant | 0.1 | 0.1 | 128 | 32 | 3e-4 | constant | ✅ done | **133.260** | 0.0317 | 0.3215 | 0.0041 | 0.0360 | 0.0355 | 0.0140 | 1 |
| lr1e-3-cosine | 0.1 | 0.1 | 128 | 32 | 1e-3 | cosine | ✅ done | **3.297** | 0.0002 | 0.2700 | 0.0045 | 0.0002 | 0.0003 | 0.0002 | 0 |
| lr1e-3-constant | 0.1 | 0.1 | 128 | 32 | 1e-3 | constant | ✅ done | **255.334** | 0.0614 | 0.2934 | 0.0035 | 0.0689 | 0.0691 | 0.0252 | 1 |

**Winner**: `lr1e-3-cosine` — score=3.297 (MSE=0.0002, MMD=0.2700, interp=0.0045)

*`score` = mse/mse_min + mmd/mmd_min + interp/interp_min (lower is better, ≈3.0 means tied on all axes with the per-metric best).*

## Summary

- Phase A reloaded from disk. Composite-best would have been `mmd0-int1` (score=12.152). Per user direction this is overridden.
- Phase B+C loss weights forced by user: λ_mmd=0.1, λ_interp=0.1.
- Phase B winner (smallest within +10% of best composite score): `cmid128-dlat32` (c_mid=128, d_latent=32) score=3.269  [MSE=0.0006, MMD=0.3304, interp=0.0054]
- Phase C winner: `lr1e-3-cosine` (lr=0.001, sched=cosine) score=3.297  [MSE=0.0002, MMD=0.2700, interp=0.0045]
- **Recommended Stage 0 config**: λ_mmd=0.1, λ_interp=0.1, λ_smooth=0, c_mid=128, d_latent=32, lr=0.001, scheduler=cosine
- Sweep complete.

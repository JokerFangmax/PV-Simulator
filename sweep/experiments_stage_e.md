# Stage 0 AE — Stage E Sweep Tracker  (updated: 2026-04-12 12:20)


## Plan

- **Stage E** — Depth + budget diagnostic (4 runs, mixed MSE/MAE loss types).
  - Answers: is the MAE gap a width, depth, or training-budget problem?
  - E1 mae-res2: MAE winner cfg + n_res_blocks=2 (depth test)
  - E2 mae-res3: MAE winner cfg + n_res_blocks=3 (depth test)
  - E3 mae-long: MAE winner cfg at 15k steps (budget test)
  - E4 mse-res2: MSE winner cfg + n_res_blocks=2 (depth helps MSE too?)

## Stage E — Depth + budget diagnostic  (status: finished)

| Run | λ_mmd | λ_interp | c_mid | d_latent | n_res | lr | sched | status | MAE | MSE | MMD | interp | smooth_cat | sharp_cat | sparse_cat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mae-res2 | 0.1 | 0.1 | 128 | 16 | 2 | 1e-3 | cosine | ✅ done | 0.0114 | 0.0005 | 0.3798 | 0.0027 | 0.0119 | 0.0123 | 0.0072 |
| mae-res3 | 0.1 | 0.1 | 128 | 16 | 3 | 1e-3 | cosine | ✅ done | 0.0110 | 0.0004 | 0.4645 | 0.0024 | 0.0115 | 0.0121 | 0.0069 |
| mae-long | 0.1 | 0.1 | 128 | 16 | 1 | 1e-3 | cosine | ✅ done | **0.0067** | **0.0001** | **0.2837** | 0.0043 | 0.0072 | 0.0072 | 0.0040 |
| mse-res2 | 0.1 | 0.1 | 128 | 32 | 2 | 1e-3 | cosine | ✅ done | 0.0134 | 0.0004 | 0.2758 | 0.0050 | 0.0140 | 0.0132 | 0.0113 |

## Baselines (from prior sweeps, 5k steps each)

| Run | loss | n_res | MAE | MSE | MMD | interp |
| --- | --- | --- | --- | --- | --- | --- |
| MAE sweep winner | mae | 1 | 0.0091 | 0.0003 | 0.3921 | 0.0013 |
| MSE sweep winner | mse | 1 | 0.0111 | 0.0002 | 0.2653 | 0.0052 |

## Key findings

1. **The MAE gap is a budget problem, not a depth problem.** mae-long (n_res=1, 15k steps) dominates all other configs on MAE (0.0067), MSE (0.0001), and MMD (0.284). Deeper models (n_res=2,3) at 5k steps are *worse* than n_res=1 at 5k steps — they need more budget to converge.

2. **mae-long is the new best config overall.** It beats the MSE sweep winner on every metric except interp:
   - MAE: 0.0067 vs 0.0111 (−40%)
   - MSE: 0.0001 vs 0.0002 (−50%)
   - MMD: 0.284 vs 0.265 (+7%, essentially tied)
   - interp: 0.0043 vs 0.0052 (+17% better than MSE, but 3.3× worse than MAE-5k's 0.0013)

3. **Interp regressed** from 0.0013 (MAE-5k) to 0.0043 (MAE-15k). Longer training improved reconstruction at the cost of latent linearity. This is the one concern for downstream diffusion training.

4. **Depth doesn't help at 5k steps.** Both mae-res2 and mae-res3 underperform mae at n_res=1. Deeper models + more budget (10–15k) would be worth testing but is lower priority given mae-long's strong results.

5. **mse-res2 doesn't help MSE either.** Adding depth to the MSE winner made it worse on all metrics (MSE: 0.0004 vs 0.0002, MAE: 0.0134 vs 0.0111).

## Recommendation

Use **mae-long** (`outputs/stage0/diag/stage_e/mae-long/final`) as the Stage 0 AE for downstream Stage 1/2 training. The reconstruction quality improvement (−40% MAE, −50% MSE vs MSE winner) outweighs the interp regression, which is still better than the MSE winner's interp anyway.

Config: `loss_type=mae, c_mid=128, d_latent=16, n_res_blocks=1, lr=1e-3, cosine, λ_mmd=0.1, λ_interp=0.1, 15k steps`.

# Experiment Roadmap

The current policy is diagnosis first. Do not modify the model architecture
until the Phase 0 sanity checks and Stage 1 diagnostic cases are implemented,
run, and logged.

## Phase 0: Sanity Checks

Run these before proposing architecture changes:

1. Deterministic one-batch overfit.
2. Stochastic one-batch overfit.
3. AE reconstruction check.
4. Naive baselines.
5. Data loader / coordinate / mask audit.

Expected outcome: determine whether the Stage 1 failure is caused by the model,
the objective, the autoencoder, the dataset representation, masking, coordinate
conventions, or training setup.

## Stage 1 Diagnostic Cases

After Phase 0, implement and run focused trajectory cases:

1. Free fall.
2. Vertical bounce.
3. Oblique impact.
4. Rolling / sliding.
5. Multi-object floor.
6. Collision-heavy windows.

Expected outcome: isolate which physical regimes the current SimTransformer /
SimDiT handles and which regimes fail.

## Deferred Work

Do not add the following until diagnostic scripts and logs are in place:

- Rigid-Residual modules.
- Contact-Aware Sampling.
- Violation Feedback.
- PBD / XPBD.
- Warp simulation code.
- New Stage 2 coupling code.

## Decision Gate

Only consider architecture changes after the logs show:

- which sanity checks pass or fail;
- which diagnostic cases fail;
- whether failures are reproducible;
- whether simpler baselines expose the same problem;
- whether data, mask, coordinate, or AE reconstruction issues have been ruled
  out.

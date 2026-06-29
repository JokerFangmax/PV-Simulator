# Codex Protocol

This protocol applies to PV-Simulator tasks.

## Required Sequence

For every task:

1. Inspect relevant files first.
2. Explain the minimal plan.
3. Implement only the requested scope.
4. Run smoke tests, or provide exact commands when execution is not appropriate.
5. Write a step record to `docs/EXPERIMENT_LOG.md`.
6. Do not delete existing Stage 1 SimDiT code.
7. Preserve backward compatibility.

## Scope Rules

- Keep changes narrow and tied to the requested task.
- Prefer existing repository patterns over new abstractions.
- Do not rewrite the project around PBD, XPBD, Warp, or Stage 2 assumptions.
- Do not add Rigid-Residual, Contact-Aware Sampling, Violation Feedback, PBD,
  XPBD, Warp, or new Stage 2 code until diagnostics and logs are in place.
- Treat Stage 1 simulation trajectory learning as the active priority.

## Logging Rules

Every modification must append a record to `docs/EXPERIMENT_LOG.md` using this
format:

```text
## Step <ID>: <short title>

Date:
Goal:
Hypothesis:
Files changed:
Commands run:
Results:
Pass/Fail:
Problems found:
Next action:
```

## Diagnostic Order

Before model architecture changes, complete Phase 0:

1. Deterministic one-batch overfit.
2. Stochastic one-batch overfit.
3. AE reconstruction check.
4. Naive baselines.
5. Data loader / coordinate / mask audit.

Then implement Stage 1 diagnostic cases:

1. Free fall.
2. Vertical bounce.
3. Oblique impact.
4. Rolling / sliding.
5. Multi-object floor.
6. Collision-heavy windows.

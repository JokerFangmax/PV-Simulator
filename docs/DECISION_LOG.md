# Decision Log

This log records project-level decisions that constrain implementation and
experiments.

## Decision 001: Prioritize Stage 1 Stabilization

Date: 2026-06-27

Decision: Treat Stage 1 simulation trajectory learning as the current
engineering bottleneck.

Rationale: The MoT framework is still mostly proposal-stage, while current
progress depends on making the Simulation Branch reliable and diagnosable.

Consequence: New work should first improve Stage 1 diagnostics, sanity checks,
and experiment records.

## Decision 002: Treat Stage 2 as Future Work

Date: 2026-06-27

Decision: Do not assume Stage 2 physics-video coupling is working.

Rationale: The current experimental target is not joint video/simulation
training. Stage 2 should not drive Stage 1 stabilization decisions until Stage 1
diagnostics are in place.

Consequence: Avoid adding new Stage 2 coupling code during Phase 0 and Stage 1
diagnostic work.

## Decision 003: Do Not Assume PBD/XPBD/Warp

Date: 2026-06-27

Decision: Do not assume Hybrid Differentiable PBD, XPBD, PBD, or Warp-based
simulation is implemented.

Rationale: Rewriting around solver assumptions would distract from the current
diagnosis-first Stage 1 plan.

Consequence: Defer Rigid-Residual, Contact-Aware Sampling, Violation Feedback,
PBD, XPBD, Warp, and related architecture changes until diagnostic scripts and
logs provide evidence that they are needed.

# PV-Simulator Context

PV-Simulator is a physics-grounded video generation project. The long-term
proposal is a Mixture-of-Transformers framework that connects video generation
with explicit simulation trajectory generation.

## Proposed Architecture

- Video Branch: Wan2.1 / Wan2.1-Fun based image/text-conditioned video latent
  denoising.
- Simulation Branch: SimTransformer / SimDiT trained from scratch for
  point-cloud trajectory generation.
- Future Coupling Module: Stage 2 physics-video coupling through Joint
  Attention and timestep-dependent fusion.

## Current Status

The full MoT framework is still mostly at the proposal and planning stage. The
active engineering bottleneck is Stage 1 simulation trajectory learning.

Stage 2 coupling is not the current experimental target. Existing Stage 2 files
should be treated as exploratory or incomplete unless a specific experiment
proves otherwise.

## Research Direction

The current research direction is:

Diagnosis-first Structure-Aware Physics Trajectory Generation for Video Control.

This means the next work should establish controlled diagnostics before changing
model architecture. Evidence from small, reproducible failures should drive
later changes.

## Non-Assumptions

Do not assume Hybrid Differentiable PBD is implemented.
Do not assume XPBD, Warp, or PBD-based solvers are available.
Do not assume Stage 2 physics-video coupling is working.
Do not rewrite the project around PBD.

## Current Priority

Prioritize Stage 1 stabilization through diagnostic scripts, logs, and
controlled sanity checks. Architecture additions should wait until diagnostics
identify the failure mode they are meant to solve.

# PV-Simulator Method, Architecture, and Loss Design

This document describes the current implementation in this repository. Where
project notes differ from the code, the code is treated as the source of truth.

## Implementation Notes and Discrepancies

- There is no `losses/physics_losses.py` file in this checkout. The current
  geometry and physics losses are implemented as helper functions in
  `scripts/pv_simulator/train_stage1.py`.
- The code names the autoencoder pretraining step `Stage 0`, the simulation
  diffusion step `Stage 1`, and joint video/simulation training `Stage 2`.
  Some project notes call these Stage 1 and Stage 2 respectively.
- The implemented AE is not a static PointNet-style point-cloud autoencoder.
  `CausalAE`/`MLPAE` compresses per-point 3D trajectories
  `(B, T_raw, N, 3) -> (B, T, N, D)` with 4x temporal reduction. Stage 1
  applies it separately to position and velocity, then concatenates the latents.
- The prompt mentions a LeRobot-compatible ego-camera dataset. The current
  loaders support MOVI-style point-cloud directories/WebDataset shards and
  custom NPZ annotations. Stage 2 can consume optional `video_path` fields, but
  there is no explicit LeRobot loader in the code.

## 1. Architecture Overview

### Stage 0 / Conceptual Stage 1: Autoencoder

The frozen autoencoder lives in `videox_fun/models/sim_ae.py`.

Input and output:

```text
x in R^{B x T_raw x N x 3}
z = E(x) in R^{B x T x N x D},  T = (T_raw - 1) / 4 + 1
x_hat = G(z) in R^{B x T_raw x N x 3}
```

`CausalAE` uses causal 1D temporal convolutions and residual blocks. It processes
the sequence in chunks `[1, 4, 4, ...]` and carries a feature cache across chunks,
following the WAN VAE causal pattern. `MLPAE` is also available as an ablation:
the first frame is encoded directly and later frames are grouped by fours.

Training objective in `train_stage0.py`:

```text
L_AE = L_recon + lambda_mmd L_MMD + lambda_smooth L_smooth
       + lambda_interp L_interp
```

where `L_recon` is selected by `--loss_type` (`mse`, `mae`, or `huber`),
`L_MMD` is WAE-MMD regularization with an IMQ kernel, `L_smooth` penalizes
latent temporal differences, and `L_interp` encourages linear interpolation
consistency in latent space.

This differs from the prompt's "Chamfer + Local Distance" AE objective. Chamfer
and local distance are currently Stage-1 raw-space auxiliary losses, not the
default AE training loss.

### Stage 1 / Conceptual Stage 2: Flow-Matching Simulation Diffusion

The simulation diffusion model is trained in `scripts/pv_simulator/train_stage1.py`.
The backbone is `videox_fun/models/sim_transformer.py::SimTransformer`.

State representation:

```text
position latent = AE.encode(x_s_raw[..., :3])
velocity latent = AE.encode(x_s_raw[..., 3:6])
x_s_enc = concat(position latent, velocity latent)  # default d_state = 2 * d_latent
```

With default `d_latent=16`, `d_state=32`.

Flow matching uses:

```python
noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=...)
z_t = (1 - sigma) * z + sigma * epsilon
v_target = epsilon - z
```

The first state frame is held fixed:

```python
noisy_x_s_enc[:, :1] = x_s_enc[:, :1]
```

The model input is:

```text
noised state
+ repeated initial-frame state
+ init mask
+ point anchor features
+ simulation condition embedding
+ timestep embedding
```

`SimTransformer` uses factorized attention by default: spatial attention over
points within a frame, then temporal attention over each point trajectory. It
also supports flattened global attention, optional temporal correspondence, RoPE,
and object-local attention masks.

The output is the predicted flow-matching velocity field:

```text
pred_enc in R^{B x T x N x d_state}
```

The default diffusion loss is masked weighted MSE:

```text
L_FM = mean_valid( w(sigma) * || pred_enc - (epsilon - z) ||_2^2 )
```

where `w(sigma)` is produced by `diffusers.training_utils.compute_loss_weighting_for_sd3`.

### Optional Physics-Conditioned Rigid Decoder

`videox_fun/models/physics_rigid_decoder.py` adds a Kabsch-based rigid projection
and a physics-gated residual:

1. Decode predicted position latents to raw coordinates.
2. Fit per-object SE(3) transforms from canonical frame-0 points to predicted
   coordinates using Kabsch.
3. Produce coarse rigid positions.
4. Add a residual MLP gated by object deformability.

This path is enabled with `--use_physics_conditioned_decoder`.

## 2. Loss Functions in the Current Implementation

The following losses are implemented in `scripts/pv_simulator/train_stage1.py`.
All raw-space point losses exclude the hard-conditioned first frame and respect
padded point/frame masks when padded batching is enabled.

| Loss | Implementation | Purpose |
|---|---|---|
| Flow Matching | `F.mse_loss(pred_enc, noise - x_s_enc)` with scheduler weighting | Main denoising objective in latent or raw state space. |
| Chamfer Distance | `_compute_chamfer_loss(pos_pred, pos_gt, ...)` | Reconstruction/generation fidelity for unordered point sets. |
| Local Distance Loss | `_compute_local_distance_loss(pos_pred, pos_gt, ..., k=args.knn_k)` | Preserve local same-object geometric structure and reduce collapse. |
| Centroid Loss | `_compute_centroid_consistency_loss(...)` | Enforce global object translation consistency and reduce drift. |
| Kabsch Rotation Loss | `_compute_rotation_consistency_losses(...)` | Match per-object canonical-to-frame Kabsch rotations. |
| Kabsch Temporal Rotation Loss | `_compute_rotation_consistency_losses(...)` second return value | Match frame-to-frame relative rotations. |
| Rotation Axis Loss | `_compute_rotation_axis_loss(...)` | Penalize inconsistent incremental rotation-axis direction. |
| Angular Velocity Loss | `_compute_angular_velocity_loss(...)` | Fit angular velocity from centered positions and velocity fields, then match GT. |
| Contact COM Velocity Loss | `_compute_contact_com_velocity_loss(...)` | Match object centroid finite-difference velocity after first contact. |
| Contact Pose Angular Speed Loss | `_compute_contact_pose_ang_speed_loss(...)` | Match post-contact Kabsch incremental angular speed. |
| Temporal / Object Covariance | `_compute_covariance_loss(...)` | Match per-object 3x3 position covariance after frame 0. |
| Velocity Consistency | `_compute_velocity_consistency_loss(...)` | Match finite-difference velocities from predicted and GT positions. |
| Velocity Vector Loss | `_compute_velocity_vector_loss(...)` | Direct per-point velocity vector supervision. |
| Velocity Acceleration Loss | `_compute_velocity_acceleration_loss(...)` | Match changes in per-point velocity. |
| Deformation Gradient | `_compute_local_geometry_losses(...)` | Match local deformation gradients on canonical k-NN charts. |
| Local Volume | `_compute_local_geometry_losses(...)` | Match local log-volume change from deformation gradients. |
| Local Covariance | `_compute_local_geometry_losses(...)` | Match local covariance/anisotropy around k-NN neighborhoods. |
| Momentum | `_compute_momentum_loss(...)` | Match per-object linear momentum using uniformly split object mass. |
| Floor Penetration | `_compute_floor_penetration_loss(...)` | Penalize non-static valid points below the configured floor plane. |

### Chamfer Distance

For each valid frame, the implementation computes symmetric squared Chamfer:

```text
CD(P, Q) = 1/2 [ mean_{p in P} min_{q in Q} ||p - q||^2
               + mean_{q in Q} min_{p in P} ||q - p||^2 ]
```

The code uses `torch.cdist(pred_points, gt_points).square()` and averages over
valid frames and samples.

### Local Distance Loss

The k-NN graph is built per object on the GT frame-0 point cloud. For each
valid later frame, the implementation compares log edge-length ratios:

```text
r_pred(t, i, j) = log( ||p_pred[t,i] - p_pred[t,j]|| / ||p_pred[0,i] - p_pred[0,j]|| )
r_gt(t, i, j)   = log( ||p_gt[t,i]   - p_gt[t,j]||   / ||p_gt[0,i]   - p_gt[0,j]|| )
L_local = mean (r_pred - r_gt)^2
```

This measures deformation relative to frame 0 rather than absolute world-space
edge length.

### Centroid Loss

For every object and valid later frame:

```text
c_pred(t,o) = mean_{i in object o} p_pred(t,i)
c_gt(t,o)   = mean_{i in object o} p_gt(t,i)
L_centroid = mean ||c_pred - c_gt||^2
```

This prevents object-level drift after impact.

### Kabsch Rotation and Angular-Speed Losses

The Kabsch routines fit an optimal rigid rotation for each object. The row-vector
convention in `train_stage1.py` is:

```text
target_points ~= source_points @ R + t
```

For absolute rotation consistency, source points are canonical frame-0 points
and targets are predicted or GT frame points. For temporal rotation consistency,
the relative transform is:

```text
R_rel(t -> t+1) = R_t^T R_{t+1}
```

The contact angular-speed loss fits Kabsch rotations between consecutive
post-contact frames and compares:

```text
omega_speed = acos((trace(R) - 1) / 2) / dt
```

with `dt = 1 / 12` for the MOVI sampling rate used in the script.

### Temporal Covariance

The prompt names `temporal_covariance_loss(...)`. The current implementation
does not define a function with that exact name. The closest implemented loss is
`_compute_covariance_loss`, which matches per-object raw-position covariance
matrices for valid frames after frame 0:

```text
C_pred(t,o) = X_pred_centered^T X_pred_centered / (n_o - 1)
C_gt(t,o)   = X_gt_centered^T X_gt_centered / (n_o - 1)
L_cov = mean ||C_pred - C_gt||^2
```

## 3. What Changed from Previous Versions

Previous versions were closer to pure data-driven latent diffusion with shape
losses such as Chamfer and local distance. In collision-heavy sequences this was
not enough: predictions could minimize set-level distance while collapsing local
structure, drifting globally, or rearranging points after impact.

The current Stage-1 implementation adds explicit physical constraints:

- Centroid constraints keep each object's center of mass aligned with the target
  trajectory and reduce global drift after impact.
- Kabsch rotation and temporal rotation losses supervise rigid-body orientation
  instead of allowing arbitrary point permutations.
- Contact-aware COM velocity and angular-speed losses focus supervision on the
  first frames after collision, where collapse and post-impact jitter are most
  visible.
- Momentum and floor losses add coarse physical consistency for object mass and
  environment interaction.
- Local deformation, volume, and covariance losses provide additional structure
  for deformable or partially non-rigid objects.

Empirically, this design targets better shape retention and more realistic
rolling/rotation after collisions. Some jitter remains, and the method is not
yet a general material simulator for arbitrary solids, liquids, and gases.

## 4. Comparison with PhysCtrl

PhysCtrl is useful context because it also uses point clouds as an object
representation and relies on temporal-aware covariance in latent space to
encourage smooth temporal consistency.

However, covariance-based temporal smoothing is necessary but not sufficient for
PV-Simulator's target setting. The dataset includes violent collisions, rolling,
tumbling, and strong rigid-body motion, so explicit physical priors are needed.

| Aspect | PhysCtrl | PV-Simulator |
|---|---|---|
| Physical Priors | Implicit only, mainly covariance-based temporal consistency. | Explicit raw-space constraints: centroid, Kabsch rotations, angular speed, contact COM velocity, momentum, floor, and local geometry. |
| Collision Handling | Can fail under strong collisions because covariance statistics shift abruptly. | Designed to survive collisions by constraining object centers, rotations, and local neighborhoods. |
| Rotation/Rolling | Not explicitly modeled. | Explicitly supervised through Kabsch rotation and angular-speed losses. |
| Global Drift | Not directly constrained. | Constrained by centroid and contact COM velocity losses. |
| Generalization Target | Simple, smooth trajectories. | Complex interactions with collisions, rolling, deformation, and rigid-body motion. |
| Limitation | Weak explicit physics for violent interactions. | Still has jitter; material generalization remains incomplete. |

Key insight: covariance captures second-order shape statistics, not rigid-body
constraints. After collision, point distributions can change sharply and
covariance assumptions can break. Without explicit centroid and rotation
constraints, a model can reduce Chamfer distance by collapsing or rearranging
points locally, which is visually and physically implausible.

## 5. Training Objective Summary

In full mode, Stage 1 combines the selected diffusion objective with optional
physics terms:

```text
L_total = lambda_diffusion * L_diffusion
          + physics_warmup_scale * (
              lambda_local_dist * L_local
            + lambda_chamfer * L_chamfer
            + lambda_centroid * L_centroid
            + lambda_rotation * L_rotation
            + lambda_rotation_temporal * L_rotation_temporal
            + lambda_rotation_axis * L_rotation_axis
            + lambda_contact_com_velocity * L_contact_com
            + lambda_contact_pose_ang_speed * L_contact_ang_speed
            + lambda_covariance * L_cov
            + lambda_momentum * L_momentum
            + lambda_floor * L_floor
            + ...
          )
```

`--physics_mode minimal` keeps a smaller independently testable subset centered
on centroid, rotation, contact, and local-distance terms. `--physics_mode
shape_only` disables auxiliary physics terms.

## 6. Data Flow

For a MOVI sample:

```text
point_cloud_states.pkl / physics.npz / metadata.json
  -> MoviSimulationDataset
  -> x_s_raw:       (B, T_raw, N, 6)
  -> c_force_raw:   (B, T_raw, N, 6)
  -> object fields: floor, friction, restitution, mass, static flag, ids
  -> point_obj_idx: (B, N)
  -> AE encodes position, velocity, force, contact channels
  -> SimConditionEmbedder expands object properties to per-point tokens
  -> SimTransformer predicts flow velocity
  -> optional decode to raw points for geometry/physics losses
```

This format is compatible with point-cloud extraction workflows, but the actual
checked-in loader expects the MOVI/NPZ fields above rather than a LeRobot API.

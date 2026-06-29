# PV-Simulator

PV-Simulator generates physically plausible point-cloud sequences for deformable
and rigid objects under complex interactions such as collisions, rolling,
tumbling, and post-impact motion. The project builds on the VideoX-Fun codebase,
but the active simulation work is concentrated in `scripts/pv_simulator/` and
the simulation model/data modules under `videox_fun/`.

The current code uses explicit physical constraints in addition to data-driven
flow matching. These constraints are intended to prevent point-cloud collapse,
uncontrolled global drift, and arbitrary point rearrangement after collisions.

## Current Pipeline

The code names the stages slightly differently from some project notes:

| Conceptual stage | Code stage | Main script | Purpose |
|---|---:|---|---|
| Autoencoder pretraining | Stage 0 | `scripts/pv_simulator/train_stage0.py` | Train a frozen `CausalAE`/`MLPAE` that compresses 3D trajectories. |
| Simulation diffusion | Stage 1 | `scripts/pv_simulator/train_stage1.py` | Train `SimTransformer` with `FlowMatchEulerDiscreteScheduler` in AE latent space or raw-state ablation mode. |
| Joint video/simulation training | Stage 2 | `scripts/pv_simulator/train_stage2.py` | Optional MoT-style joint training with a Wan video branch and simulation branch. |

### Stage 0: Autoencoder

`CausalAE` compresses per-point 3D trajectories with 4x temporal reduction:

```text
(B, T_raw, N, 3) -> (B, T, N, D), where T = (T_raw - 1) / 4 + 1
```

The AE is applied separately to position and velocity channels in Stage 1, then
the latents are concatenated. With the default `d_latent=16`, the simulation
state dimension is `2 * d_latent = 32`.

### Stage 1: Flow-Matching Simulation Diffusion

`SimTransformer` predicts a flow-matching velocity field for noisy simulation
states. The scheduler is `diffusers.FlowMatchEulerDiscreteScheduler`, so the
training target is continuous-time flow matching rather than discrete DDPM:

```text
z_t = (1 - sigma) * z + sigma * noise
target velocity = noise - z
```

The model conditions on the initial frame, point anchors, object properties
(material, mass, static flag), floor height, force/contact channels, and
point-to-object mappings.

### Physics Integration

Unlike a pure data-driven diffusion model, Stage 1 can add raw-space geometry
and physics losses after decoding predicted latents. Implemented constraints
include Chamfer distance, local distance preservation, centroid consistency,
Kabsch rotation consistency, angular velocity, contact-aware center-of-mass and
angular-speed losses, momentum, floor penetration, and local deformation terms.

See [docs/METHOD.md](docs/METHOD.md) for details.

## Data Formats

The implemented loaders are in `videox_fun/data/dataset_simulation.py`.

### MOVI-style Directory or WebDataset

Each directory sample contains:

```text
sample_id/
  point_cloud_states.pkl   # point_states: (T_raw, N, 6), instances with point ranges
  metadata.json            # per-instance friction, restitution, mass, positions, velocities
  physics.npz              # optional force/contact/object physics arrays
```

The loader also supports a sharded WebDataset layout with
`dataset_manifest.json`, `manifest.jsonl`, and `shards/`.

### Custom NPZ

Use `--dataset_type simulation` with an annotation JSON whose entries contain
`file_path` and optional `text`/`video_path`. Each `.npz` should provide:

| Key | Shape | Meaning |
|---|---|---|
| `x_s_raw` | `(T_raw, N, 6)` | Point state: position `(x,y,z)` plus velocity `(vx,vy,vz)`. |
| `c_force_raw` | `(T_raw, N, 6)` | Force vector plus contact point. Defaults to zeros if absent. |
| `c_floor` | scalar | Floor height. |
| `c_mat` | `(n_objects, 2)` | Friction and restitution. |
| `c_mass` | `(n_objects,)` | Object mass. |
| `c_static` | `(n_objects,)` | Static object flag. |
| `c_init` | `(n_objects, 6)` | Initial object position and velocity. |
| `point_obj_idx` | `(N,)` | Object index for each point. |

`T_raw` must satisfy `T_raw = 4k + 1` unless padded batching clips to a valid
window.

Note: the prompt mentioned a LeRobot-compatible dataset with ego-camera
observations. This checkout does not contain a dedicated LeRobot loader. It
contains MOVI-style point-cloud loaders and optional video paths for Stage 2.

## Quick Start

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Train the autoencoder:

```bash
python scripts/pv_simulator/train_stage0.py \
  --output_dir outputs/ae \
  --max_train_steps 50000 \
  --batch_size 256 \
  --mixed_precision bf16
```

Train the simulation diffusion model on MOVI-style data:

```bash
accelerate launch --num_processes=4 scripts/pv_simulator/train_stage1.py \
  --dataset_type movi \
  --data_root datasets/movi_ab_10k \
  --ae_ckpt_dir outputs/ae/final \
  --output_dir outputs/stage1 \
  --padded_batch \
  --max_T_raw 21 \
  --max_points_per_object 200 \
  --gradient_accumulation_steps 8 \
  --mixed_precision bf16
```

Train with stronger explicit physics constraints:

```bash
accelerate launch --num_processes=4 scripts/pv_simulator/train_stage1.py \
  --dataset_type movi \
  --data_root datasets/movi_ab_10k \
  --ae_ckpt_dir outputs/ae/final \
  --output_dir outputs/stage1_physics \
  --padded_batch \
  --lambda_chamfer 0.01 \
  --lambda_local_dist 0.001 \
  --lambda_centroid 0.1 \
  --lambda_rotation 0.1 \
  --lambda_rotation_temporal 0.1 \
  --lambda_contact_com_velocity 0.1 \
  --lambda_contact_pose_ang_speed 0.1 \
  --physics_loss_warmup_steps 5000 \
  --mixed_precision bf16
```

Evaluate or visualize:

```bash
python scripts/pv_simulator/eval_ae.py outputs/ae/final
python scripts/pv_simulator/visualize.py --data_dir datasets/movi_ab_10k/00000 --output /tmp/motion.mp4
```

## Key Files

| Path | Role |
|---|---|
| `videox_fun/models/sim_ae.py` | `CausalAE` and `MLPAE` trajectory autoencoders. |
| `videox_fun/models/sim_transformer.py` | Simulation DiT backbone with factorized spatial/temporal attention. |
| `videox_fun/models/sim_condition.py` | Physical condition embedding. |
| `videox_fun/models/physics_rigid_decoder.py` | Optional Kabsch rigid alignment plus residual decoder. |
| `videox_fun/data/dataset_simulation.py` | MOVI/WebDataset and NPZ simulation datasets. |
| `scripts/pv_simulator/train_stage0.py` | AE training. |
| `scripts/pv_simulator/train_stage1.py` | Simulation flow-matching and physics losses. |
| `scripts/pv_simulator/train_stage2.py` | Joint video/simulation training scaffold. |

# PV-Simulator: Usage Instructions

PV-Simulator extends VideoX-Fun with a **Simulation Branch** (SimDiT) that generates physically plausible object trajectories. Training is three-stage: Stage 0 trains a Causal AE for temporal compression (then frozen); Stage 1 trains the simulation branch alone on physics trajectory data using the frozen AE; Stage 2 (not yet complete) trains both branches jointly with cross-modal Joint Attention.

## Environment

```bash
conda activate asr
# Install any missing packages:
uv pip install matplotlib imageio imageio-ffmpeg wandb
```

---

## Dataset Preparation

### MOVI-AB format (supported out of the box)

Each sample is a subdirectory under your data root:

```
datasets/movi_ab_10k/
  00000/
    point_cloud_states.pkl   # dict: point_states (T, N, 6), instances [{point_range, num_points}]
    metadata.json            # dict: instances [{friction, restitution, mass, positions, velocities, ...}]
  00001/
    ...
```

`point_states` shape: `(T_avail, N, 6)` — pos(3) + vel(3) per point per frame.  
The dataset clips to the largest valid `T_raw = 4k+1 ≤ T_avail` (e.g. 24 → 21).

### Custom NPZ format

Each sample is an `.npz` file referenced by an annotation JSON:

```json
[
  {"file_path": "relative/path/to/sample.npz", "text": "optional description"},
  ...
]
```

NPZ keys:

| Key            | Shape             | Description                            |
|----------------|-------------------|----------------------------------------|
| `x_s_raw`      | `(T_raw, N, 6)`   | Point states: pos(3) + vel(3)          |
| `c_force_raw`  | `(T_raw, N, 6)`   | Force vector(3) + contact point(3)     |
| `c_floor`      | scalar            | Floor height                           |
| `c_mat`        | `(n_obj, 2)`      | (friction, restitution) per object     |
| `c_mass`       | `(n_obj,)`        | Mass per object                        |
| `c_static`     | `(n_obj,)`        | Static flag (0 or 1) per object        |
| `c_init`       | `(n_obj, 6)`      | Initial pos(3) + vel(3) per object (legacy; model uses per-point `x_s_raw[:1]` via frozen AE) |
| `point_obj_idx`| `(N,)`            | Maps each point to its object index    |

`T_raw` must satisfy `T_raw = 4k+1` (e.g. 5, 9, 13, 17, 21, 25, ...).

---

## Stage 0: Causal AE Training

Trains a lightweight Causal Autoencoder (`CausalAE`) for 4× temporal compression of 3D trajectories: `(T_raw, 3) → (T, 16)`. Uses synthesized random data (no real physics data needed). After training, AE weights are frozen and used in all subsequent stages.

The AE is applied separately to position(3) and velocity(3) channels, producing concatenated latents of dimension 64 (2×16×2) per point.

```bash
# Single GPU (sufficient — AE is small ~200K params)
python scripts/pv_simulator/train_stage0.py \
    --output_dir outputs/ae \
    --max_train_steps 50000 \
    --batch_size 256 \
    --lr 3e-4

# Multi-GPU (optional)
accelerate launch --num_processes=4 scripts/pv_simulator/train_stage0.py \
    --output_dir outputs/ae \
    --max_train_steps 50000
```

### Key training arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_dir` | `outputs/ae` | Checkpoint output directory |
| `--max_train_steps` | 50000 | Total training steps |
| `--batch_size` | 256 | Batch size (trajectories per step) |
| `--n_pts` | 64 | Points per trajectory (batch axis) |
| `--lr` | 3e-4 | Peak learning rate (cosine decay to 0) |
| `--lambda_mmd` | 0.1 | WAE-MMD regularization weight |
| `--lambda_smooth` | 0.01 | Temporal smoothness weight |
| `--lambda_interp` | 0.1 | Linear interpolation consistency weight |
| `--d_latent` | 16 | Latent dimension per channel group |
| `--c_mid` | 64 | Intermediate channels in encoder/decoder |

### Checkpoint layout

```
outputs/ae/
  checkpoint-10000/
    causal_ae.pt
    config.pt
  final/
    causal_ae.pt
    config.pt
```

### Verification

```python
from videox_fun.models.sim_ae import CausalAE
ae = CausalAE.load("outputs/ae/final")
x = torch.randn(2, 21, 50, 3) * 5
x_hat, z = ae(x)
assert x_hat.shape == x.shape        # (2, 21, 50, 3)
assert z.shape == (2, 6, 50, 16)     # T=(21-1)//4+1=6
```

---

## Stage 1 Training (Simulation Branch Only)

Trains `SimTransformer` + `SimConditionEmbedder` using flow matching diffusion with a frozen `CausalAE` for encoding/decoding. Noise is added and loss is computed in **raw state space** `(T_raw, N, 6)`; the frozen AE provides temporal compression around the latent-space DiT.

### Single GPU (smoke test)

```bash
python scripts/pv_simulator/train_stage1.py \
    --ae_ckpt_dir outputs/ae/final \
    --dataset_type movi \
    --data_root datasets/movi_ab_10k \
    --output_dir outputs/stage1 \
    --max_train_steps 100000 \
    --gradient_accumulation_steps 8 \
    --train_batch_size 1 \
    --learning_rate 1e-4 \
    --lr_warmup_steps 1000 \
    --mixed_precision bf16
```

### Multi-GPU (recommended)

```bash
accelerate launch --num_processes=4 scripts/pv_simulator/train_stage1.py \
    --ae_ckpt_dir outputs/ae/final \
    --dataset_type movi \
    --data_root datasets/movi_ab_10k \
    --output_dir outputs/stage1 \
    --max_train_steps 100000 \
    --gradient_accumulation_steps 8 \
    --train_batch_size 1 \
    --learning_rate 1e-4 \
    --lr_warmup_steps 1000 \
    --mixed_precision bf16
```

Effective batch size = `num_processes × train_batch_size × gradient_accumulation_steps`.

### Custom NPZ dataset

```bash
accelerate launch --num_processes=4 scripts/pv_simulator/train_stage1.py \
    --ae_ckpt_dir outputs/ae/final \
    --dataset_type simulation \
    --ann_path /path/to/annotations.json \
    --data_root /path/to/sim_data \
    --output_dir outputs/stage1 \
    --gradient_accumulation_steps 8
```

### Key training arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_type` | `movi` | `movi` for MOVI-AB directories, `simulation` for npz format |
| `--data_root` | — | Dataset root directory |
| `--ann_path` | — | Annotation JSON path (required for `simulation` type) |
| `--output_dir` | `outputs/stage1` | Checkpoint output directory |
| `--max_train_steps` | 100000 | Total optimizer steps |
| `--gradient_accumulation_steps` | 8 | Accumulation steps per optimizer step |
| `--train_batch_size` | 1 | Samples per GPU per step |
| `--learning_rate` | 1e-4 | Peak learning rate |
| `--lr_scheduler` | `cosine` | LR schedule type |
| `--lr_warmup_steps` | 1000 | Linear warmup steps |
| `--mixed_precision` | `bf16` | `bf16`, `fp16`, or `no` |
| `--checkpointing_steps` | 5000 | Save a checkpoint every N global steps |
| `--ae_ckpt_dir` | — | **Required.** Path to Stage 0 CausalAE checkpoint directory |
| `--d_state` | 64 | Encoded point state dimension (2×d_latent, auto-corrected from AE) |
| `--d_sim` | 512 | SimTransformer hidden dimension |
| `--sim_num_layers` | 10 | Number of transformer blocks |
| `--padded_batch` | off | Enable padded batch mode (allows `--train_batch_size > 1`) |
| `--max_T_raw` | 21 | Max raw frames in padded mode (must be 4k+1) |
| `--max_points_per_object` | 200 | Max surface points per object in padded mode |
| `--report_to` | `tensorboard` | Logging backend: `tensorboard` or `wandb` |
| `--wandb_project` | `pv_simulator` | wandb project name (when `--report_to wandb`) |
| `--wandb_run_name` | — | wandb run name; auto-assigned if omitted |
| `--vis_steps` | 0 | Log GT+pred trajectory videos to wandb every N steps (0 = off) |
| `--num_vis_samples` | 4 | Number of fixed training samples to visualize |
| `--vis_num_inference_steps` | 50 | Denoising steps used during visualization inference |
| `--vis_fps` | 10 | FPS of visualization videos logged to wandb |

### Logging with wandb

Pass `--report_to wandb` to switch from TensorBoard to wandb. Both `train_loss` and `lr` are logged at every optimizer step.

```bash
accelerate launch --num_processes=4 scripts/pv_simulator/train_stage1.py \
    --ae_ckpt_dir outputs/ae/final \
    --dataset_type movi \
    --data_root datasets/movi_ab_10k \
    --output_dir outputs/stage1 \
    --report_to wandb \
    --wandb_project pv_simulator \
    --wandb_run_name exp_001 \
    --vis_steps 1000 \
    --num_vis_samples 4 \
    --gradient_accumulation_steps 8
```

With `--vis_steps N`, the trainer picks the first `--num_vis_samples` samples from the dataset at startup and runs full inference on them every N optimizer steps (main process only). Both the ground-truth trajectory and the model prediction are rendered as multi-view MP4s and logged to the `vis/` section of the wandb run. This is useful for tracking generation quality throughout training without a separate evaluation loop.

### Padded Batch Mode

By default, each sample has variable `T_raw`, `n_objects`, and `N` (total points), so the dataloader requires `--train_batch_size 1`. Enable `--padded_batch` to allow larger batches by zero-padding samples to fixed shape caps.

```bash
accelerate launch --num_processes=4 scripts/pv_simulator/train_stage1.py \
    --ae_ckpt_dir outputs/ae/final \
    --dataset_type movi \
    --data_root datasets/movi_ab_10k \
    --output_dir outputs/stage1_padded \
    --padded_batch \
    --max_objects 5 \
    --max_T_raw 21 \
    --max_points_per_object 200 \
    --train_batch_size 8 \
    --mixed_precision bf16
```

**How it works:**

- `max_N = max_objects × max_points_per_object` (e.g. 5 × 200 = 1000) is the fixed total point count per sample.
- Each sample's objects are packed into contiguous blocks of `max_points_per_object` slots. Points exceeding the cap are truncated; objects fewer than the cap get zero-padded blocks.
- `T_raw` is clipped to the largest valid `4k+1 ≤ max_T_raw`, then zero-padded to `max_T_raw`.
- Two boolean masks are computed per batch:
  - `point_mask (B, N)` — True for non-padded points (used to zero condition embeddings)
  - `valid_seq_mask (B, T*N)` — True for valid latent (time, point) pairs (used as DiT attention key bias)
- Loss uses a **raw-space mask** `(B, T_raw, N)` combining temporal and point validity; averaged only over valid positions.
- Inference (`infer_stage1.py`) always uses B=1 without padding — no changes needed.

### Checkpoint layout

```
outputs/stage1/
  checkpoint-5000/
    sim_transformer.pt
    sim_cond_embedder.pt
  final/
    sim_transformer.pt
    sim_cond_embedder.pt
```

---

## Stage 1 Inference

Run the trained SimTransformer to generate physics trajectories from pure noise, given physics conditions from a dataset sample.

```bash
python scripts/pv_simulator/infer_stage1.py \
    --ckpt_dir outputs/stage1/final \
    --ae_ckpt_dir outputs/ae/final \
    --data_dir datasets/movi_ab_10k/00000 \
    --output_dir outputs/infer/00000 \
    --num_inference_steps 50
```

This produces:
- `outputs/infer/00000/gt.mp4` — ground truth trajectory
- `outputs/infer/00000/pred.mp4` — predicted trajectory

### From a saved numpy array

```bash
python scripts/pv_simulator/infer_stage1.py \
    --ckpt_dir outputs/stage1/final \
    --ae_ckpt_dir outputs/ae/final \
    --point_states_npy /path/to/states.npy \
    --point_obj_idx_npy /path/to/obj_idx.npy \
    --output_dir outputs/infer/custom
```

### Key inference arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--ckpt_dir` | — | **Required.** Path to Stage 1 checkpoint directory |
| `--ae_ckpt_dir` | — | **Required.** Path to Stage 0 CausalAE checkpoint directory |
| `--data_dir` | — | MOVI-AB sample dir (mutually exclusive with `--point_states_npy`) |
| `--point_states_npy` | — | Ground truth states `.npy` (T_raw, N, 6) |
| `--point_obj_idx_npy` | — | Object index `.npy` (N,) — optional with npy input |
| `--output_dir` | `outputs/infer` | Directory to save MP4 animations |
| `--num_inference_steps` | 50 | Denoising steps (more = slower but better) |
| `--seed` | 42 | Random seed for reproducibility |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--dtype` | `bf16` | `bf16` or `fp32` |
| `--view` | `birdseye` | Camera view: `birdseye`, `side`, or `front` |
| `--fps` | 10 | Output animation frame rate |

### Programmatic usage

```python
import torch
from videox_fun.pipeline.pipeline_simulation import SimulationPipeline

pipeline = SimulationPipeline.from_pretrained(
    ckpt_dir="outputs/stage1/final",
    ae_ckpt_dir="outputs/ae/final",
    device="cuda",
    dtype=torch.bfloat16,
)

# All inputs need a batch dim (B=1 here)
result = pipeline(
    c_floor=torch.tensor([0.0]),                          # (B,)
    c_id=torch.tensor([[0, 1]]),                          # (B, n_objects)
    c_mat=torch.tensor([[[0.5, 0.3], [0.4, 0.8]]]),      # (B, n_objects, 2)
    c_mass=torch.tensor([[1.0, 2.0]]),                    # (B, n_objects)
    c_static=torch.tensor([[0, 0]]),                      # (B, n_objects)
    c_force_raw=torch.zeros(1, 21, 400, 6),               # (B, T_raw, N, 6)
    x_s_init=torch.randn(1, 1, 400, 6),                   # (B, 1, N, 6) first frame
    point_obj_idx=torch.zeros(1, 400, dtype=torch.long),  # (B, N)
    T=6,                                                   # latent frames (T_raw=4*(T-1)+1)
    num_inference_steps=50,
)

x_s_pred = result['x_s']   # (B, T_raw, N, 6)
```

---

## Visualization

Render point cloud motion as an animated MP4 or GIF, colored by object.

### From a MOVI-AB sample directory

```bash
python scripts/pv_simulator/visualize.py \
    --data_dir datasets/movi_ab_10k/00000 \
    --output /tmp/gt.mp4
```

### From numpy arrays

```bash
python scripts/pv_simulator/visualize.py \
    --point_states_npy /path/to/states.npy \
    --point_obj_idx_npy /path/to/obj_idx.npy \
    --output /tmp/out.gif
```

### Visualization options

| Argument | Default | Description |
|----------|---------|-------------|
| `--output` | `/tmp/point_cloud_motion.gif` | Output path (`.gif` or `.mp4`) |
| `--view` | `birdseye` | `birdseye` (XY), `side` (XZ), or `front` (YZ) |
| `--fps` | 10 | Animation frame rate |
| `--max_points_per_object` | — | Subsample points per object for speed |
| `--show_velocity` | off | Overlay velocity arrows |
| `--velocity_scale` | 0.1 | Arrow length multiplier |
| `--dpi` | 100 | Output resolution |

### Programmatic usage

```python
import numpy as np
from scripts.pv_simulator.visualize import visualize_point_cloud_motion

point_states = np.load("states.npy")      # (T, N, 6)
point_obj_idx = np.load("obj_idx.npy")    # (N,)

visualize_point_cloud_motion(
    point_states=point_states,
    point_obj_idx=point_obj_idx,
    output_path="out.mp4",
    fps=10,
    view="birdseye",
    show_velocity=True,
)
```

---

## Architecture Summary

| Component | File | Role |
|-----------|------|------|
| `CausalAE` | `videox_fun/models/sim_ae.py` | Frozen 4× causal autoencoder: `(T_raw, 3) → (T, 16)` per channel group; trained in Stage 0 |
| `SimConditionEmbedder` | `videox_fun/models/sim_condition.py` | Encodes floor, object ID, material, mass, static flag, force → `(B, T, N, 368)` |
| `SimTransformer` | `videox_fun/models/sim_transformer.py` | 10-block DiT; denoises in latent space with init conditioning |
| `SimulationDataset` | `videox_fun/data/dataset_simulation.py` | Loads custom NPZ format |
| `MoviSimulationDataset` | `videox_fun/data/dataset_simulation.py` | Loads MOVI-AB directory format |
| `SimulationPipeline` | `videox_fun/pipeline/pipeline_simulation.py` | Stage 1 inference: frozen AE Encoder→DiT→frozen AE Decoder denoising |

The full denoising network is `frozen AE Encoder → SimTransformer → frozen AE Decoder`. Diffusion noise and scheduler steps operate in **raw state space** `(B, T_raw, N, 6)`. The AE provides 4× temporal compression, applied separately to pos(3) and vel(3) channels → d_state=64 concatenated latents.

### Temporal compression

The frozen `CausalAE` (Stage 0) does `T_raw = 4k+1 → T_latent = k+1` (e.g. T_raw=21 → T=6). Applied separately to pos(3) and vel(3) → concatenated d_state=64. This matches the video VAE's 4× temporal compression ratio, keeping the two branches in sync for Stage 2 joint training.

```
AE Encoder:  [4k+1] --stride2--> [2k+1] --stride2--> [k+1]   (3 → 16 channels)
AE Decoder:  [k+1]  --upsample--> [2k+1] --upsample--> [4k+1] --trim--> [T_raw]  (16 → 3 channels)
```

### Initial state conditioning

The first raw frame `x_s_raw[:, :1]` is encoded with the same frozen AE (pos/vel separately → 64 dims), zero-padded to T latent frames, and concatenated with a binary mask `(B, T, N, 1)` where 0=given (t=0) and 1=unknown. The DiT `input_proj` takes `[x_enc(64), init_enc(64), init_mask(1), c_sim(368)] = 697 → 512`.

### Point state representation

- **Per-point**: `(x, y, z, vx, vy, vz)` — 6D position + velocity
- **Initial frame conditioning**: per-point first frame encoded via frozen AE (replaces old per-object `c_init`)
- **N points** total = sum of surface and volume samples across all objects
- `point_obj_idx: (N,)` maps each point to its object index `[0, n_objects)`

# PV-Simulator: Usage Instructions

PV-Simulator extends VideoX-Fun with a **Simulation Branch** (SimDiT) that generates physically plausible object trajectories. Training is two-stage: Stage 1 trains the simulation branch alone on physics trajectory data; Stage 2 (not yet complete) trains both branches jointly with cross-modal Joint Attention.

## Environment

```bash
conda activate asr
# Install any missing packages:
uv pip install matplotlib imageio imageio-ffmpeg
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
| `c_init`       | `(n_obj, 6)`      | Initial pos(3) + vel(3) per object     |
| `point_obj_idx`| `(N,)`            | Maps each point to its object index    |

`T_raw` must satisfy `T_raw = 4k+1` (e.g. 5, 9, 13, 17, 21, 25, ...).

---

## Stage 1 Training (Simulation Branch Only)

Trains `SimTransformer` + `CausalTemporalEncoder/Decoder` + `SimConditionEmbedder` using flow matching diffusion on physics trajectory data.

### Single GPU (smoke test)

```bash
python scripts/pv_simulator/train_stage1.py \
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
| `--d_state` | 256 | Encoded point state dimension |
| `--d_sim` | 512 | SimTransformer hidden dimension |
| `--sim_num_layers` | 10 | Number of transformer blocks |

### Checkpoint layout

```
outputs/stage1/
  checkpoint-5000/
    sim_transformer.pt
    x_s_encoder.pt
    x_s_decoder.pt
    sim_cond_embedder.pt
  final/
    sim_transformer.pt
    ...
```

---

## Stage 1 Inference

Run the trained SimTransformer to generate physics trajectories from pure noise, given physics conditions from a dataset sample.

```bash
python scripts/pv_simulator/infer_stage1.py \
    --ckpt_dir outputs/stage1/final \
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
    --point_states_npy /path/to/states.npy \
    --point_obj_idx_npy /path/to/obj_idx.npy \
    --output_dir outputs/infer/custom
```

### Key inference arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--ckpt_dir` | — | **Required.** Path to checkpoint directory |
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
    c_init=torch.zeros(1, 2, 7),                          # (B, n_objects, 7) pos+vel+mask
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
| `CausalTemporalEncoder` | `videox_fun/models/sim_causal_encoder.py` | Compresses `(B, T_raw, N, 6)` → `(B, T, N, d_state)` via 2× stride-2 causal conv |
| `CausalTemporalDecoder` | `videox_fun/models/sim_causal_encoder.py` | Expands `(B, T, N, d_state)` → `(B, T_raw, N, 6)` |
| `SimConditionEmbedder` | `videox_fun/models/sim_condition.py` | Encodes floor, object ID, material, mass, static flag, force, init state → `(B, T, N, d_cond)` |
| `SimTransformer` | `videox_fun/models/sim_transformer.py` | 10-block DiT; denoises latent trajectories |
| `SimulationDataset` | `videox_fun/data/dataset_simulation.py` | Loads custom NPZ format |
| `MoviSimulationDataset` | `videox_fun/data/dataset_simulation.py` | Loads MOVI-AB directory format |
| `SimulationPipeline` | `videox_fun/pipeline/pipeline_simulation.py` | Stage 1 inference: loads checkpoint, runs DDIM denoising |

### Temporal compression

The causal encoder does `T_raw = 4k+1 → T_latent = k+1` (e.g. T_raw=21 → T=6). This matches the video VAE's 4× temporal compression ratio, keeping the two branches in sync for Stage 2 joint training.

```
Encoder:  [4k+1] --stride2--> [2k+1] --stride2--> [k+1]
Decoder:  [k+1]  --upsample--> [2k+1] --upsample--> [4k+1] --trim--> [T_raw]
```

### Point state representation

- **Per-point**: `(x, y, z, vx, vy, vz)` — 6D position + velocity
- **Per-object**: `c_init = (x0, y0, z0, vx0, vy0, vz0, mask)` — 7D initial state + validity mask
- **N points** total = sum of surface and volume samples across all objects
- `point_obj_idx: (N,)` maps each point to its object index `[0, n_objects)`

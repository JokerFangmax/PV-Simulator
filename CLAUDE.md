# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PV-Simulator** (Physically Plausible Video Generation) is built on top of [VideoX-Fun](https://github.com/alibaba/VideoX-Fun), a video/image generation pipeline supporting 15+ Diffusion Transformer models. PV-Simulator extends it with a **Simulation Branch** that generates physically plausible object trajectories alongside video frames, using a **Mixture-of-Transformers (MoT)** architecture.

### Core Idea

Two parallel DiT branches share information via Joint Attention during diffusion denoising:

- **Video Branch**: Pre-trained `Wan2.1-Fun-V1.1-1.3B-InP` (30 blocks, dim=2048). Generates video latents from text + first-frame image.
- **Simulation Branch**: Trained from scratch (10 blocks, dim=512). Generates physics trajectories (per-point positions + velocities) from physical conditions.

Training is three-stage:
0. **Stage 0**: Train a Causal AE for 4x temporal compression of raw states `(T_raw, 3) → (T, 16)`. Uses WAE-style training (MSE + MMD + smoothness + interpolation consistency) on synthesized data. After training, AE weights are frozen for all subsequent stages.
1. **Stage 1**: Train SimDiT alone on physics trajectory data (no video branch). Uses frozen AE for encoding/decoding.
2. **Stage 2**: Joint training with LoRA on video branch + cross-modal Joint Attention. Uses frozen AE.

## Environment

Use conda environment: asr
```bash
conda activate asr
```

Install missing packages using `uv pip install`.

## Running

```bash
# Original VideoX-Fun web UI (for reference)
python examples/wan2.1_fun/app.py

# PV-Simulator Stage 0: Train Causal AE
accelerate launch --num_processes=1 scripts/pv_simulator/train_stage0.py \
  --output_dir outputs/ae \
  --max_steps 50000

# PV-Simulator Stage 1 training (simulation branch only)
accelerate launch --num_processes=4 scripts/pv_simulator/train_stage1.py \
  --ae_ckpt_dir outputs/ae/final \
  --ann_path /path/to/annotations.json \
  --data_root /path/to/sim_data \
  --output_dir outputs/stage1 \
  --gradient_accumulation_steps 8

# PV-Simulator Stage 2 training (joint MoT + LoRA)
accelerate launch --num_processes=4 scripts/pv_simulator/train_stage2.py \
  --ae_ckpt_dir outputs/ae/final \
  --stage1_ckpt outputs/stage1/final \
  --video_model_path /path/to/Wan2.1-Fun-V1.1-1.3B-InP \
  --ann_path /path/to/annotations.json \
  --data_root /path/to/sim_data \
  --output_dir outputs/stage2 \
  --gradient_accumulation_steps 4
```

There are **no automated tests** in this repository.

## Architecture

### PV-Simulator New Files

| File | Purpose |
|------|---------|
| `videox_fun/models/sim_ae.py` | `CausalAE`: 4x causal autoencoder for temporal compression `(T_raw, 3) → (T, 16)`; trained in Stage 0, frozen after |
| `videox_fun/models/sim_causal_encoder.py` | Building blocks (CausalConv1d, ResidualBlock1d, CausalDownsample1d, CausalUpsample1d); used by `sim_ae.py` and `sim_condition.py` force encoder |
| `videox_fun/models/sim_condition.py` | `SimConditionEmbedder`: encodes physics conditions (floor, object ID, material, mass, static flag, force) → d_cond=368 |
| `videox_fun/models/sim_transformer.py` | `SimTransformer`: 10-block DiT for physics simulation (d=512, 8 heads) |
| `videox_fun/models/joint_attention.py` | `JointAttention`: gated cross-modal attention between video and sim branches |
| `videox_fun/models/mot_wrapper.py` | `MoTWrapper`: orchestrates paired forward pass of both branches |
| `videox_fun/data/dataset_simulation.py` | `SimulationDataset`/`MoviSimulationDataset`: loads physics trajectory data; `sim_collate_fn` (bs=1) and `sim_collate_fn_padded` (bs>1 with zero-padding) |
| `videox_fun/pipeline/pipeline_simulation.py` | Stage 1 inference: standalone sim branch denoising with frozen AE |
| `videox_fun/pipeline/pipeline_mot.py` | Stage 2 inference: joint video + sim denoising |
| `scripts/pv_simulator/train_stage0.py` | Stage 0 training script: trains CausalAE on synthesized data |
| `scripts/pv_simulator/train_stage1.py` | Stage 1 training script (SimDiT only, frozen AE) |
| `scripts/pv_simulator/train_stage2.py` | Stage 2 training script (joint MoT + LoRA, frozen AE) |
| `scripts/pv_simulator/infer_stage1.py` | Stage 1 inference script |
| `scripts/pv_simulator/visualize.py` | Visualize point cloud motion |

### Key Conventions

- **Default bs=1**: T_raw, n_objects, and N_i (points per object) all vary per sample. Default mode uses `sim_collate_fn` which requires identical shapes across the batch → always bs=1 in practice. Use multi-GPU DDP + gradient accumulation for effective batch size.
- **Padded batch mode** (`--padded_batch`): Opt-in mode that enables bs>1 by zero-padding samples to fixed caps:
  - `max_T_raw` (default 21): raw frames clipped/padded to this (must be 4k+1)
  - `max_objects` (default 5): objects clipped/padded to this count
  - `max_points_per_object` (default 200): surface points per object, clipped/padded
  - `max_N = max_objects × max_points_per_object`: fixed total points per sample
  - Each object occupies a contiguous block of `max_points_per_object` slots in the flattened point array; `point_obj_idx[i*MPO:(i+1)*MPO] = i` regardless of padding
  - `point_mask (B, N)` bool: True for non-padded points; zeroes condition embeddings
  - `valid_seq_mask (B, T*N)` bool: True for valid latent (t, n) pairs; used as additive attention key bias in DiT (padded tokens → −∞)
  - Loss uses a **raw-space mask** `(B, T_raw, N)` combining temporal and point validity; averaged only over valid positions
- **max_objects=5**: Default cap on objects per scene (used in both modes).
- **4x causal AE (Stage 0)**: `CausalAE` encodes `(T_raw, 3) → (T, 16)` per pos/vel channel group via 2 stride-2 causal conv layers. Applied separately to pos(3) and vel(3) → concatenated d_state=64 (2×16×2). Frozen after Stage 0, used in all subsequent stages. Matches video VAE's 4× temporal compression ratio.
- **Initial state conditioning**: First frame `x_s_raw[:, :1]` is encoded with the same frozen AE, zero-padded to T latent frames, and concatenated with a binary inpainting mask `(B, T, N, 1)` where 0=given (t=0) and 1=unknown. DiT `input_proj` takes `[x_enc, init_enc, init_mask, c_sim]` → `Linear(2×64+1+368=697, 512)`.
- **Variable N**: Objects have different numbers of surface points N_i. All points are concatenated: N = ΣN_i. Per-object properties are expanded to per-point via `point_obj_idx` (gather-based, works in both modes).
- **Flow matching diffusion**: Noise is added in **raw state space** `(B, T_raw, N, 6)`. The full denoising network is `frozen AE Encoder → DiT → frozen AE Decoder`. Training target = `noise - x_s_raw` and loss is computed in raw space. At each inference step: `noisy_raw → AE encode → DiT → AE decode → velocity_raw → scheduler step in raw space`.

### Block Pairing (Video ↔ Sim)

```
Dense (blocks 0-3):   vid[0]↔sim[0], vid[1]↔sim[1], vid[2]↔sim[2], vid[3]↔sim[3]
Dilated (blocks 4-9): vid[8]↔sim[4], vid[12]↔sim[5], vid[16]↔sim[6], vid[20]↔sim[7], vid[24]↔sim[8], vid[28]↔sim[9]
```

### Joint Attention Design

`JointAttention` uses **bidirectional gated cross-attention** (not concatenated attention) to avoid the quadratic cost of large video sequences:
- `sim_cross`: sim queries attend to video K/V (sim learns from video context)
- `vid_cross`: video queries attend to sim K/V (video learns from physics)
- Learnable sigmoid gates initialized to `sigmoid(-10) ≈ 0` — start fully closed
- External `bias_scale` further controls gating:
  - **Training**: decays 1.0→0.0 over `bias_decay_steps` (gradual unlock)
  - **Inference**: set to `t/T_max` so gates open more at high noise, close at low noise

### Upstream VideoX-Fun Package Layout: `videox_fun/`

| Module | Purpose |
|--------|---------|
| `models/` | 55+ model architectures — Transformers 2D/3D, VAEs, text encoders. Key: `wan_transformer3d.py`, `wan_vae.py`, `attention_utils.py`. |
| `pipeline/` | ~40 inference pipelines. Each model has dedicated pipeline(s). |
| `ui/` | Gradio interfaces. `controller.py` manages model loading and generation. |
| `api/` | REST API (`api.py`) and multi-node inference. |
| `data/` | Dataset classes (`dataset_video.py`, `dataset_image_video.py`) and `bucket_sampler.py`. |
| `utils/` | LoRA loading (`lora_utils.py`), flow matching solvers (`fm_solvers.py`), etc. |
| `dist/` | Distributed training: FSDP and XFuser multi-GPU acceleration. |

### Key Reuse Points from VideoX-Fun

| What | Where |
|------|-------|
| `WanRMSNorm`, `WanLayerNorm`, `sinusoidal_embedding_1d` | `videox_fun/models/wan_transformer3d.py` |
| `attention()` (FlashAttention/SDPA backend) | `videox_fun/models/attention_utils.py` |
| `CausalConv3d`, `ResidualBlock`, `RMS_norm`, `Resample` | `videox_fun/models/wan_vae.py` |
| `create_network()`, `LoRANetwork` | `videox_fun/utils/lora_utils.py` |
| Training loop pattern | `scripts/wan2.1_fun/train_lora.py` |

### Data Format (for `SimulationDataset`)

Each sample is an `.npz` file with:

| Key | Shape | Description |
|-----|-------|-------------|
| `x_s_raw` | `(T_raw, N, 9)` | Point states: pos(3)+vel(3)+ang_vel(3) |
| `c_force_raw` | `(T_raw, N, 6)` | Force vector(3)+contact point(3) |
| `c_floor` | scalar | Floor height |
| `c_mat` | `(n_obj, 2)` | (friction, restitution) per object |
| `c_mass` | `(n_obj,)` | Mass per object |
| `c_static` | `(n_obj,)` | Static flag (0 or 1) per object |
| `c_init` | `(n_obj, 9)` | Initial pos+vel+ang_vel per object (legacy; model uses per-point `x_s_raw[:1]` via frozen AE instead) |
| `point_obj_idx` | `(N,)` | Maps each point to its object index |

Annotation JSON entries must have `file_path` (path to npz). Optional: `text`, `video_path` (for Stage 2).

## Sync Configuration

`mutagen.yml` configures bidirectional sync with remote server `Lab_3090:/data/szy/projects/phys_video/PV-Simulator`. This is a local dev tool — do not modify or commit changes to `mutagen.yml.lock`.

"""Stage 1 training for PV-Simulator: Simulation Branch only.

Trains the SimTransformer + SimConditionEmbedder on physics trajectory data
using LDM-style flow matching diffusion in the frozen CausalAE's latent
space (pre-trained in Stage 0). Raw states are encoded once to x_s_enc;
noise, target (noise - x_s_enc), DiT prediction, and loss are all in
latent space. The AE decoder is not used during training — gradients are
completely independent of the AE.

No video branch is loaded. No Joint Attention is used.

Usage:
    # MOVI-AB dataset (directory-based):
    accelerate launch --num_processes=4 scripts/pv_simulator/train_stage1.py \
        --dataset_type movi \
        --data_root datasets/movi_ab_10k \
        --ae_ckpt_dir outputs/ae/final \
        --output_dir outputs/stage1 \
        --gradient_accumulation_steps 8

    # Custom npz dataset:
    accelerate launch --num_processes=4 scripts/pv_simulator/train_stage1.py \
        --dataset_type simulation \
        --data_root /path/to/sim_data \
        --ann_path /path/to/annotations.json \
        --ae_ckpt_dir outputs/ae/final \
        --output_dir outputs/stage1 \
        --gradient_accumulation_steps 8

Based on scripts/wan2.1_fun/train_lora.py training loop structure.
"""

import argparse
import gc
import json
import logging
import math
import os
import random
import shutil
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (compute_density_for_timestep_sampling,
                                      compute_loss_weighting_for_sd3)
from tqdm.auto import tqdm

# Add project root to path
current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]
for project_root in project_roots:
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from functools import partial

from videox_fun.data.dataset_simulation import (MoviSimulationDataset,
                                                  SimulationDataset,
                                                  sim_collate_fn,
                                                  sim_collate_fn_padded)
from videox_fun.models.sim_ae import CausalAE
from videox_fun.models.sim_condition import SimConditionEmbedder
from videox_fun.models.sim_transformer import SimSTTransformer, SimTransformer
from videox_fun.utils.discrete_sampler import DiscreteSampling

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Train Simulation Branch")

    # Data
    parser.add_argument("--dataset_type", type=str, default="movi",
                        choices=["simulation", "movi"],
                        help="Dataset type: 'movi' for MOVI-AB directory format, "
                             "'simulation' for npz annotation format.")
    parser.add_argument("--ann_path", type=str, default=None,
                        help="Path to annotation JSON (required for --dataset_type simulation).")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Root directory for data files or MOVI dataset root.")
    parser.add_argument("--movi_temporal_compression_ratio", type=int, default=None,
                        help="MOVI frame clipping ratio. Default: 4 for latent AE mode, "
                             "keep all frames for --raw_xyz_diffusion. Use 0 to keep all frames.")

    # Pre-trained AE (from Stage 0)
    parser.add_argument("--ae_ckpt_dir", type=str, default=None,
                        help="Path to Stage 0 CausalAE checkpoint directory (e.g. outputs/ae/final)")
    parser.add_argument("--raw_xyz_diffusion", action="store_true",
                        help="Bypass the AE and train flow matching directly on normalized pos+vel states.")
    parser.add_argument("--xyz_norm_fac", type=float, default=5.0,
                        help="Position normalization center. xyz_norm=(xyz - norm_fac) / xyz_norm_scale.")
    parser.add_argument("--xyz_norm_scale", type=float, default=2.0,
                        help="Position normalization scale in raw-state mode.")
    parser.add_argument("--velocity_norm_scale", type=float, default=5.0,
                        help="Velocity normalization scale in raw-state mode.")
    parser.add_argument("--force_norm_scale", type=float, default=10.0,
                        help="Force-vector normalization scale in raw-state mode.")
    parser.add_argument("--frame_cond", action=argparse.BooleanOptionalAction, default=True,
                        help="Raw-state mode: condition on the clean initial 6-D state via a dedicated channel.")
    parser.add_argument("--pred_offset", action=argparse.BooleanOptionalAction, default=False,
                        help="Deprecated compatibility option; raw-state flow matching always predicts flow.")

    # Model architecture
    parser.add_argument("--d_state", type=int, default=32,
                        help="Encoded point state dimension (2 * AE d_latent for pos+vel concat).")
    parser.add_argument("--d_sim", type=int, default=256, help="SimTransformer hidden dimension.")
    parser.add_argument("--sim_ffn_dim", type=int, default=1024, help="SimTransformer FFN dimension.")
    parser.add_argument("--sim_num_heads", type=int, default=8, help="SimTransformer attention heads.")
    parser.add_argument("--sim_num_layers", type=int, default=10, help="SimTransformer blocks.")
    parser.add_argument("--max_objects", type=int, default=5, help="Maximum number of objects per scene.")

    # Training
    parser.add_argument("--output_dir", type=str, default="outputs/stage1", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # Padded batch mode (enables batch_size > 1 with variable-length samples)
    parser.add_argument("--padded_batch", action="store_true",
                        help="Enable padded batch mode: zero-pad samples to fixed shapes "
                             "so batch_size > 1 can be used with variable-length data.")
    parser.add_argument("--max_T_raw", type=int, default=21,
                        help="Max raw frame count in padded batch mode (must be 4k+1). Default: 21.")
    parser.add_argument("--max_points_per_object", type=int, default=200,
                        help="Max surface points per object in padded batch mode. "
                             "max_N = max_objects * max_points_per_object. Default: 200.")
    parser.add_argument("--max_train_steps", type=int, default=100000)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--lr_warmup_steps", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--allow_cpu", action="store_true",
                        help="Allow CPU training. By default, Stage 1 fails fast when CUDA is unavailable.")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)

    # Diffusion
    parser.add_argument("--train_sampling_steps", type=int, default=1000)
    parser.add_argument("--uniform_sampling", action="store_true")
    parser.add_argument("--weighting_scheme", type=str, default="logit_normal",
                        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--prediction_type", type=str, default="flow_prediction",
                        choices=["flow_prediction"],
                        help="Raw-state mode predicts the rectified-flow velocity field.")
    parser.add_argument("--clip_sample", action="store_true",
                        help="Deprecated compatibility option; unused by flow matching.")
    parser.add_argument("--lambda_xyz", type=float, default=1.0,
                        help="Raw xyz mode coordinate MSE weight.")
    parser.add_argument("--lambda_vel", type=float, default=0.1,
                        help="Raw xyz mode velocity consistency weight.")
    parser.add_argument("--lambda_floor", type=float, default=0.1,
                        help="Raw xyz mode floor penetration weight.")
    parser.add_argument("--floor_axis", type=str, default="z", choices=["x", "y", "z"],
                        help="Raw xyz mode vertical axis for floor loss. Kubric/MOVI uses z.")
    parser.add_argument("--lambda_mask", type=float, default=1.0,
                        help="Raw xyz mode extra MSE weight for drag_mask if present.")

    # Checkpointing & logging
    parser.add_argument("--checkpointing_steps", type=int, default=5000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=5)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="tensorboard", choices=["tensorboard", "wandb"])
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="pv_simulator",
                        help="wandb project name (used when --report_to wandb).")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="wandb run name. Defaults to None (auto-assigned by wandb).")
    parser.add_argument("--vis_steps", type=int, default=0,
                        help="Log trajectory visualization to wandb every N steps. 0 = disabled.")
    parser.add_argument("--num_vis_samples", type=int, default=3,
                        help="Number of fixed training samples to visualize.")
    parser.add_argument("--vis_num_inference_steps", type=int, default=50,
                        help="Denoising steps during visualization inference.")
    parser.add_argument("--vis_fps", type=int, default=10,
                        help="FPS for wandb visualization videos.")

    args = parser.parse_args()

    if args.dataset_type == "simulation" and args.ann_path is None:
        parser.error("--ann_path is required when --dataset_type simulation")
    if not args.raw_xyz_diffusion and args.ae_ckpt_dir is None:
        parser.error("--ae_ckpt_dir is required unless --raw_xyz_diffusion is enabled")
    if args.movi_temporal_compression_ratio is not None and args.movi_temporal_compression_ratio <= 0:
        args.movi_temporal_compression_ratio = None
    if not args.allow_cpu and not torch.cuda.is_available():
        parser.error(
            "CUDA is not available in this Python process. Refusing to train on CPU. "
            "Use the conda env's launcher explicitly, e.g. "
            "`/data/fhr/miniconda3/envs/videox/bin/python -m accelerate.commands.launch ...`, "
            "or pass --allow_cpu for an intentional CPU debug run."
        )

    return args


def _normalize_xyz(x, norm_fac: float, norm_scale: float = 2.0):
    return (x - norm_fac) / norm_scale


def _denormalize_xyz(x, norm_fac: float, norm_scale: float = 2.0):
    return x * norm_scale + norm_fac


def _normalize_raw_state(x_s_raw, xyz_center: float, xyz_scale: float, velocity_scale: float):
    """Normalize position and velocity without discarding either state component."""
    return torch.cat([
        _normalize_xyz(x_s_raw[..., :3], xyz_center, xyz_scale),
        x_s_raw[..., 3:6] / velocity_scale,
    ], dim=-1)


def _denormalize_raw_state(state, xyz_center: float, xyz_scale: float, velocity_scale: float):
    return torch.cat([
        _denormalize_xyz(state[..., :3], xyz_center, xyz_scale),
        state[..., 3:6] * velocity_scale,
    ], dim=-1)


def _normalize_raw_force(c_force_raw, norm_fac: float, xyz_scale: float, force_scale: float):
    """Normalize force vectors and contact positions to comparable numeric scales."""
    return torch.cat([
        c_force_raw[..., :3] / force_scale,
        _normalize_xyz(c_force_raw[..., 3:6], norm_fac, xyz_scale),
    ], dim=-1)


def _masked_mean(x, mask, eps: float = 1e-8):
    if mask is None:
        return x.mean()
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(dtype=x.dtype)
    return (x * mask).sum() / mask.sum().mul(x.shape[-1]).clamp(min=eps)


def _run_vis(vis_samples, sim_transformer, ae, sim_cond_embedder,
             global_step, device, weight_dtype, num_inference_steps, fps):
    """Run inference on fixed samples, render GT+pred videos, and log to wandb.

    Must be called on main process only. Imports wandb and pipeline lazily so
    that training without wandb does not require these packages.
    """
    import wandb
    from diffusers import FlowMatchEulerDiscreteScheduler
    from videox_fun.pipeline.pipeline_simulation import SimulationPipeline

    # visualize.py lives in the same directory — defer import so matplotlib
    # backend is set inside (it calls matplotlib.use('Agg') at import time).
    _vis_dir = os.path.dirname(os.path.abspath(__file__))
    if _vis_dir not in sys.path:
        sys.path.insert(0, _vis_dir)
    from visualize import visualize_point_cloud_motion

    pipeline = SimulationPipeline(
        sim_transformer=sim_transformer,
        ae=ae,
        sim_cond_embedder=sim_cond_embedder,
        scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000),
    )

    log_dict = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, sample in enumerate(vis_samples):
            def _b(t, as_float=False):
                if not isinstance(t, torch.Tensor):
                    t = torch.as_tensor(t)
                t = t.unsqueeze(0).to(device)
                return t.to(weight_dtype) if as_float else t

            x_s_raw       = _b(sample['x_s_raw'],       as_float=True)   # (1, T_raw, N, 6)
            c_force_raw   = _b(sample['c_force_raw'],   as_float=True)   # (1, T_raw, N, 6)
            c_floor       = _b(sample['c_floor'],        as_float=True)   # (1,)
            c_id          = _b(sample['c_id'])                             # (1, n_objects) int
            c_mat         = _b(sample['c_mat'],          as_float=True)   # (1, n_objects, 2)
            c_mass        = _b(sample['c_mass'],         as_float=True)   # (1, n_objects)
            c_static      = _b(sample['c_static'])                        # (1, n_objects) int
            point_obj_idx = _b(sample['point_obj_idx'])                   # (1, N) int
            x_s_init      = x_s_raw[:, :1, :, :]                          # (1, 1, N, 6)

            T_raw = x_s_raw.shape[1]
            T = (T_raw - 1) // 4 + 1

            with torch.amp.autocast("cuda", dtype=weight_dtype):
                result = pipeline(
                    c_floor=c_floor, c_id=c_id, c_mat=c_mat, c_mass=c_mass,
                    c_static=c_static, c_force_raw=c_force_raw,
                    x_s_init=x_s_init,
                    point_obj_idx=point_obj_idx, T=T,
                    num_inference_steps=num_inference_steps, show_progress=False,
                )
            x_s_pred = result['x_s'][0].float().cpu()   # (T_raw, N, 6)

            gt_np  = sample['x_s_raw']        # (T_raw, N, 6) — numpy/tensor from dataset
            obj_np = sample['point_obj_idx']   # (N,)

            gt_path   = os.path.join(tmpdir, f"s{i}_gt.mp4")
            pred_path = os.path.join(tmpdir, f"s{i}_pred.mp4")

            visualize_point_cloud_motion(gt_np,    obj_np, gt_path,   fps=fps, views=["birdseye", "side", "iso"])
            visualize_point_cloud_motion(x_s_pred, obj_np, pred_path, fps=fps, views=["birdseye", "side", "iso"])

            log_dict[f"vis/sample{i}_gt"]   = wandb.Video(gt_path,   fps=fps, format="mp4")
            log_dict[f"vis/sample{i}_pred"] = wandb.Video(pred_path, fps=fps, format="mp4")

        if log_dict:
            wandb.log(log_dict, step=global_step)

    logger.info(f"[Step {global_step}] Logged {len(vis_samples) * 2} vis videos to wandb.")


def _run_vis_raw_xyz(vis_samples, sim_transformer, sim_cond_embedder, args,
                     global_step, device, weight_dtype, num_inference_steps, fps):
    """Visualize raw-state flow-matching predictions against GT trajectories."""
    import wandb

    _vis_dir = os.path.dirname(os.path.abspath(__file__))
    if _vis_dir not in sys.path:
        sys.path.insert(0, _vis_dir)
    from visualize import visualize_point_cloud_motion

    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=args.train_sampling_steps,
    )

    log_dict = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, sample in enumerate(vis_samples):
            def _b(t, as_float=False):
                if not isinstance(t, torch.Tensor):
                    t = torch.as_tensor(t)
                t = t.unsqueeze(0).to(device)
                return t.to(weight_dtype) if as_float else t

            x_s_raw = _b(sample["x_s_raw"], as_float=True)
            c_force_raw = _b(sample["c_force_raw"], as_float=True)
            c_floor = _b(sample["c_floor"], as_float=True)
            c_id = _b(sample["c_id"])
            c_mat = _b(sample["c_mat"], as_float=True)
            c_mass = _b(sample["c_mass"], as_float=True)
            c_static = _b(sample["c_static"])
            point_obj_idx = _b(sample["point_obj_idx"])

            state = _normalize_raw_state(
                x_s_raw, args.xyz_norm_fac, args.xyz_norm_scale,
                args.velocity_norm_scale,
            )
            init_state = state[:, 0]
            c_force = _normalize_raw_force(
                c_force_raw, args.xyz_norm_fac, args.xyz_norm_scale,
                args.force_norm_scale,
            )
            c_floor_norm = _normalize_xyz(
                c_floor, args.xyz_norm_fac, args.xyz_norm_scale
            )
            B, T_raw, N, _ = state.shape

            c_sim = sim_cond_embedder(
                c_floor=c_floor_norm,
                c_id=c_id,
                c_mat=c_mat,
                c_mass=c_mass,
                c_static=c_static,
                c_force_enc=c_force,
                point_obj_idx=point_obj_idx,
                T=T_raw,
            )
            frame_mask = torch.ones(B, T_raw, device=device, dtype=torch.bool)
            sample_state = torch.randn(B, T_raw, N, 6, device=device, dtype=weight_dtype)
            scheduler.set_timesteps(num_inference_steps, device=device)

            for t in scheduler.timesteps:
                t_batch = t.expand(B)
                pred_flow = sim_transformer(
                    sample_state, c_sim, t_batch, dtype=weight_dtype,
                    init_state=init_state if args.frame_cond else None,
                    frame_mask=frame_mask,
                )
                sample_state = scheduler.step(
                    pred_flow, t, sample_state
                ).prev_sample

            pred_state = _denormalize_raw_state(
                sample_state.float(), args.xyz_norm_fac, args.xyz_norm_scale,
                args.velocity_norm_scale,
            )[0].cpu()

            gt_path = os.path.join(tmpdir, f"s{i}_gt.mp4")
            pred_path = os.path.join(tmpdir, f"s{i}_pred.mp4")
            visualize_point_cloud_motion(sample["x_s_raw"], sample["point_obj_idx"], gt_path,
                                         fps=fps, views=["birdseye", "side", "iso"])
            visualize_point_cloud_motion(pred_state, sample["point_obj_idx"], pred_path,
                                         fps=fps, views=["birdseye", "side", "iso"])
            log_dict[f"vis/sample{i}_gt"] = wandb.Video(gt_path, fps=fps, format="mp4")
            log_dict[f"vis/sample{i}_pred"] = wandb.Video(pred_path, fps=fps, format="mp4")

        if log_dict:
            wandb.log(log_dict, step=global_step)

    logger.info(f"[Step {global_step}] Logged {len(vis_samples) * 2} raw-state vis videos to wandb.")


def main():
    args = parse_args()

    # Accelerator setup
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    ae = None
    if args.raw_xyz_diffusion:
        if args.pred_offset:
            raise ValueError("--pred_offset is incompatible with raw-state flow matching.")
        args.d_state = 6
        sim_cond_embedder = SimConditionEmbedder(
            max_objects=args.max_objects,
            d_force=6,
        )
        d_cond = sim_cond_embedder.d_cond
        sim_transformer = SimSTTransformer(
            d_state=args.d_state,
            d_cond=d_cond,
            d_sim=args.d_sim,
            ffn_dim=args.sim_ffn_dim,
            num_heads=args.sim_num_heads,
            num_layers=args.sim_num_layers,
            frame_cond=args.frame_cond,
            pred_offset=args.pred_offset,
        )
        logger.info(
            "Raw-state flow matching enabled: AE bypassed, frame_cond=%s, "
            "xyz_center=%s, xyz_scale=%s, velocity_scale=%s, force_scale=%s",
            args.frame_cond, args.xyz_norm_fac, args.xyz_norm_scale,
            args.velocity_norm_scale, args.force_norm_scale,
        )
    else:
        # --- Load frozen CausalAE (from Stage 0) ---
        ae = CausalAE.load(args.ae_ckpt_dir)
        ae.to(accelerator.device, dtype=weight_dtype)
        ae.eval()
        for p in ae.parameters():
            p.requires_grad_(False)
        d_latent = ae.d_latent
        d_state_actual = 2 * d_latent  # pos + vel AE latents concatenated
        if args.d_state != d_state_actual:
            logger.warning(
                f"--d_state={args.d_state} overridden to {d_state_actual} "
                f"(2 * AE d_latent={d_latent})"
            )
            args.d_state = d_state_actual
        logger.info(f"Loaded frozen CausalAE from {args.ae_ckpt_dir} (d_latent={d_latent})")

        # --- Build Models ---
        # d_force = 2 * d_latent to match the AE-encoded force+contact representation.
        sim_cond_embedder = SimConditionEmbedder(
            max_objects=args.max_objects,
            d_force=2 * d_latent,
        )
        d_cond = sim_cond_embedder.d_cond

        sim_transformer = SimTransformer(
            d_state=args.d_state,
            d_cond=d_cond,
            d_sim=args.d_sim,
            ffn_dim=args.sim_ffn_dim,
            num_heads=args.sim_num_heads,
            num_layers=args.sim_num_layers,
        )

    # Move to device
    sim_cond_embedder.to(accelerator.device, dtype=weight_dtype)
    sim_transformer.to(accelerator.device, dtype=weight_dtype)

    # Count parameters
    total_params = (
        sum(p.numel() for p in sim_transformer.parameters()) +
        sum(p.numel() for p in sim_cond_embedder.parameters())
    )
    logger.info(f"Total trainable parameters: {total_params:,}")

    # --- Optimizer ---
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = (
        list(sim_transformer.parameters()) +
        list(sim_cond_embedder.parameters())
    )

    optimizer = optimizer_cls(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # --- Dataset & Dataloader ---
    if args.dataset_type == "movi":
        movi_temporal_compression_ratio = args.movi_temporal_compression_ratio
        if movi_temporal_compression_ratio is None and not args.raw_xyz_diffusion:
            movi_temporal_compression_ratio = 4
        train_dataset = MoviSimulationDataset(
            data_root=args.data_root,
            max_objects=args.max_objects,
            temporal_compression_ratio=movi_temporal_compression_ratio,
            prefer_physics_npz=True,
        )
    else:
        train_dataset = SimulationDataset(
            ann_path=args.ann_path,
            data_root=args.data_root,
            load_video=False,
            temporal_compression_ratio=None if args.raw_xyz_diffusion else 4,
        )

    if args.padded_batch:
        collate_fn = partial(
            sim_collate_fn_padded,
            max_T_raw=args.max_T_raw,
            max_objects=args.max_objects,
            max_points_per_object=args.max_points_per_object,
            temporal_compression_ratio=None if args.raw_xyz_diffusion else 4,
        )
        logger.info(
            f"Padded batch mode enabled: max_T_raw={args.max_T_raw}, "
            f"max_objects={args.max_objects}, max_points_per_object={args.max_points_per_object}, "
            f"max_N={args.max_objects * args.max_points_per_object}"
        )
    else:
        collate_fn = sim_collate_fn

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # --- Scheduler ---
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # --- Noise scheduler ---
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=args.train_sampling_steps,
    )

    # --- Accelerate prepare ---
    # All trainable modules are wrapped so DDP syncs their gradients across GPUs.
    # AE is frozen and not wrapped (no gradient sync needed).
    (sim_transformer, sim_cond_embedder,
     optimizer, train_dataloader, lr_scheduler) = accelerator.prepare(
        sim_transformer, sim_cond_embedder,
        optimizer, train_dataloader, lr_scheduler,
    )

    # --- Fixed visualization samples (loaded once, main process only) ---
    vis_samples = []
    if args.vis_steps > 0 and args.report_to == "wandb" and accelerator.is_main_process:
        num_vis = min(args.num_vis_samples, len(train_dataset))
        for idx in range(num_vis):
            vis_samples.append(train_dataset[idx])
        logger.info(f"Loaded {len(vis_samples)} fixed vis samples (indices 0..{num_vis - 1}).")

    # --- Resume from checkpoint ---
    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        global_step = int(os.path.basename(args.resume_from_checkpoint).split("-")[1])
        first_epoch = global_step // len(train_dataloader)

    # --- Timestep sampling ---
    idx_sampling = DiscreteSampling(args.train_sampling_steps, uniform_sampling=args.uniform_sampling)
    torch_rng = torch.Generator(device=accelerator.device).manual_seed(args.seed + accelerator.process_index)

    # --- Training loop ---
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    init_kwargs = {}
    if args.report_to == "wandb":
        init_kwargs["wandb"] = {"name": args.wandb_run_name}
    accelerator.init_trackers(args.wandb_project, config=vars(args), init_kwargs=init_kwargs)

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        accum_count = 0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(sim_transformer, sim_cond_embedder):
                # --- Unpack batch ---
                # Tensors have shape (B, ...) after collation
                x_s_raw = batch['x_s_raw'].to(accelerator.device, dtype=weight_dtype)    # (B, T_raw, N, 6)
                c_force_raw = batch['c_force_raw'].to(accelerator.device, dtype=weight_dtype)  # (B, T_raw, N, 6)
                c_floor = batch['c_floor'].to(accelerator.device)                          # (B,)
                c_id = batch['c_id'].to(accelerator.device)                                # (B, n_objects)
                c_mat = batch['c_mat'].to(accelerator.device)                              # (B, n_objects, 2)
                c_mass = batch['c_mass'].to(accelerator.device)                            # (B, n_objects,)
                c_static = batch['c_static'].to(accelerator.device)                        # (B, n_objects,)
                point_obj_idx = batch['point_obj_idx'].to(accelerator.device)              # (B, N)

                # Padded batch mode: masks and per-sample T_raw
                if args.padded_batch:
                    point_mask = batch['point_mask'].to(accelerator.device)   # (B, N) bool
                    T_raw_tensor = batch['T_raw'].to(accelerator.device)      # (B,) int
                else:
                    point_mask = None
                    T_raw_tensor = None

                B_sz = x_s_raw.shape[0]
                T_raw_dim = x_s_raw.shape[1]
                N_sz = x_s_raw.shape[2]
                bsz = B_sz

                loss_logs = {}
                if args.raw_xyz_diffusion:
                    # --- Raw 6-D state flow-matching path ---
                    state = _normalize_raw_state(
                        x_s_raw, args.xyz_norm_fac, args.xyz_norm_scale,
                        args.velocity_norm_scale,
                    )
                    init_state = state[:, 0]                              # (B, N, 6)
                    c_force_cond = _normalize_raw_force(
                        c_force_raw, args.xyz_norm_fac, args.xyz_norm_scale,
                        args.force_norm_scale,
                    )
                    c_floor_norm = _normalize_xyz(
                        c_floor.to(weight_dtype), args.xyz_norm_fac,
                        args.xyz_norm_scale,
                    )

                    if args.padded_batch:
                        t_idx = torch.arange(T_raw_dim, device=accelerator.device).unsqueeze(0)
                        frame_valid = t_idx < T_raw_tensor.unsqueeze(1)    # (B, T_raw)
                        state_mask = frame_valid.unsqueeze(2) & point_mask.unsqueeze(1)
                    else:
                        frame_valid = torch.ones(B_sz, T_raw_dim, device=accelerator.device, dtype=torch.bool)
                        state_mask = None

                    noise = torch.randn_like(state)
                    if not args.uniform_sampling:
                        u = compute_density_for_timestep_sampling(
                            weighting_scheme=args.weighting_scheme,
                            batch_size=bsz,
                            logit_mean=args.logit_mean,
                            logit_std=args.logit_std,
                            mode_scale=args.mode_scale,
                        )
                        indices = (u * noise_scheduler.config.num_train_timesteps).long()
                    else:
                        indices = idx_sampling(
                            bsz, generator=torch_rng, device=accelerator.device
                        ).long().cpu()
                    timesteps = noise_scheduler.timesteps[indices].to(accelerator.device)
                    sigmas = noise_scheduler.sigmas.to(
                        device=accelerator.device, dtype=weight_dtype
                    )
                    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
                    step_indices = [
                        (schedule_timesteps == t).nonzero().item() for t in timesteps
                    ]
                    sigma = sigmas[step_indices].flatten()
                    while sigma.ndim < state.ndim:
                        sigma = sigma.unsqueeze(-1)
                    noisy_state = (1.0 - sigma) * state + sigma * noise
                    target = noise - state

                    c_sim = sim_cond_embedder(
                        c_floor=c_floor_norm,
                        c_id=c_id,
                        c_mat=c_mat,
                        c_mass=c_mass,
                        c_static=c_static,
                        c_force_enc=c_force_cond,
                        point_obj_idx=point_obj_idx,
                        T=T_raw_dim,
                        point_mask=point_mask,
                    )  # (B, T_raw, N, d_cond)

                    with torch.amp.autocast(
                        "cuda",
                        dtype=weight_dtype,
                        enabled=accelerator.device.type == "cuda" and weight_dtype != torch.float32,
                    ):
                        pred_flow = sim_transformer(
                            noisy_state,
                            c_sim,
                            timesteps,
                            dtype=weight_dtype,
                            init_state=init_state if args.frame_cond else None,
                            point_mask=point_mask,
                            frame_mask=frame_valid,
                        )

                    pred_clean = noisy_state - sigma * pred_flow
                    flow_loss = _masked_mean(
                        F.mse_loss(pred_flow.float(), target.float(), reduction="none"),
                        state_mask,
                    )
                    velocity_loss = _masked_mean(
                        F.mse_loss(
                            pred_clean[..., 3:6].float(), state[..., 3:6].float(),
                            reduction="none",
                        ),
                        state_mask,
                    )

                    floor_axis_idx = {"x": 0, "y": 1, "z": 2}[args.floor_axis]
                    floor_coord = c_floor_norm.view(B_sz, 1, 1)
                    floor_penalty = F.relu(
                        floor_coord - pred_clean[..., floor_axis_idx]
                    ).pow(2).unsqueeze(-1)
                    floor_loss = _masked_mean(floor_penalty.float(), state_mask)

                    mask_loss = pred_clean.new_tensor(0.0)
                    if "drag_mask" in batch:
                        drag_mask = batch["drag_mask"].to(accelerator.device).bool()
                        if drag_mask.ndim == 2:
                            drag_mask = drag_mask.unsqueeze(1).expand(-1, T_raw_dim, -1)
                        drag_mask = drag_mask & state_mask if state_mask is not None else drag_mask
                        mask_loss = _masked_mean(
                            F.mse_loss(
                                pred_clean[..., :3].float(), state[..., :3].float(),
                                reduction="none",
                            ),
                            drag_mask,
                        )

                    loss = (
                        args.lambda_xyz * flow_loss
                        + args.lambda_vel * velocity_loss
                        + args.lambda_floor * floor_loss
                        + args.lambda_mask * mask_loss
                    )
                    loss_logs = {
                        "loss_flow": flow_loss.detach().item(),
                        "loss_vel": velocity_loss.detach().item(),
                        "loss_floor": floor_loss.detach().item(),
                        "loss_mask": mask_loss.detach().item(),
                    }
                else:
                    # --- Encode x_s_raw and c_force_raw once via frozen AE (LDM-style) ---
                    with torch.no_grad():
                        pos_enc = ae.encode(x_s_raw[..., :3])        # (B, T, N, d_latent)
                        vel_enc = ae.encode(x_s_raw[..., 3:6])       # (B, T, N, d_latent)
                        # c_force_raw: force(3) and contact(3) each -> d_latent via frozen AE
                        force_enc = ae.encode(c_force_raw[..., :3])
                        contact_enc = ae.encode(c_force_raw[..., 3:6])
                    x_s_enc = torch.cat([pos_enc, vel_enc], dim=-1)  # (B, T, N, d_state)
                    c_force_enc = torch.cat([force_enc, contact_enc], dim=-1)
                    T = x_s_enc.shape[1]

                    # --- Flow matching noise (in latent space) ---
                    noise = torch.randn_like(x_s_enc)   # (B, T, N, d_state)

                    if not args.uniform_sampling:
                        u = compute_density_for_timestep_sampling(
                            weighting_scheme=args.weighting_scheme,
                            batch_size=bsz,
                            logit_mean=args.logit_mean,
                            logit_std=args.logit_std,
                            mode_scale=args.mode_scale,
                        )
                        indices = (u * noise_scheduler.config.num_train_timesteps).long()
                    else:
                        indices = idx_sampling(bsz, generator=torch_rng, device=accelerator.device)
                        indices = indices.long().cpu()

                    timesteps = noise_scheduler.timesteps[indices].to(device=accelerator.device)

                    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=weight_dtype)
                    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
                    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
                    sigma = sigmas[step_indices].flatten()
                    while len(sigma.shape) < x_s_enc.ndim:
                        sigma = sigma.unsqueeze(-1)

                    noisy_x_s_enc = (1.0 - sigma) * x_s_enc + sigma * noise
                    target = noise - x_s_enc

                    init_enc_1 = x_s_enc[:, :1]
                    init_enc_padded = torch.cat([
                        init_enc_1,
                        torch.zeros(B_sz, T - 1, N_sz, args.d_state,
                                    device=accelerator.device, dtype=weight_dtype),
                    ], dim=1)
                    init_mask = torch.ones(B_sz, T, N_sz, 1,
                                           device=accelerator.device, dtype=weight_dtype)
                    init_mask[:, 0, :, :] = 0.0

                    if args.padded_batch:
                        t_latent = (T_raw_tensor - 1) // 4 + 1
                        t_idx = torch.arange(T, device=accelerator.device).unsqueeze(0)
                        t_valid = t_idx < t_latent.unsqueeze(1)
                        latent_seq_mask = t_valid.unsqueeze(2) & point_mask.unsqueeze(1)
                        valid_seq_mask = latent_seq_mask.view(B_sz, T * N_sz)
                    else:
                        valid_seq_mask = None

                    c_sim = sim_cond_embedder(
                        c_floor=c_floor,
                        c_id=c_id,
                        c_mat=c_mat,
                        c_mass=c_mass,
                        c_static=c_static,
                        c_force_enc=c_force_enc,
                        point_obj_idx=point_obj_idx,
                        T=T,
                        point_mask=point_mask,
                    )

                    with torch.amp.autocast(
                        "cuda",
                        dtype=weight_dtype,
                        enabled=accelerator.device.type == "cuda" and weight_dtype != torch.float32,
                    ):
                        pred_enc = sim_transformer(
                            noisy_x_s_enc, init_enc_padded, init_mask, c_sim, timesteps,
                            dtype=weight_dtype, valid_seq_mask=valid_seq_mask,
                        )

                    weighting = compute_loss_weighting_for_sd3(
                        weighting_scheme=args.weighting_scheme, sigmas=sigma,
                    )
                    loss_per_elem = F.mse_loss(pred_enc.float(), target.float(), reduction='none')

                    if args.padded_batch and point_mask is not None:
                        t_latent = (T_raw_tensor - 1) // 4 + 1
                        t_idx = torch.arange(T, device=accelerator.device).unsqueeze(0)
                        t_valid = t_idx < t_latent.unsqueeze(1)
                        latent_mask = t_valid.unsqueeze(2) & point_mask.unsqueeze(1)
                        loss_per_elem = loss_per_elem * latent_mask.unsqueeze(-1).float()
                        n_valid = latent_mask.float().sum() * args.d_state
                        loss = (loss_per_elem * weighting.float()).sum() / n_valid.clamp(min=1)
                    else:
                        loss = (loss_per_elem * weighting.float()).mean()

                # --- Backward ---
                avg_loss = accelerator.gather(loss.repeat(bsz)).mean()
                train_loss += avg_loss.item()
                accum_count += 1

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # --- Logging & checkpointing ---
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                current_lr = lr_scheduler.get_last_lr()[0]
                log_payload = {"train_loss": train_loss / max(accum_count, 1), "lr": current_lr}
                log_payload.update(loss_logs)
                accelerator.log(log_payload, step=global_step)
                train_loss = 0.0
                accum_count = 0

                # --- Visualization ---
                if (
                    args.vis_steps > 0
                    and args.report_to == "wandb"
                    and global_step % args.vis_steps == 0
                    and accelerator.is_main_process
                    and len(vis_samples) > 0
                ):
                    unwrapped_sim = accelerator.unwrap_model(sim_transformer)
                    unwrapped_cond = accelerator.unwrap_model(sim_cond_embedder)
                    unwrapped_sim.eval()
                    unwrapped_cond.eval()
                    try:
                        if args.raw_xyz_diffusion:
                            _run_vis_raw_xyz(
                                vis_samples, unwrapped_sim, unwrapped_cond, args,
                                global_step, accelerator.device, weight_dtype,
                                args.vis_num_inference_steps, args.vis_fps,
                            )
                        else:
                            _run_vis(
                                vis_samples, unwrapped_sim, ae, unwrapped_cond,
                                global_step, accelerator.device, weight_dtype,
                                args.vis_num_inference_steps, args.vis_fps,
                            )
                    except Exception as e:
                        logger.warning(f"Visualization failed at step {global_step}: {e}")
                    finally:
                        unwrapped_sim.train()
                        unwrapped_cond.train()

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # Check total limit
                        if args.checkpoints_total_limit is not None:
                            checkpoints = [
                                d for d in os.listdir(args.output_dir)
                                if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                for removing_checkpoint in checkpoints[:num_to_remove]:
                                    shutil.rmtree(os.path.join(args.output_dir, removing_checkpoint))

                        # Save trainable sim components (unwrap DDP wrappers)
                        save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_dir, exist_ok=True)

                        torch.save(accelerator.unwrap_model(sim_transformer).state_dict(),
                                   os.path.join(save_dir, "sim_transformer.pt"))
                        torch.save(accelerator.unwrap_model(sim_cond_embedder).state_dict(),
                                   os.path.join(save_dir, "sim_cond_embedder.pt"))
                        with open(os.path.join(save_dir, "training_args.json"), "w") as f:
                            json.dump(vars(args), f, indent=2, sort_keys=True)
                        logger.info(f"Saved checkpoint to {save_dir}")

                        gc.collect()
                        torch.cuda.empty_cache()

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    # Final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_dir = os.path.join(args.output_dir, "final")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(accelerator.unwrap_model(sim_transformer).state_dict(),
                   os.path.join(save_dir, "sim_transformer.pt"))
        torch.save(accelerator.unwrap_model(sim_cond_embedder).state_dict(),
                   os.path.join(save_dir, "sim_cond_embedder.pt"))
        with open(os.path.join(save_dir, "training_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)
        logger.info(f"Saved final model to {save_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()

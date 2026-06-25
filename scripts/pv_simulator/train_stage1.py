"""Stage 1 training for PV-Simulator: Simulation Branch only.

Trains the SimTransformer + SimConditionEmbedder on physics trajectory data
using LDM-style flow matching diffusion in the frozen CausalAE's latent
space (pre-trained in Stage 0). Raw states are encoded once to x_s_enc;
noise and the flow target (noise - x_s_enc) remain latent. By default the
loss is latent MSE; --use_raw_point_mse_target instead decodes the predicted
and target x0 point clouds through the frozen AE and uses correspondence-aware
raw-space per-point L2 loss.

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
import logging
import math
import os
import random
import shutil
import sys
import tempfile
from dataclasses import dataclass

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
from torch.utils.data import Dataset

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
from videox_fun.models.sim_transformer import SimTransformer
from videox_fun.models.physics_rigid_decoder import PhysicsConditionedRigidResidualDecoder
from videox_fun.utils.discrete_sampler import DiscreteSampling
from videox_fun.utils.sim_metrics import (
    ae_reconstruction_chamfer,
    frame0_error,
    knn_edge_error,
    velocity_drift,
)

logger = get_logger(__name__, log_level="INFO")


@dataclass
class RepeatedSampleDataset(Dataset):
    """Expose one base-dataset sample many times for overfit diagnostics."""

    base_dataset: Dataset
    sample_idx: int
    length: int

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.base_dataset[self.sample_idx]


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
    parser.add_argument("--overfit_single_sample_idx", type=int, default=None,
                        help="If set, train on only this sample index repeated many times. "
                             "Useful for single-sample overfit diagnostics.")
    parser.add_argument("--overfit_repeat_length", type=int, default=1024,
                        help="Logical dataset length when --overfit_single_sample_idx is used. "
                             "Keeps dataloader/epochs stable under distributed training.")

    # Pre-trained AE (from Stage 0)
    parser.add_argument("--ae_ckpt_dir", type=str, required=True,
                        help="Path to Stage 0 CausalAE checkpoint directory (e.g. outputs/ae/final)")

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
    parser.add_argument("--use_raw_transformer", action="store_true",
                        help="Stage-1-only ablation: operate directly on raw (position, velocity) "
                             "trajectories instead of AE-compressed state latents.")
    parser.add_argument("--center_on_contact", action="store_true",
                        help="For MOVi padded batches, crop each trajectory to a max_T_raw window "
                             "centred on its first metadata contact frame. Defaults to the initial window.")
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
    parser.add_argument("--dataloader_num_workers", type=int, default=4)

    # Diffusion
    parser.add_argument("--train_sampling_steps", type=int, default=1000)
    parser.add_argument("--uniform_sampling", action="store_true")
    parser.add_argument("--weighting_scheme", type=str, default="logit_normal",
                        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)

    # Shape-preserving inductive bias
    parser.add_argument("--lambda_diffusion", type=float, default=1.0,
                        help="Weight for latent-space flow-matching diffusion loss.")
    parser.add_argument("--use_raw_point_mse_target", action="store_true",
                        help="Replace latent flow-matching MSE with correspondence-aware per-point "
                             "L2 loss between AE-decoded predicted and target x0 positions.")
    parser.add_argument("--use_physics_conditioned_decoder", action="store_true",
                        help="Use the rigid SE(3) plus physics-gated residual coordinate decoder.")
    parser.add_argument("--lambda_physics_point", type=float, default=1.0,
                        help="Weight for the physics-decoder raw coordinate-space objective.")
    parser.add_argument("--lambda_chamfer_coarse", type=float, default=0.5,
                        help="Weight for coarse rigid-motion Chamfer supervision.")
    parser.add_argument("--lambda_residual_reg", type=float, default=0.1,
                        help="Weight for physics-conditioned deformability supervision.")
    parser.add_argument("--lambda_residual_mag", type=float, default=0.1,
                        help="Weight for softness-weighted residual magnitude regularization.")
    parser.add_argument("--lambda_local_dist", type=float, default=1e-3,
                        help="Weight for frame-0-relative local KNN edge deformation loss.")
    parser.add_argument("--lambda_covariance", type=float, default=0.0,
                        help="Weight for per-object raw-position covariance consistency loss.")
    parser.add_argument("--lambda_vel", type=float, default=0.1,
                        help="Weight for frame-to-frame raw-position velocity consistency loss.")
    parser.add_argument("--lambda_ang_vel", type=float, default=0.0,
                        help="Weight for per-object angular-velocity field consistency loss.")
    parser.add_argument("--lambda_velocity_vector", type=float, default=0.0,
                        help="Weight for direct per-point 3D velocity-vector supervision.")
    parser.add_argument("--lambda_velocity_accel", type=float, default=0.0,
                        help="Weight for temporal velocity-change consistency supervision.")
    parser.add_argument("--lambda_centroid", type=float, default=0.0,
                        help="Weight for per-object predicted/GT centroid consistency.")
    parser.add_argument("--lambda_rotation", type=float, default=0.0,
                        help="Weight for direct Kabsch rotation-matrix supervision.")
    parser.add_argument("--lambda_rotation_temporal", type=float, default=0.0,
                        help="Weight for relative Kabsch rotation consistency across frames.")
    parser.add_argument("--lambda_rotation_axis", type=float, default=0.0,
                        help="Weight for consecutive incremental Kabsch rotation-axis consistency.")
    parser.add_argument("--lambda_contact_com_velocity", type=float, default=0.0,
                        help="Weight for post-contact object-centroid finite-difference velocity matching.")
    parser.add_argument("--lambda_contact_pose_ang_speed", type=float, default=0.0,
                        help="Weight for post-contact Kabsch incremental angular-speed matching.")
    parser.add_argument("--contact_loss_post_frames", type=int, default=3,
                        help="Number of consecutive frame pairs starting at first contact used by contact losses.")
    parser.add_argument("--lambda_deformation_gradient", type=float, default=0.0,
                        help="Weight for local deformation-gradient matching on canonical k-NN neighborhoods.")
    parser.add_argument("--lambda_local_volume", type=float, default=0.0,
                        help="Weight for local log-volume-change matching derived from deformation gradients.")
    parser.add_argument("--lambda_local_covariance", type=float, default=0.0,
                        help="Weight for per-particle local covariance/anisotropy matching.")
    # Direction A: ramp auxiliary physics terms after the shape objective stabilizes.
    parser.add_argument("--physics_loss_warmup_steps", type=int, default=0,
                        help="Ramp physics auxiliary losses from 0 to 1 over this many optimizer steps. 0 disables.")
    # Direction B: staged parameter freezing for ablations.
    parser.add_argument("--training_stage", type=str, default="joint",
                        choices=["joint", "shape_only", "physics_only"],
                        help="joint=all trainable; shape_only=freeze decoder; physics_only=train output head plus decoder.")
    # Direction C: rotate predicted velocities into the GT Kabsch frame before vector MSE.
    parser.add_argument("--velocity_coord_align", action="store_true",
                        help="Align predicted velocity vectors to the GT object frame for velocity-vector supervision.")
    # Direction D: choose an independently testable subset of physics losses.
    parser.add_argument("--physics_mode", type=str, default="full",
                        choices=["full", "minimal", "shape_only"],
                        help="full=all enabled losses; minimal=rotation/centroid/local-dist only; shape_only=disable physics auxiliaries.")
    parser.add_argument("--minimal_include_momentum", action="store_true",
                        help="Include --lambda_momentum in physics_mode=minimal. Off by default "
                             "to preserve the original minimal-loss ablation.")
    parser.add_argument("--lambda_chamfer", type=float, default=0.01,
                        help="Weight for per-frame raw-position Chamfer distance loss.")
    parser.add_argument("--lambda_momentum", type=float, default=0.01,
                        help="Weight for per-object raw linear momentum consistency loss.")
    parser.add_argument("--lambda_floor", type=float, default=0.1,
                        help="Weight for non-static point floor-penetration loss.")
    parser.add_argument("--debug_raw_loss_gradients", action="store_true",
                        help="Assert that local-distance raw-loss gradients reach SimTransformer. "
                             "Use only for short gradient-path checks.")
    parser.add_argument("--gravity_axis", type=str, default="z", choices=["x", "y", "z"],
                        help="Coordinate axis normal to the floor for penetration loss.")
    parser.add_argument("--knn_k", type=int, default=8,
                        help="Number of same-object frame-0 KNN edges used in local distance loss.")
    # Backward-compatible alias for existing launch scripts.
    parser.add_argument("--local_dist_k", dest="knn_k", type=int, default=argparse.SUPPRESS,
                        help=argparse.SUPPRESS)
    parser.add_argument("--anchor_mode", type=str, default="local",
                        choices=["local"],
                        help="How to encode point anchors from the initial frame. "
                             "'local' uses centered world coordinates. "
                             "The earlier 'canonical_pca' variant is disabled because "
                             "its basis was unstable across samples.")
    parser.add_argument("--disable_point_anchor", action="store_true",
                        help="Disable point_anchor input entirely for ablation.")
    parser.add_argument("--use_temporal_correspondence", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Add an explicit same-point temporal attention block before the "
                            "flat Transformer. This tests whether point collapse is caused by "
                            "missing correspondence structure rather than by local distance loss. "
                            "Enabled by default; disable with --no-use-temporal-correspondence.")
    parser.add_argument("--use_temporal_rope", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Apply 1D RoPE inside the same-point temporal attention block. "
                            "This adds explicit frame-order information on each point track. "
                            "Enabled by default; disable with --no-use-temporal-rope.")
    parser.add_argument("--use_object_local_attention", action="store_true",
                        help="Restrict flat self-attention to tokens from the same object.")
    parser.add_argument("--use_factorized_attention", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Use spatial-then-temporal attention in every SimTransformer block. "
                             "Disable with --no-use-factorized-attention for global-attention ablations.")

    # Checkpointing & logging
    parser.add_argument("--checkpointing_steps", type=int, default=5000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=5)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="tensorboard", choices=["tensorboard", "wandb"])
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume a full Accelerator checkpoint (model, optimizer, scheduler, and RNG state).")
    parser.add_argument("--init_from_model_dir", type=str, default=None,
                        help="Initialize model weights from a directory containing sim_transformer.pt and "
                             "sim_cond_embedder.pt. Optimizer state is reset.")
    parser.add_argument("--initial_global_step", type=int, default=0,
                        help="Optimizer step represented by --init_from_model_dir; keeps epoch and LR schedule aligned.")
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
    parser.add_argument("--sim_metrics_steps", type=int, default=100,
                        help="Log detached raw-space simulation diagnostics every N optimizer steps. 0 = disabled.")

    args = parser.parse_args()

    if args.dataset_type == "simulation" and args.ann_path is None:
        parser.error("--ann_path is required when --dataset_type simulation")
    if args.overfit_single_sample_idx is not None and args.overfit_repeat_length < 1:
        parser.error("--overfit_repeat_length must be >= 1")
    if args.knn_k < 1:
        parser.error("--knn_k must be >= 1")
    if args.sim_metrics_steps < 0:
        parser.error("--sim_metrics_steps must be >= 0")
    if args.initial_global_step < 0:
        parser.error("--initial_global_step must be >= 0")
    if args.physics_loss_warmup_steps < 0:
        parser.error("--physics_loss_warmup_steps must be >= 0")
    if args.contact_loss_post_frames < 1:
        parser.error("--contact_loss_post_frames must be >= 1")
    if args.resume_from_checkpoint and args.init_from_model_dir:
        parser.error("Use only one of --resume_from_checkpoint and --init_from_model_dir")
    if args.initial_global_step and not args.init_from_model_dir:
        parser.error("--initial_global_step requires --init_from_model_dir")
    for name in (
        "lambda_diffusion", "lambda_vel", "lambda_ang_vel", "lambda_velocity_vector",
        "lambda_velocity_accel", "lambda_centroid", "lambda_rotation", "lambda_rotation_temporal",
        "lambda_rotation_axis", "lambda_contact_com_velocity", "lambda_contact_pose_ang_speed",
        "lambda_deformation_gradient", "lambda_local_volume", "lambda_local_covariance",
        "lambda_local_dist", "lambda_covariance", "lambda_chamfer",
        "lambda_momentum", "lambda_floor", "lambda_chamfer_coarse", "lambda_residual_reg",
        "lambda_residual_mag", "lambda_physics_point",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name} must be >= 0")

    return args


def _run_vis(vis_samples, sim_transformer, ae, sim_cond_embedder,
             global_step, device, weight_dtype, num_inference_steps, fps, anchor_mode,
             physics_decoder=None, use_raw_transformer=False):
    """Run inference on fixed samples, render GT+pred videos, and log to wandb.

    Must be called on main process only. Imports wandb and pipeline lazily so
    that training without wandb does not require these packages.
    """
    import wandb
    from diffusers import FlowMatchEulerDiscreteScheduler
    if not use_raw_transformer:
        from videox_fun.pipeline.pipeline_simulation import SimulationPipeline

    # visualize.py lives in the same directory — defer import so matplotlib
    # backend is set inside (it calls matplotlib.use('Agg') at import time).
    _vis_dir = os.path.dirname(os.path.abspath(__file__))
    if _vis_dir not in sys.path:
        sys.path.insert(0, _vis_dir)
    from visualize import visualize_point_cloud_motion

    pipeline = None
    if not use_raw_transformer:
        pipeline = SimulationPipeline(
            sim_transformer=sim_transformer,
            ae=ae,
            sim_cond_embedder=sim_cond_embedder,
            physics_decoder=physics_decoder,
            scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000),
            anchor_mode=anchor_mode,
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
            if use_raw_transformer:
                # Stage-1 raw-mode sampler.  The production SimulationPipeline
                # intentionally remains latent-only; this branch is only for
                # validating raw-transformer checkpoints during Stage 1.
                T = T_raw
                B, _, N, _ = x_s_raw.shape
                init_padded = x_s_init.expand(-1, T, -1, -1).contiguous()
                init_mask = torch.ones(B, T, N, 1, device=device, dtype=weight_dtype)
                init_mask[:, :1] = 0.0
                point_anchor = _compute_point_anchor(
                    x_s_init, point_obj_idx, anchor_mode,
                ).to(device=device, dtype=weight_dtype)
                if sim_transformer.d_anchor == 0:
                    point_anchor = point_anchor[..., :0]
                point_anchor = point_anchor.expand(-1, T, -1, -1).contiguous()
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=weight_dtype):
                    c_sim = sim_cond_embedder(
                        c_floor=c_floor,
                        c_id=c_id,
                        c_mat=c_mat,
                        c_mass=c_mass,
                        c_static=c_static,
                        c_force_enc=c_force_raw,
                        point_obj_idx=point_obj_idx,
                        T=T,
                        point_mask=None,
                    )
                    raw_sample = torch.randn(B, T, N, 6, device=device, dtype=weight_dtype)
                    raw_sample[:, :1] = x_s_init
                    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
                    scheduler.set_timesteps(num_inference_steps, device=device)
                    for timestep in scheduler.timesteps:
                        timestep_batch = timestep.unsqueeze(0)
                        pred_raw = sim_transformer(
                            raw_sample, init_padded, init_mask, point_anchor, c_sim,
                            timestep_batch, dtype=weight_dtype,
                        )
                        raw_sample = scheduler.step(
                            pred_raw, timestep, raw_sample,
                        ).prev_sample
                        raw_sample[:, :1] = x_s_init

                    if physics_decoder is not None:
                        physics_attrs = torch.ones(
                            B, physics_decoder.num_objects, 4,
                            device=device, dtype=torch.float32,
                        )
                        physics_attrs[:, :, 0] = 1.0
                        point_mask = torch.ones(B, N, device=device, dtype=torch.bool)
                        decoder_out = physics_decoder(
                            latent=raw_sample[..., :3].float(),
                            canonical_points=x_s_init[:, 0, :, :3].float(),
                            physics_attrs=physics_attrs,
                            point_obj_idx=point_obj_idx,
                            point_mask=point_mask,
                        )
                        raw_sample[..., :3] = torch.cat(
                            [raw_sample[:, :1, :, :3], decoder_out["positions"][:, 1:]],
                            dim=1,
                        ).to(raw_sample.dtype)
                x_s_pred = raw_sample[0].float().cpu()
            else:
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


def _compute_point_anchor(x_s_init: torch.Tensor, point_obj_idx: torch.Tensor, anchor_mode: str) -> torch.Tensor:
    """Build per-point anchors from the initial frame.

    Modes:
      - local: centered world coordinates
    Returns (B, 1, N, 3).
    """
    init_pos = x_s_init[..., :3]    # (B, 1, N, 3)
    anchor = torch.zeros_like(init_pos)
    B = init_pos.shape[0]
    for b in range(B):
        obj_ids = torch.unique(point_obj_idx[b])
        for obj_id in obj_ids.tolist():
            obj_mask = point_obj_idx[b] == obj_id
            if not torch.any(obj_mask):
                continue
            obj_pos = init_pos[b, 0, obj_mask]
            centroid = obj_pos.mean(dim=0, keepdim=True)
            centered = obj_pos - centroid
            # NOTE:
            # The previous canonical_pca branch is intentionally commented out.
            # For near-symmetric objects or noisy point samples, the PCA basis can
            # swap axes or flip signs across samples, which makes cross-sample point
            # identity less stable instead of more stable.
            #
            # if anchor_mode == "canonical_pca" and centered.shape[0] >= 3:
            #     cov = centered.T @ centered
            #     eigvals, eigvecs = torch.linalg.eigh(cov)
            #     order = torch.argsort(eigvals, descending=True)
            #     basis = eigvecs[:, order]
            #     proj = centered @ basis
            #     for ax_i in range(3):
            #         idx = torch.argmax(proj[:, ax_i].abs())
            #         sign = torch.sign(proj[idx, ax_i])
            #         if sign == 0:
            #             sign = centered.new_tensor(1.0)
            #         basis[:, ax_i] = basis[:, ax_i] * sign
            #         proj[:, ax_i] = proj[:, ax_i] * sign
            #     if torch.det(basis) < 0:
            #         basis[:, -1] = -basis[:, -1]
            #     centered = centered @ basis
            anchor[b, 0, obj_mask] = centered
    return anchor


def _compute_local_distance_loss(
    pos_pred: torch.Tensor,
    pos_gt: torch.Tensor,
    point_obj_idx: torch.Tensor,
    point_mask: torch.Tensor | None,
    valid_frame_mask: torch.Tensor | None,
    k: int,
) -> torch.Tensor:
    """Compare same-object KNN deformation ratios relative to frame 0.

    The graph is built on ground-truth frame 0. For every later frame, this
    compares ``log(pred_edge / pred_edge_0)`` with
    ``log(gt_edge / gt_edge_0)``. Thus the loss measures local deformation,
    rather than absolute world-coordinate edge error.
    """
    device = pos_gt.device
    total_loss = pos_gt.new_tensor(0.0)
    total_weight = pos_gt.new_tensor(0.0)
    B, T_raw, N, _ = pos_gt.shape

    for b in range(B):
        if point_mask is not None:
            valid_points = point_mask[b]
        else:
            valid_points = torch.ones(N, device=device, dtype=torch.bool)

        if valid_frame_mask is not None:
            valid_frames = valid_frame_mask[b]
        else:
            valid_frames = torch.ones(T_raw, device=device, dtype=torch.bool)
        frame_idx = valid_frames.nonzero(as_tuple=True)[0]
        frame_idx = frame_idx[frame_idx > 0]
        if frame_idx.numel() == 0:
            continue

        obj_ids = torch.unique(point_obj_idx[b][valid_points])
        for obj_id in obj_ids.tolist():
            obj_mask = (point_obj_idx[b] == obj_id) & valid_points
            obj_idx = obj_mask.nonzero(as_tuple=True)[0]
            n_obj = obj_idx.numel()
            if n_obj < 2:
                continue

            # KNN graph is built independently for each object from frame 0;
            # points from different objects can never become neighbors here.
            obj_gt0 = pos_gt[b, 0, obj_idx]  # (n_obj, 3)
            obj_gt = pos_gt[b, frame_idx][:, obj_idx]   # (T_valid, n_obj, 3)
            obj_pred = pos_pred[b, frame_idx][:, obj_idx]
            obj_pred0 = pos_pred[b, 0, obj_idx]

            pairwise = torch.cdist(obj_gt0, obj_gt0)
            pairwise.fill_diagonal_(torch.finfo(pairwise.dtype).max)
            k_eff = min(k, n_obj - 1)
            nbr_local = torch.topk(pairwise, k=k_eff, largest=False).indices  # (n_obj, k_eff)

            gt_edge0 = torch.linalg.vector_norm(
                obj_gt0.unsqueeze(1) - obj_gt0[nbr_local], dim=-1,
            )
            pred_edge0 = torch.linalg.vector_norm(
                obj_pred0.unsqueeze(1) - obj_pred0[nbr_local], dim=-1,
            )
            gt_edges = torch.linalg.vector_norm(
                obj_gt.unsqueeze(2) - obj_gt[:, nbr_local, :], dim=-1,
            )
            pred_edges = torch.linalg.vector_norm(
                obj_pred.unsqueeze(2) - obj_pred[:, nbr_local, :], dim=-1,
            )

            # Log-ratios penalize expansion and shrinkage symmetrically.
            gt_log_ratio = torch.log(gt_edges.clamp_min(1e-4) / gt_edge0.clamp_min(1e-4))
            pred_log_ratio = torch.log(pred_edges.clamp_min(1e-4) / pred_edge0.clamp_min(1e-4))
            obj_loss = (pred_log_ratio - gt_log_ratio).pow(2).mean()

            total_loss = total_loss + obj_loss * (obj_gt.shape[0] * n_obj * k_eff)
            total_weight = total_weight + (obj_gt.shape[0] * n_obj * k_eff)

    if total_weight.item() == 0:
        return pos_gt.new_tensor(0.0)
    return total_loss / total_weight


def _compute_covariance_loss(
    pos_pred: torch.Tensor,
    pos_gt: torch.Tensor,
    point_obj_idx: torch.Tensor,
    point_mask: torch.Tensor | None,
    valid_frame_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Match per-object 3×3 position covariance matrices after frame 0."""
    device = pos_gt.device
    total_loss = pos_gt.new_tensor(0.0)
    total_weight = pos_gt.new_tensor(0.0)
    B, T_raw, N, _ = pos_gt.shape

    for b in range(B):
        valid_points = (
            point_mask[b]
            if point_mask is not None
            else torch.ones(N, device=device, dtype=torch.bool)
        )
        valid_frames = (
            valid_frame_mask[b]
            if valid_frame_mask is not None
            else torch.ones(T_raw, device=device, dtype=torch.bool)
        )
        frame_idx = valid_frames.nonzero(as_tuple=True)[0]
        frame_idx = frame_idx[frame_idx > 0]
        if frame_idx.numel() == 0:
            continue

        for obj_id in torch.unique(point_obj_idx[b][valid_points]).tolist():
            obj_idx = ((point_obj_idx[b] == obj_id) & valid_points).nonzero(as_tuple=True)[0]
            n_obj = obj_idx.numel()
            if n_obj < 2:
                continue

            pred_points = pos_pred[b, frame_idx][:, obj_idx]
            gt_points = pos_gt[b, frame_idx][:, obj_idx]
            pred_centered = pred_points - pred_points.mean(dim=1, keepdim=True)
            gt_centered = gt_points - gt_points.mean(dim=1, keepdim=True)
            pred_cov = pred_centered.transpose(1, 2) @ pred_centered / (n_obj - 1)
            gt_cov = gt_centered.transpose(1, 2) @ gt_centered / (n_obj - 1)
            obj_loss = (pred_cov - gt_cov).square().mean()

            total_loss = total_loss + obj_loss * frame_idx.numel()
            total_weight = total_weight + frame_idx.numel()

    return total_loss / total_weight.clamp_min(1)


def _compute_velocity_consistency_loss(
    pos_pred: torch.Tensor,
    pos_gt: torch.Tensor,
    point_mask: torch.Tensor | None,
    valid_frame_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Match finite-difference position velocities on valid raw frames and points."""
    pred_vel = pos_pred[:, 1:] - pos_pred[:, :-1]
    gt_vel = pos_gt[:, 1:] - pos_gt[:, :-1]
    sq_error = (pred_vel - gt_vel).pow(2)

    B, T_minus_1, N, _ = sq_error.shape
    if point_mask is None:
        valid_points = torch.ones(B, N, device=pos_pred.device, dtype=torch.bool)
    else:
        valid_points = point_mask
    if valid_frame_mask is None:
        valid_pairs = torch.ones(B, T_minus_1, device=pos_pred.device, dtype=torch.bool)
    else:
        valid_pairs = valid_frame_mask[:, 1:] & valid_frame_mask[:, :-1]

    mask = valid_pairs.unsqueeze(-1) & valid_points.unsqueeze(1)
    return (sq_error * mask.unsqueeze(-1)).sum() / (mask.sum() * 3).clamp_min(1)


def _compute_angular_velocity_loss(
    pos_pred: torch.Tensor,
    velocity_pred: torch.Tensor,
    pos_gt: torch.Tensor,
    velocity_gt: torch.Tensor,
    point_obj_idx: torch.Tensor,
    point_mask: torch.Tensor | None,
    valid_frame_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Match angular velocity fitted from each object's 3D velocity field.

    For centred points ``r`` and centred velocities ``v``, fit
    ``v = omega x r`` by least squares.  Centring removes the translational
    component, so a uniform velocity field has angular velocity near zero.
    """
    device = pos_pred.device
    B, T_raw, N, _ = pos_pred.shape
    total_loss = pos_pred.new_tensor(0.0)
    total_weight = pos_pred.new_tensor(0.0)
    identity = torch.eye(3, device=device, dtype=pos_pred.dtype)

    def _fit_omega(positions: torch.Tensor, velocities: torch.Tensor) -> torch.Tensor:
        # positions/velocities: (F, P, 3), with P >= 3
        centred_positions = positions - positions.mean(dim=1, keepdim=True)
        centred_velocities = velocities - velocities.mean(dim=1, keepdim=True)
        rx, ry, rz = centred_positions.unbind(dim=-1)
        zeros = torch.zeros_like(rx)
        skew = torch.stack([
            zeros, -rz, ry,
            rz, zeros, -rx,
            -ry, rx, zeros,
        ], dim=-1).view(*centred_positions.shape[:-1], 3, 3)
        # omega x r == -[r]_x omega
        design = -skew
        normal = (design.transpose(-1, -2) @ design).sum(dim=1)
        rhs = (design.transpose(-1, -2) @ centred_velocities.unsqueeze(-1)).sum(dim=1)
        return torch.linalg.solve(normal + 1e-6 * identity, rhs).squeeze(-1)

    for b in range(B):
        valid_points = (
            point_mask[b]
            if point_mask is not None
            else torch.ones(N, device=device, dtype=torch.bool)
        )
        valid_frames = (
            valid_frame_mask[b]
            if valid_frame_mask is not None
            else torch.ones(T_raw, device=device, dtype=torch.bool)
        )
        frame_idx = valid_frames.nonzero(as_tuple=True)[0]
        frame_idx = frame_idx[frame_idx > 0]
        if frame_idx.numel() == 0:
            continue

        for object_id in torch.unique(point_obj_idx[b][valid_points]).tolist():
            object_idx = ((point_obj_idx[b] == object_id) & valid_points).nonzero(
                as_tuple=True,
            )[0]
            if object_idx.numel() < 3:
                continue

            omega_pred = _fit_omega(
                pos_pred[b, frame_idx][:, object_idx],
                velocity_pred[b, frame_idx][:, object_idx],
            )
            omega_gt = _fit_omega(
                pos_gt[b, frame_idx][:, object_idx],
                velocity_gt[b, frame_idx][:, object_idx],
            )
            frame_loss = F.mse_loss(omega_pred, omega_gt)
            total_loss = total_loss + frame_loss * frame_idx.numel()
            total_weight = total_weight + frame_idx.numel()

    return total_loss / total_weight.clamp_min(1)


def _compute_velocity_vector_loss(
    velocity_pred: torch.Tensor,
    velocity_gt: torch.Tensor,
    point_mask: torch.Tensor | None,
    valid_frame_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Direct masked 3D velocity-vector supervision after the given frame."""
    B, T_raw, N, _ = velocity_pred.shape
    if point_mask is None:
        valid_points = torch.ones(B, N, device=velocity_pred.device, dtype=torch.bool)
    else:
        valid_points = point_mask
    if valid_frame_mask is None:
        valid_frames = torch.ones(B, T_raw, device=velocity_pred.device, dtype=torch.bool)
    else:
        valid_frames = valid_frame_mask
    valid_frames = valid_frames.clone()
    valid_frames[:, :1] = False
    mask = valid_frames.unsqueeze(-1) & valid_points.unsqueeze(1)
    sq_error = (velocity_pred - velocity_gt).square()
    return (sq_error * mask.unsqueeze(-1)).sum() / (mask.sum() * 3).clamp_min(1)


def _compute_velocity_acceleration_loss(
    velocity_pred: torch.Tensor,
    velocity_gt: torch.Tensor,
    point_mask: torch.Tensor | None,
    valid_frame_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Match changes in per-point velocity without smoothing away GT dynamics."""
    B, T_raw, N, _ = velocity_pred.shape
    if T_raw < 3:
        return velocity_pred.sum() * 0.0
    if point_mask is None:
        valid_points = torch.ones(B, N, device=velocity_pred.device, dtype=torch.bool)
    else:
        valid_points = point_mask
    if valid_frame_mask is None:
        valid_triplets = torch.ones(
            B, T_raw - 1, device=velocity_pred.device, dtype=torch.bool,
        )
    else:
        valid_triplets = valid_frame_mask[:, 1:] & valid_frame_mask[:, :-1]
    valid_triplets = valid_triplets.clone()
    valid_triplets[:, :1] = False
    mask = valid_triplets.unsqueeze(-1) & valid_points.unsqueeze(1)
    pred_delta = velocity_pred[:, 1:] - velocity_pred[:, :-1]
    gt_delta = velocity_gt[:, 1:] - velocity_gt[:, :-1]
    sq_error = (pred_delta - gt_delta).square()
    return (sq_error * mask.unsqueeze(-1)).sum() / (mask.sum() * 3).clamp_min(1)


def _compute_centroid_consistency_loss(
    pos_pred: torch.Tensor,
    pos_gt: torch.Tensor,
    point_obj_idx: torch.Tensor,
    point_mask: torch.Tensor | None,
    valid_frame_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Match object centroids, distinct from canonical-frame displacement."""
    B, T_raw, N, _ = pos_pred.shape
    total_loss = pos_pred.new_tensor(0.0)
    total_weight = pos_pred.new_tensor(0.0)
    for b in range(B):
        valid_points = (
            point_mask[b]
            if point_mask is not None
            else torch.ones(N, device=pos_pred.device, dtype=torch.bool)
        )
        valid_frames = (
            valid_frame_mask[b]
            if valid_frame_mask is not None
            else torch.ones(T_raw, device=pos_pred.device, dtype=torch.bool)
        )
        frame_idx = valid_frames.nonzero(as_tuple=True)[0]
        frame_idx = frame_idx[frame_idx > 0]
        for object_id in torch.unique(point_obj_idx[b][valid_points]).tolist():
            object_idx = ((point_obj_idx[b] == object_id) & valid_points).nonzero(
                as_tuple=True,
            )[0]
            if object_idx.numel() == 0 or frame_idx.numel() == 0:
                continue
            pred_centroid = pos_pred[b, frame_idx][:, object_idx].mean(dim=1)
            gt_centroid = pos_gt[b, frame_idx][:, object_idx].mean(dim=1)
            total_loss = total_loss + (pred_centroid - gt_centroid).square().mean() * frame_idx.numel()
            total_weight = total_weight + frame_idx.numel()
    return total_loss / total_weight.clamp_min(1)


def _contact_post_pair_mask(
    c_force_raw: torch.Tensor,
    point_mask: torch.Tensor | None,
    valid_frame_mask: torch.Tensor | None,
    post_frames: int,
) -> torch.Tensor:
    """Select consecutive frame pairs immediately following the first contact.

    ``c_force_raw[..., :3]`` is the per-point applied-force vector.  A frame
    is a contact frame when any valid point has a nonzero force.  For a first
    contact at frame ``c``, this returns pairs ``(c, c + 1)``, ...,
    ``(c + post_frames - 1, c + post_frames)`` when those frames exist.
    Samples with no in-window contact have no selected pairs.
    """
    B, T_raw, _, _ = c_force_raw.shape
    pair_mask = torch.zeros(
        B, max(T_raw - 1, 0), device=c_force_raw.device, dtype=torch.bool,
    )
    if T_raw < 2:
        return pair_mask

    # Contact is an input condition, not a learned quantity: detach avoids
    # retaining an unnecessary autograd path through the condition tensor.
    force_norm = c_force_raw[..., :3].detach().norm(dim=-1)
    if point_mask is not None:
        force_norm = force_norm.masked_fill(~point_mask[:, None, :], 0.0)
    contact_frames = force_norm.amax(dim=-1) > 1e-8
    if valid_frame_mask is not None:
        contact_frames = contact_frames & valid_frame_mask
        valid_pairs = valid_frame_mask[:, :-1] & valid_frame_mask[:, 1:]
    else:
        valid_pairs = torch.ones_like(pair_mask)

    for b in range(B):
        first_contact = contact_frames[b].nonzero(as_tuple=True)[0]
        if first_contact.numel() == 0:
            continue
        start = int(first_contact[0].item())
        end = min(start + post_frames, T_raw - 1)
        pair_mask[b, start:end] = True
    return pair_mask & valid_pairs


def _compute_contact_com_velocity_loss(
    pos_pred: torch.Tensor,
    pos_gt: torch.Tensor,
    c_force_raw: torch.Tensor,
    point_obj_idx: torch.Tensor,
    point_mask: torch.Tensor | None,
    valid_frame_mask: torch.Tensor | None,
    num_objects: int,
    post_frames: int,
) -> torch.Tensor:
    """Match post-contact object-centroid finite-difference velocities.

    For object centroid ``c_t``, the supervised quantity is
    ``v_com,t = c_(t+1) - c_t``.  This uses position channels only, so it
    directly couples the predicted rolling pose trajectory to translation.
    """
    B, _, N, _ = pos_pred.shape
    pair_mask = _contact_post_pair_mask(
        c_force_raw, point_mask, valid_frame_mask, post_frames,
    )
    total_loss = pos_pred.sum() * 0.0
    total_weight = pos_pred.new_tensor(0.0)
    for b in range(B):
        selected_pairs = pair_mask[b]
        if not torch.any(selected_pairs):
            continue
        valid_points = (
            point_mask[b]
            if point_mask is not None
            else torch.ones(N, device=pos_pred.device, dtype=torch.bool)
        )
        for object_id in range(num_objects):
            object_points = (point_obj_idx[b] == object_id) & valid_points
            if not torch.any(object_points):
                continue
            pred_centroid = pos_pred[b, :, object_points].mean(dim=1)
            gt_centroid = pos_gt[b, :, object_points].mean(dim=1)
            pred_velocity = pred_centroid[1:] - pred_centroid[:-1]
            gt_velocity = gt_centroid[1:] - gt_centroid[:-1]
            error = (pred_velocity - gt_velocity).square().mean(dim=-1)
            total_loss = total_loss + error[selected_pairs].sum()
            total_weight = total_weight + selected_pairs.float().sum()
    return total_loss / total_weight.clamp_min(1)


def _row_kabsch_incremental_angle(
    source: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor | None:
    """Return the row-vector Kabsch angle for one same-object frame pair.

    With ``H = source.T @ target = U S V^T``, the row-vector map is
    ``R = U D V^T`` such that ``target ~= source @ R + t``.  Degenerate fits
    are omitted rather than injecting an arbitrary identity-angle target.
    """
    if source.shape[0] < 3:
        return None
    source_centered = source - source.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    covariance = source_centered.transpose(0, 1) @ target_centered
    U, singular_values, Vh = torch.linalg.svd(covariance, full_matrices=False)
    if singular_values[-1].detach().item() < 1e-6:
        return None
    correction = torch.eye(3, device=source.device, dtype=source.dtype)
    correction[-1, -1] = torch.where(
        torch.det(U @ Vh) < 0,
        correction.new_tensor(-1.0),
        correction.new_tensor(1.0),
    )
    rotation = U @ correction @ Vh
    trace = rotation.diagonal().sum()
    # ``acos`` has an infinite derivative at +/-1.  Keep the requested trace
    # clamp, with a tiny interior margin, so a stationary predicted pair does
    # not turn this auxiliary loss into NaN gradients.
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cosine)


def _compute_contact_pose_ang_speed_loss(
    pos_pred: torch.Tensor,
    pos_gt: torch.Tensor,
    c_force_raw: torch.Tensor,
    point_obj_idx: torch.Tensor,
    point_mask: torch.Tensor | None,
    valid_frame_mask: torch.Tensor | None,
    num_objects: int,
    post_frames: int,
    dt: float = 1.0 / 12.0,
) -> torch.Tensor:
    """Match post-contact Kabsch incremental angular speeds from positions.

    For each selected pair, independently fit row-vector Kabsch transforms
    from frame ``t`` to ``t + 1`` and compare ``acos((tr(R)-1)/2) / dt``.
    The fixed ``dt=1/12`` matches the MOVi sampling rate used by this setup.
    """
    B, _, N, _ = pos_pred.shape
    pair_mask = _contact_post_pair_mask(
        c_force_raw, point_mask, valid_frame_mask, post_frames,
    )
    total_loss = pos_pred.sum() * 0.0
    total_weight = pos_pred.new_tensor(0.0)
    for b in range(B):
        selected_pair_idx = pair_mask[b].nonzero(as_tuple=True)[0]
        if selected_pair_idx.numel() == 0:
            continue
        valid_points = (
            point_mask[b]
            if point_mask is not None
            else torch.ones(N, device=pos_pred.device, dtype=torch.bool)
        )
        for object_id in range(num_objects):
            object_points = (point_obj_idx[b] == object_id) & valid_points
            if int(object_points.sum().item()) < 3:
                continue
            for frame_idx in selected_pair_idx.tolist():
                pred_angle = _row_kabsch_incremental_angle(
                    pos_pred[b, frame_idx, object_points],
                    pos_pred[b, frame_idx + 1, object_points],
                )
                gt_angle = _row_kabsch_incremental_angle(
                    pos_gt[b, frame_idx, object_points],
                    pos_gt[b, frame_idx + 1, object_points],
                )
                if pred_angle is None or gt_angle is None:
                    continue
                pred_speed = pred_angle / dt
                gt_speed = gt_angle / dt
                total_loss = total_loss + (pred_speed - gt_speed).square()
                total_weight = total_weight + 1
    return total_loss / total_weight.clamp_min(1)


def _fit_kabsch_rotations(
    canonical_points: torch.Tensor,
    positions: torch.Tensor,
    point_obj_idx: torch.Tensor,
    point_mask: torch.Tensor | None,
    num_objects: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return row-vector Kabsch rotations and a mask of non-degenerate fits.

    The returned ``R`` follows ``target_points = source_points @ R + t``.
    Equivalently, the physics decoder's column-vector convention stores
    ``R_column = R.T`` and applies ``source_points @ R_column.T + t``.
    """
    B, T_raw, N, _ = positions.shape
    identity = torch.eye(3, device=positions.device, dtype=positions.dtype)
    rotations = identity.view(1, 1, 1, 3, 3).expand(
        B, T_raw, num_objects, -1, -1,
    ).clone()
    valid_rotation = torch.zeros(
        B, T_raw, num_objects, device=positions.device, dtype=torch.bool,
    )
    for b in range(B):
        valid_points = (
            point_mask[b]
            if point_mask is not None
            else torch.ones(N, device=positions.device, dtype=torch.bool)
        )
        for object_id in range(num_objects):
            object_points = (point_obj_idx[b] == object_id) & valid_points
            if int(object_points.sum().item()) < 3:
                continue
            source = canonical_points[b, object_points]
            source_centered = source - source.mean(dim=0)
            target = positions[b, :, object_points]
            target_centered = target - target.mean(dim=1, keepdim=True)
            covariance = torch.einsum("pi,tpj->tij", source_centered, target_centered)
            U, singular_values, Vh = torch.linalg.svd(covariance, full_matrices=False)
            non_degenerate = singular_values[:, -1] >= 1e-6
            if not torch.any(non_degenerate):
                continue
            U_valid = U[non_degenerate]
            Vh_valid = Vh[non_degenerate]
            correction = identity.expand(U_valid.shape[0], -1, -1).clone()
            correction[:, 2, 2] = torch.where(
                torch.det(U_valid @ Vh_valid) < 0,
                correction.new_tensor(-1.0),
                correction.new_tensor(1.0),
            )
            # Row-vector Kabsch: H = source^T target = U S V^T, therefore
            # target_points = source_points @ (U D V^T) + t.
            rotations[b, non_degenerate, object_id] = (
                U_valid @ correction @ Vh_valid
            )
            valid_rotation[b, non_degenerate, object_id] = True
    return rotations, valid_rotation


def _compute_rotation_consistency_losses(
    pos_pred: torch.Tensor,
    pos_gt: torch.Tensor,
    canonical_points: torch.Tensor,
    point_obj_idx: torch.Tensor,
    point_mask: torch.Tensor | None,
    valid_frame_mask: torch.Tensor | None,
    num_objects: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match absolute and frame-to-frame Kabsch rotations to GT."""
    rotations_pred, valid_pred = _fit_kabsch_rotations(
        canonical_points, pos_pred, point_obj_idx, point_mask, num_objects,
    )
    rotations_gt, valid_gt = _fit_kabsch_rotations(
        canonical_points, pos_gt, point_obj_idx, point_mask, num_objects,
    )
    valid = valid_pred & valid_gt
    if valid_frame_mask is not None:
        valid = valid & valid_frame_mask[:, :, None]
    valid = valid.clone()
    valid[:, :1] = False
    absolute_error = (rotations_pred - rotations_gt).square().mean(dim=(-1, -2))
    absolute_loss = (absolute_error * valid.float()).sum() / valid.float().sum().clamp_min(1)

    if rotations_pred.shape[1] < 2:
        return absolute_loss, absolute_loss * 0.0
    # For row-vector canonical-to-frame maps, the t -> t+1 map is R_t^T R_{t+1}.
    relative_pred = rotations_pred[:, :-1].transpose(-1, -2) @ rotations_pred[:, 1:]
    relative_gt = rotations_gt[:, :-1].transpose(-1, -2) @ rotations_gt[:, 1:]
    valid_pairs = valid[:, 1:] & valid[:, :-1]
    temporal_error = (relative_pred - relative_gt).square().mean(dim=(-1, -2))
    temporal_loss = (
        (temporal_error * valid_pairs.float()).sum()
        / valid_pairs.float().sum().clamp_min(1)
    )
    return absolute_loss, temporal_loss


def _compute_rotation_axis_loss(
    pos_pred: torch.Tensor,
    canonical_points: torch.Tensor,
    point_obj_idx: torch.Tensor,
    point_mask: torch.Tensor | None,
    valid_frame_mask: torch.Tensor | None,
    num_objects: int,
    min_angle: float = 1e-3,
) -> torch.Tensor:
    """Penalize sign flips between consecutive *incremental* rotation axes.

    Kabsch rotations map canonical row-vectors to each raw predicted frame. The
    relevant rolling direction is therefore the relative transform
    ``R[t]^T @ R[t+1]``, not the cumulative canonical-to-frame rotation. Its
    Rodrigues axis is signed, so alternating forward/backward rolling yields
    antiparallel adjacent axes and a loss near 2.  Near-zero-angle increments
    have no stable axis and are excluded from the average.
    """
    rotations, valid_rotation = _fit_kabsch_rotations(
        canonical_points, pos_pred, point_obj_idx, point_mask, num_objects,
    )
    if rotations.shape[1] < 3:
        return pos_pred.new_tensor(0.0)

    relative = rotations[:, :-1].transpose(-1, -2) @ rotations[:, 1:]
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    angles = torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))
    # vee(R - R^T) = 2 sin(theta) * axis.  Normalizing the vee vector is
    # numerically safer than dividing by sin(theta), except when it vanishes.
    axis_vector = torch.stack(
        [
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ],
        dim=-1,
    )
    axis_norm = axis_vector.norm(dim=-1)
    valid_increment = valid_rotation[:, 1:] & valid_rotation[:, :-1]
    if valid_frame_mask is not None:
        valid_increment = (
            valid_increment
            & valid_frame_mask[:, 1:, None]
            & valid_frame_mask[:, :-1, None]
        )
    valid_increment = (
        valid_increment
        & (angles > min_angle)
        & (axis_norm > 1e-6)
    )
    axes = axis_vector / axis_norm.unsqueeze(-1).clamp_min(1e-6)

    # Compare intervals (t -> t+1) and (t+1 -> t+2).
    valid_axis_pairs = valid_increment[:, 1:] & valid_increment[:, :-1]
    if not torch.any(valid_axis_pairs):
        return pos_pred.new_tensor(0.0)
    axis_dot = (axes[:, 1:] * axes[:, :-1]).sum(dim=-1).clamp(-1.0, 1.0)
    return (
        ((1.0 - axis_dot) * valid_axis_pairs.float()).sum()
        / valid_axis_pairs.float().sum().clamp_min(1)
    )


def _align_velocity_to_gt_frame(
    velocity_pred: torch.Tensor,
    pos_pred: torch.Tensor,
    pos_gt: torch.Tensor,
    canonical_points: torch.Tensor,
    point_obj_idx: torch.Tensor,
    point_mask: torch.Tensor | None,
    num_objects: int,
) -> torch.Tensor:
    """Direction C: map predicted velocity vectors into each GT object frame.

    If row-vector R_pred and R_gt map canonical coordinates to predicted and
    GT frames, ``A = R_pred^T @ R_gt`` maps a predicted-frame vector into the
    GT frame via ``v_gt = v_pred @ A``.
    Translation is deliberately absent because velocity vectors are translationally
    invariant.
    """
    rotations_pred, valid_pred = _fit_kabsch_rotations(
        canonical_points, pos_pred, point_obj_idx, point_mask, num_objects,
    )
    rotations_gt, valid_gt = _fit_kabsch_rotations(
        canonical_points, pos_gt, point_obj_idx, point_mask, num_objects,
    )
    alignment = rotations_pred.transpose(-1, -2) @ rotations_gt
    aligned_velocity = velocity_pred.clone()
    B, T_raw, N, _ = velocity_pred.shape
    for b in range(B):
        valid_points = (
            point_mask[b]
            if point_mask is not None
            else torch.ones(N, device=velocity_pred.device, dtype=torch.bool)
        )
        for object_id in range(num_objects):
            if not torch.any(valid_pred[b, :, object_id] & valid_gt[b, :, object_id]):
                continue
            object_points = (point_obj_idx[b] == object_id) & valid_points
            if not torch.any(object_points):
                continue
            aligned_velocity[b, :, object_points] = torch.matmul(
                velocity_pred[b, :, object_points],
                alignment[b, :, object_id],
            )
    return aligned_velocity


def _compute_local_geometry_losses(
    pos_pred: torch.Tensor,
    pos_gt: torch.Tensor,
    point_obj_idx: torch.Tensor,
    point_mask: torch.Tensor | None,
    valid_frame_mask: torch.Tensor | None,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match local deformation, volume, and covariance on canonical k-NN charts.

    For reference offsets dX from GT frame 0 and current offsets dx,
    ``F = sum_j dx_j outer dX_j @ (sum_j dX_j outer dX_j + eps I)^-1``.
    ``det(F)`` is the local volume-change proxy and
    ``C = mean_j(dx_j outer dx_j)`` is a local covariance/anisotropy field.
    """
    device = pos_pred.device
    B, T_raw, N, _ = pos_pred.shape
    deformation_total = pos_pred.new_tensor(0.0)
    volume_total = pos_pred.new_tensor(0.0)
    covariance_total = pos_pred.new_tensor(0.0)
    total_weight = pos_pred.new_tensor(0.0)
    identity = torch.eye(3, device=device, dtype=pos_pred.dtype)

    for b in range(B):
        valid_points = (
            point_mask[b]
            if point_mask is not None
            else torch.ones(N, device=device, dtype=torch.bool)
        )
        valid_frames = (
            valid_frame_mask[b]
            if valid_frame_mask is not None
            else torch.ones(T_raw, device=device, dtype=torch.bool)
        )
        frame_idx = valid_frames.nonzero(as_tuple=True)[0]
        frame_idx = frame_idx[frame_idx > 0]
        if frame_idx.numel() == 0:
            continue

        for object_id in torch.unique(point_obj_idx[b][valid_points]).tolist():
            object_idx = ((point_obj_idx[b] == object_id) & valid_points).nonzero(
                as_tuple=True,
            )[0]
            num_points = object_idx.numel()
            if num_points < 2:
                continue

            reference = pos_gt[b, 0, object_idx]
            pairwise = torch.cdist(reference, reference)
            pairwise.fill_diagonal_(torch.finfo(pairwise.dtype).max)
            k_eff = min(k, num_points - 1)
            neighbors = torch.topk(pairwise, k=k_eff, largest=False).indices
            reference_offsets = reference.unsqueeze(1) - reference[neighbors]

            # (P, 3, 3), regularized local reference chart inverse.
            reference_chart = torch.einsum(
                "pki,pkj->pij", reference_offsets, reference_offsets,
            ) / k_eff
            reference_chart_inv = torch.linalg.inv(reference_chart + 1e-6 * identity)

            pred_points = pos_pred[b, frame_idx][:, object_idx]
            gt_points = pos_gt[b, frame_idx][:, object_idx]
            pred_offsets = pred_points.unsqueeze(2) - pred_points[:, neighbors]
            gt_offsets = gt_points.unsqueeze(2) - gt_points[:, neighbors]

            pred_cross = torch.einsum("fpki,pkj->fpij", pred_offsets, reference_offsets) / k_eff
            gt_cross = torch.einsum("fpki,pkj->fpij", gt_offsets, reference_offsets) / k_eff
            F_pred = pred_cross @ reference_chart_inv
            F_gt = gt_cross @ reference_chart_inv

            covariance_pred = torch.einsum("fpki,fpkj->fpij", pred_offsets, pred_offsets) / k_eff
            covariance_gt = torch.einsum("fpki,fpkj->fpij", gt_offsets, gt_offsets) / k_eff
            log_volume_pred = torch.log(torch.linalg.det(F_pred).abs().clamp_min(1e-6))
            log_volume_gt = torch.log(torch.linalg.det(F_gt).abs().clamp_min(1e-6))

            weight = frame_idx.numel() * num_points
            deformation_total = deformation_total + (F_pred - F_gt).square().mean() * weight
            volume_total = volume_total + (log_volume_pred - log_volume_gt).square().mean() * weight
            covariance_total = covariance_total + (covariance_pred - covariance_gt).square().mean() * weight
            total_weight = total_weight + weight

    normalizer = total_weight.clamp_min(1)
    return (
        deformation_total / normalizer,
        volume_total / normalizer,
        covariance_total / normalizer,
    )


def _compute_momentum_loss(
    velocity_pred: torch.Tensor,
    velocity_gt: torch.Tensor,
    c_mass: torch.Tensor,
    point_obj_idx: torch.Tensor,
    point_mask: torch.Tensor | None,
    valid_frame_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Match total linear momentum per object after uniform per-point mass split.

    ``c_mass[b, obj_id]`` is the object mass.  Surface samples have no
    individual masses in MOVi, so the mass is deliberately distributed as
    ``m_object / n_valid_object_points``.  Both padded points and padded frames
    are excluded before the sum.  This is a world-frame velocity comparison;
    force/contact conditioning is not used by this loss and cannot introduce a
    coordinate-frame mismatch in raw-transformer mode.
    """
    device = velocity_pred.device
    total_loss = velocity_pred.new_tensor(0.0)
    total_weight = velocity_pred.new_tensor(0.0)
    B, T_raw, N, _ = velocity_pred.shape

    for b in range(B):
        valid_points = (
            point_mask[b]
            if point_mask is not None
            else torch.ones(N, device=device, dtype=torch.bool)
        )
        valid_frames = (
            valid_frame_mask[b]
            if valid_frame_mask is not None
            else torch.ones(T_raw, device=device, dtype=torch.bool)
        )
        frame_idx = valid_frames.nonzero(as_tuple=True)[0]
        if frame_idx.numel() == 0:
            continue

        for obj_id in torch.unique(point_obj_idx[b][valid_points]).tolist():
            point_idx = ((point_obj_idx[b] == obj_id) & valid_points).nonzero(as_tuple=True)[0]
            n_obj_points = point_idx.numel()
            if n_obj_points == 0:
                continue

            # Each object's scalar mass is distributed uniformly over its valid points.
            point_mass = c_mass[b, obj_id].to(dtype=velocity_pred.dtype) / n_obj_points
            pred_momentum = (velocity_pred[b, frame_idx][:, point_idx] * point_mass).sum(dim=1)
            gt_momentum = (velocity_gt[b, frame_idx][:, point_idx] * point_mass).sum(dim=1)
            total_loss = total_loss + (pred_momentum - gt_momentum).square().sum()
            total_weight = total_weight + frame_idx.numel()

    return total_loss / total_weight.clamp_min(1)


def _compute_floor_penetration_loss(
    pos_pred: torch.Tensor,
    c_floor: torch.Tensor,
    c_static: torch.Tensor,
    point_obj_idx: torch.Tensor,
    gravity_axis: str,
    point_mask: torch.Tensor | None,
    valid_frame_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Penalize non-static valid points below the configured floor plane."""
    axis_idx = {"x": 0, "y": 1, "z": 2}[gravity_axis]
    B, T_raw, N, _ = pos_pred.shape
    if point_mask is None:
        valid_points = torch.ones(B, N, device=pos_pred.device, dtype=torch.bool)
    else:
        valid_points = point_mask
    if valid_frame_mask is None:
        valid_frames = torch.ones(B, T_raw, device=pos_pred.device, dtype=torch.bool)
    else:
        valid_frames = valid_frame_mask

    point_static = c_static.gather(1, point_obj_idx).bool()
    active_points = valid_points & ~point_static
    floor_height = c_floor.to(dtype=pos_pred.dtype).view(B, 1, 1)
    penetration_sq = (floor_height - pos_pred[..., axis_idx]).relu().square()
    mask = valid_frames.unsqueeze(-1) & active_points.unsqueeze(1)
    return (penetration_sq * mask).sum() / mask.sum().clamp_min(1)


def _compute_chamfer_loss(
    pos_pred: torch.Tensor,
    pos_gt: torch.Tensor,
    point_mask: torch.Tensor | None,
    valid_frame_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Mean symmetric squared Chamfer distance over valid raw point-cloud frames."""
    return _compute_chamfer_loss_per_sample(
        pos_pred, pos_gt, point_mask, valid_frame_mask,
    ).mean()


def _compute_chamfer_loss_per_sample(
    pos_pred: torch.Tensor,
    pos_gt: torch.Tensor,
    point_mask: torch.Tensor | None,
    valid_frame_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Symmetric squared Chamfer distance per sample, averaged over valid frames."""
    B, T_raw, N, _ = pos_gt.shape
    per_sample_losses = []

    for b in range(B):
        total_loss = pos_gt.new_tensor(0.0)
        total_weight = pos_gt.new_tensor(0.0)
        if point_mask is None:
            valid_points = torch.ones(N, device=pos_gt.device, dtype=torch.bool)
        else:
            valid_points = point_mask[b]
        if not torch.any(valid_points):
            continue

        valid_frames = (
            valid_frame_mask[b]
            if valid_frame_mask is not None
            else torch.ones(T_raw, device=pos_gt.device, dtype=torch.bool)
        )
        for frame_idx in valid_frames.nonzero(as_tuple=True)[0]:
            pred_points = pos_pred[b, frame_idx, valid_points]
            gt_points = pos_gt[b, frame_idx, valid_points]
            sq_dist = torch.cdist(pred_points, gt_points).square()
            frame_loss = 0.5 * (
                sq_dist.min(dim=1).values.mean() + sq_dist.min(dim=0).values.mean()
            )
            total_loss = total_loss + frame_loss
            total_weight = total_weight + 1

        if total_weight.item() == 0:
            # Preserve a valid zero-gradient path for empty padded samples.
            per_sample_losses.append(pos_pred[b].sum() * 0.0)
        else:
            per_sample_losses.append(total_loss / total_weight)

    return torch.stack(per_sample_losses)


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

    # --- Load frozen CausalAE (from Stage 0) ---
    ae = CausalAE.load(args.ae_ckpt_dir)
    ae.to(accelerator.device, dtype=weight_dtype)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    d_latent = ae.d_latent
    d_state_actual = 6 if args.use_raw_transformer else 2 * d_latent
    if args.d_state != d_state_actual:
        logger.warning(
            f"--d_state={args.d_state} overridden to {d_state_actual} "
            + (
                "for --use_raw_transformer"
                if args.use_raw_transformer
                else f"(2 * AE d_latent={d_latent})"
            )
        )
        args.d_state = d_state_actual
    logger.info(
        "Loaded frozen CausalAE from %s (d_latent=%s); Stage-1 state representation=%s",
        args.ae_ckpt_dir,
        d_latent,
        "raw" if args.use_raw_transformer else "latent",
    )

    # --- Build Models ---
    # Raw mode keeps force/contact at the native temporal resolution so that
    # c_sim has the same 21-frame timeline as the raw state trajectory.
    sim_cond_embedder = SimConditionEmbedder(
        max_objects=args.max_objects,
        d_force=6 if args.use_raw_transformer else 2 * d_latent,
    )
    d_cond = sim_cond_embedder.d_cond

    sim_transformer = SimTransformer(
        d_state=args.d_state,
        d_cond=d_cond,
        d_anchor=0 if args.disable_point_anchor else 3,
        d_sim=args.d_sim,
        ffn_dim=args.sim_ffn_dim,
        num_heads=args.sim_num_heads,
        num_layers=args.sim_num_layers,
        use_factorized_attention=args.use_factorized_attention,
        use_temporal_correspondence=args.use_temporal_correspondence,
        use_temporal_rope=args.use_temporal_rope,
        use_object_local_attention=args.use_object_local_attention,
        input_representation="raw" if args.use_raw_transformer else "latent",
    )

    physics_decoder = None
    if args.use_physics_conditioned_decoder:
        physics_decoder = PhysicsConditionedRigidResidualDecoder(
            num_objects=args.max_objects,
        )
        logger.info(
            "Physics-conditioned rigid residual decoder enabled: "
            "lambda_chamfer_coarse=%g, lambda_residual_reg=%g, lambda_residual_mag=%g",
            args.lambda_chamfer_coarse,
            args.lambda_residual_reg,
            args.lambda_residual_mag,
        )
    logger.info(
        "Temporal structure: sinusoidal token positions=on, correspondence=%s, RoPE=%s, "
        "factorized_attention=%s, object_local_attention=%s, "
        "raw_point_mse_target=%s, lambda_diffusion=%g, lambda_local_dist=%g, lambda_covariance=%g, lambda_vel=%g, lambda_chamfer=%g, "
        "lambda_momentum=%g, lambda_floor=%g(axis=%s)",
        args.use_temporal_correspondence,
        args.use_temporal_rope,
        args.use_factorized_attention,
        args.use_object_local_attention,
        args.use_raw_point_mse_target,
        args.lambda_diffusion,
        args.lambda_local_dist,
        args.lambda_covariance,
        args.lambda_vel,
        args.lambda_chamfer,
        args.lambda_momentum,
        args.lambda_floor,
        args.gravity_axis,
    )

    if args.init_from_model_dir:
        transformer_path = os.path.join(args.init_from_model_dir, "sim_transformer.pt")
        cond_embedder_path = os.path.join(args.init_from_model_dir, "sim_cond_embedder.pt")
        checkpoint_paths = [transformer_path, cond_embedder_path]

        if args.use_physics_conditioned_decoder:
            physics_decoder_path = os.path.join(
                args.init_from_model_dir, "physics_decoder.pt"
            )
            checkpoint_paths.append(physics_decoder_path)

        for checkpoint_path in checkpoint_paths:
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f"Model-only initialization requires {checkpoint_path}"
                )

        sim_transformer.load_state_dict(torch.load(transformer_path, map_location="cpu"))
        sim_cond_embedder.load_state_dict(torch.load(cond_embedder_path, map_location="cpu"))
        if args.use_physics_conditioned_decoder:
            physics_decoder.load_state_dict(
                torch.load(physics_decoder_path, map_location="cpu")
            )

        logger.info(
            "Initialized Stage 1 model weights from %s at optimizer step %s; "
            "optimizer state will be reset.",
            args.init_from_model_dir,
            args.initial_global_step,
        )

    # Move to device
    sim_cond_embedder.to(accelerator.device, dtype=weight_dtype)
    sim_transformer.to(accelerator.device, dtype=weight_dtype)
    if physics_decoder is not None:
        # Keep this module FP32 because its specified physics attributes are FP32.
        physics_decoder.to(accelerator.device)

    # Direction B: parameter-stage ablations. The rigid Kabsch transform is
    # parameter-free, so physics_only retains SimTransformer's output head to
    # preserve a gradient path from coordinate-space physics losses.
    if args.training_stage == "shape_only" and physics_decoder is not None:
        for parameter in physics_decoder.parameters():
            parameter.requires_grad_(False)
        logger.info("training_stage=shape_only: physics decoder parameters frozen")
    elif args.training_stage == "physics_only":
        for parameter in sim_transformer.parameters():
            parameter.requires_grad_(False)
        for parameter in sim_cond_embedder.parameters():
            parameter.requires_grad_(False)
        output_head_params = [
            parameter
            for name, parameter in sim_transformer.named_parameters()
            if name.startswith("head_proj")
        ]
        if not output_head_params:
            raise RuntimeError(
                "training_stage=physics_only requires SimTransformer.head_proj parameters"
            )
        for parameter in output_head_params:
            parameter.requires_grad_(True)
        if physics_decoder is not None:
            for parameter in physics_decoder.parameters():
                parameter.requires_grad_(True)
        logger.info(
            "training_stage=physics_only: frozen transformer trunk/condition embedder; "
            "training output head and decoder parameters"
        )

    # Count parameters
    total_params = (
        sum(p.numel() for p in sim_transformer.parameters()) +
        sum(p.numel() for p in sim_cond_embedder.parameters()) +
        (
            sum(p.numel() for p in physics_decoder.parameters())
            if physics_decoder is not None
            else 0
        )
    )
    logger.info(f"Total trainable parameters: {total_params:,}")

    # --- Optimizer ---
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = [
        parameter
        for parameter in (
            list(sim_transformer.parameters()) + list(sim_cond_embedder.parameters())
        )
        if parameter.requires_grad
    ]
    if physics_decoder is not None:
        trainable_params += [
            parameter for parameter in physics_decoder.parameters() if parameter.requires_grad
        ]
    if not trainable_params:
        raise RuntimeError("No trainable parameters remain after applying --training_stage")


    optimizer = optimizer_cls(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # --- Dataset & Dataloader ---
    if args.dataset_type == "movi":
        train_dataset = MoviSimulationDataset(
            data_root=args.data_root,
            max_objects=args.max_objects,
            max_T_raw=args.max_T_raw if args.padded_batch else None,
            center_on_contact=args.center_on_contact,
        )
    else:
        train_dataset = SimulationDataset(
            ann_path=args.ann_path,
            data_root=args.data_root,
            load_video=False,
        )

    if args.overfit_single_sample_idx is not None:
        if not (0 <= args.overfit_single_sample_idx < len(train_dataset)):
            raise ValueError(
                f"--overfit_single_sample_idx={args.overfit_single_sample_idx} out of range "
                f"for dataset of length {len(train_dataset)}"
            )
        train_dataset = RepeatedSampleDataset(
            base_dataset=train_dataset,
            sample_idx=args.overfit_single_sample_idx,
            length=args.overfit_repeat_length,
        )
        logger.info(
            "Single-sample overfit mode enabled: sample_idx=%s repeated to logical length %s",
            args.overfit_single_sample_idx,
            args.overfit_repeat_length,
        )

    if args.padded_batch:
        collate_fn = partial(
            sim_collate_fn_padded,
            max_T_raw=args.max_T_raw,
            max_objects=args.max_objects,
            max_points_per_object=args.max_points_per_object,
        )
        logger.info(
            f"Padded batch mode enabled: max_T_raw={args.max_T_raw}, "
            f"max_objects={args.max_objects}, max_points_per_object={args.max_points_per_object}, "
            f"max_N={args.max_objects * args.max_points_per_object}, "
            f"center_on_contact={args.center_on_contact}"
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
    scheduler_last_epoch = -1
    if args.init_from_model_dir and args.initial_global_step:
        for param_group in optimizer.param_groups:
            param_group.setdefault("initial_lr", param_group["lr"])
        scheduler_last_epoch = args.initial_global_step - 1

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        last_epoch=scheduler_last_epoch,
    )

    # --- Noise scheduler ---
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=args.train_sampling_steps,
    )

    # --- Accelerate prepare ---
    # All trainable modules are wrapped so DDP syncs their gradients across GPUs.
    # AE is frozen and not wrapped (no gradient sync needed).
    if physics_decoder is not None:
        (
            sim_transformer,
            sim_cond_embedder,
            physics_decoder,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            sim_transformer,
            sim_cond_embedder,
            physics_decoder,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )
        accumulation_models = (
            sim_transformer,
            sim_cond_embedder,
            physics_decoder,
        )
    else:
        (
            sim_transformer,
            sim_cond_embedder,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            sim_transformer,
            sim_cond_embedder,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )
        accumulation_models = (sim_transformer, sim_cond_embedder)


    # --- Fixed visualization samples (loaded once, main process only) ---
    vis_samples = []
    if args.vis_steps > 0 and args.report_to == "wandb" and accelerator.is_main_process:
        num_vis = min(args.num_vis_samples, len(train_dataset))
        for idx in range(num_vis):
            vis_samples.append(train_dataset[idx])
        logger.info(f"Loaded {len(vis_samples)} fixed vis samples (indices 0..{num_vis - 1}).")

    # --- Resume from checkpoint ---
    global_step = args.initial_global_step
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        global_step = int(os.path.basename(args.resume_from_checkpoint).split("-")[1])
    first_epoch = global_step // num_update_steps_per_epoch

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
            with accelerator.accumulate(*accumulation_models):
                # --- Unpack batch ---
                # Tensors have shape (B, ...) after collation
                x_s_raw = batch['x_s_raw'].to(accelerator.device, dtype=weight_dtype)   # (B, T_raw, N, 6)
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

                # --- Build latent or raw Stage-1 state representation ---
                B_sz = x_s_raw.shape[0]
                T_raw_dim = x_s_raw.shape[1]
                N_sz = x_s_raw.shape[2]
                bsz = B_sz
                if args.use_raw_transformer:
                    # Preserve all raw frames for both state and time-varying
                    # force/contact conditions. No AE encode/decode is used.
                    x_s_enc = x_s_raw
                    c_force_enc = c_force_raw
                else:
                    with torch.no_grad():
                        pos_enc = ae.encode(x_s_raw[..., :3])
                        vel_enc = ae.encode(x_s_raw[..., 3:6])
                        force_enc = ae.encode(c_force_raw[..., :3])
                        contact_enc = ae.encode(c_force_raw[..., 3:6])
                    x_s_enc = torch.cat([pos_enc, vel_enc], dim=-1)
                    c_force_enc = torch.cat([force_enc, contact_enc], dim=-1)
                T = x_s_enc.shape[1]

                # --- Flow-matching noise in the selected state representation ---
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

                # Get sigmas for flow matching
                sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=weight_dtype)
                schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
                step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]  # timestep到index的反向映射
                sigma = sigmas[step_indices].flatten()
                while len(sigma.shape) < x_s_enc.ndim:
                    sigma = sigma.unsqueeze(-1)

                # Noisy state: zt = (1 - sigma) * z + sigma * noise.
                # Keep frame 0 fixed so Stage 1 conditions on the true initial state
                # instead of trying to regenerate it.
                noisy_x_s_enc = (1.0 - sigma) * x_s_enc + sigma * noise # (B, T, N, d_state)
                noisy_x_s_enc[:, :1] = x_s_enc[:, :1]

                # Flow-matching target: velocity = noise - z
                target = noise - x_s_enc    # sigma(t) = t          (B, T, N, d_state)

                # --- Initial frame conditioning ---
                # Repeat frame-0 state on every timestep token. Zero-padding the
                # future steps removed the strongest per-point identity signal and
                # forced the model to rely on a weak 3D anchor plus object-level
                # conditions, which is a direct source of identity ambiguity.
                init_enc_1 = x_s_enc[:, :1]  # (B, 1, N, d_state)
                init_enc_padded = init_enc_1.expand(-1, T, -1, -1).contiguous()  # (B, T, N, d_state)
                init_mask = torch.ones(B_sz, T, N_sz, 1,
                                       device=accelerator.device, dtype=weight_dtype)
                init_mask[:, 0, :, :] = 0.0  # first state frame is given
                point_anchor_1 = _compute_point_anchor(
                    x_s_raw[:, :1], point_obj_idx, args.anchor_mode
                ).to(device=accelerator.device, dtype=weight_dtype)
                if args.disable_point_anchor:
                    point_anchor_1 = point_anchor_1[..., :0]
                point_anchor = point_anchor_1.expand(-1, T, -1, -1).contiguous()  # (B, T, N, d_anchor)

                # --- Build valid sequence mask for DiT attention (padded batch mode) ---
                if args.padded_batch:
                    t_state = (
                        T_raw_tensor
                        if args.use_raw_transformer
                        else (T_raw_tensor - 1) // 4 + 1
                    )
                    t_idx = torch.arange(T, device=accelerator.device).unsqueeze(0)  # (1, T)
                    t_valid = t_idx < t_state.unsqueeze(1)    # (B, T)
                    state_seq_mask = t_valid.unsqueeze(2) & point_mask.unsqueeze(1)
                    valid_seq_mask = state_seq_mask.view(B_sz, T * N_sz)
                else:
                    valid_seq_mask = None

                # --- Encode conditions ---
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
                )  # (B, T, N, d_cond)

                # --- Forward pass in latent or raw state space ---
                with torch.amp.autocast("cuda", dtype=weight_dtype):
                    pred_enc = sim_transformer(
                        noisy_x_s_enc, init_enc_padded, init_mask, point_anchor, c_sim, timesteps,
                        dtype=weight_dtype,
                        valid_seq_mask=valid_seq_mask,
                        point_obj_idx=(
                            point_obj_idx
                            if args.use_object_local_attention
                            else None
                        ),
                    )  # (B, T, N, d_state) — predicted flow-matching velocity

                # --- Default flow-matching loss in the selected state space ---
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, sigmas=sigma,
                )
                loss_per_elem = F.mse_loss(pred_enc.float(), target.float(), reduction='none')

                if args.padded_batch and point_mask is not None:
                    # Build state-space mask: (B, T, N). Padded positions are excluded
                    # from both the numerator and denominator (zero contribution, not counted).
                    t_state = (
                        T_raw_tensor
                        if args.use_raw_transformer
                        else (T_raw_tensor - 1) // 4 + 1
                    )
                    t_idx = torch.arange(T, device=accelerator.device).unsqueeze(0)  # (1, T)
                    t_valid = t_idx < t_state.unsqueeze(1)                 # (B, T)
                    state_mask = t_valid.unsqueeze(2) & point_mask.unsqueeze(1)  # (B, T, N)
                    state_mask[:, :1] = False
                    loss_per_elem = loss_per_elem * state_mask.unsqueeze(-1).float()
                    n_valid = state_mask.float().sum() * args.d_state
                    latent_diffusion_loss = (loss_per_elem * weighting.float()).sum() / n_valid.clamp(min=1)
                else:
                    state_mask = torch.ones(B_sz, T, N_sz, device=accelerator.device, dtype=torch.bool)
                    state_mask[:, :1] = False
                    loss_per_elem = loss_per_elem * state_mask.unsqueeze(-1).float()
                    n_valid = state_mask.float().sum() * args.d_state
                    latent_diffusion_loss = (loss_per_elem * weighting.float()).sum() / n_valid.clamp(min=1)

                # Selected below when a raw-space objective is enabled.
                diffusion_loss = latent_diffusion_loss
                raw_point_mse_target_loss = x_s_raw.new_tensor(0.0)

                loss_mse_raw = x_s_raw.new_tensor(0.0)
                loss_mse_total = x_s_raw.new_tensor(0.0)
                decoder_refinement_loss = x_s_raw.new_tensor(0.0)
                loss_chamfer_coarse = x_s_raw.new_tensor(0.0)
                loss_residual_reg = x_s_raw.new_tensor(0.0)
                loss_residual_mag = x_s_raw.new_tensor(0.0)
                deformability_mean = x_s_raw.new_tensor(0.0)
                svd_condition = x_s_raw.new_tensor(0.0)
                fallback_rate = x_s_raw.new_tensor(0.0)
                centroid_error = x_s_raw.new_tensor(0.0)
                relative_rotation_angle = x_s_raw.new_tensor(0.0)

                local_dist_loss = x_s_raw.new_tensor(0.0)
                covariance_loss = x_s_raw.new_tensor(0.0)
                velocity_loss = x_s_raw.new_tensor(0.0)
                angular_velocity_loss = x_s_raw.new_tensor(0.0)
                velocity_vector_loss = x_s_raw.new_tensor(0.0)
                velocity_acceleration_loss = x_s_raw.new_tensor(0.0)
                centroid_loss = x_s_raw.new_tensor(0.0)
                rotation_loss = x_s_raw.new_tensor(0.0)
                rotation_temporal_loss = x_s_raw.new_tensor(0.0)
                rotation_axis_loss = x_s_raw.new_tensor(0.0)
                contact_com_velocity_loss = x_s_raw.new_tensor(0.0)
                contact_pose_ang_speed_loss = x_s_raw.new_tensor(0.0)
                deformation_gradient_loss = x_s_raw.new_tensor(0.0)
                local_volume_loss = x_s_raw.new_tensor(0.0)
                local_covariance_loss = x_s_raw.new_tensor(0.0)
                chamfer_loss = x_s_raw.new_tensor(0.0)
                momentum_loss = x_s_raw.new_tensor(0.0)
                floor_loss = x_s_raw.new_tensor(0.0)
                should_log_sim_metrics = (
                    args.sim_metrics_steps > 0
                    and accelerator.sync_gradients
                    and (global_step + 1) % args.sim_metrics_steps == 0
                )
                use_raw_auxiliary_loss = any([
                    args.use_raw_point_mse_target,
                    args.use_physics_conditioned_decoder,
                    args.lambda_local_dist > 0.0,
                    args.lambda_covariance > 0.0,
                    args.lambda_vel > 0.0,
                    args.lambda_ang_vel > 0.0,
                    args.lambda_velocity_vector > 0.0,
                    args.lambda_velocity_accel > 0.0,
                    args.lambda_centroid > 0.0,
                    args.lambda_rotation > 0.0,
                    args.lambda_rotation_temporal > 0.0,
                    args.lambda_rotation_axis > 0.0,
                    args.lambda_contact_com_velocity > 0.0,
                    args.lambda_contact_pose_ang_speed > 0.0,
                    args.lambda_deformation_gradient > 0.0,
                    args.lambda_local_volume > 0.0,
                    args.lambda_local_covariance > 0.0,
                    args.lambda_chamfer > 0.0,
                    args.lambda_momentum > 0.0,
                    args.lambda_floor > 0.0,
                ])
                sim_metric_values = None
                if use_raw_auxiliary_loss or should_log_sim_metrics:
                    pred_x0_enc = noise - pred_enc
                    pred_x0_enc[:, :1] = x_s_enc[:, :1]
                    if args.debug_raw_loss_gradients and args.lambda_local_dist > 0.0:
                        assert pred_x0_enc.requires_grad and pred_x0_enc.grad_fn is not None, (
                            "pred_x0_enc is detached before the raw-space auxiliary losses"
                        )
                    if args.padded_batch:
                        raw_t_idx = torch.arange(T_raw_dim, device=accelerator.device).unsqueeze(0)
                        valid_raw_frame_mask = raw_t_idx < T_raw_tensor.unsqueeze(1)
                    else:
                        valid_raw_frame_mask = None

                    # The raw point-MSE target preserves the simulator's exact point
                    # correspondence. Both sides are decoded x0 representations:
                    # pred_x0_enc is obtained from the predicted velocity, while
                    # x_s_enc == noise - target.
                    # Decode the coordinate prediction once. The physics decoder,
                    # when enabled, replaces only the raw point-MSE objective.
                    pred_x0_pos_raw = None
                    target_points = None
                    decoder_out = None

                    if point_mask is None:
                        raw_point_mask = torch.ones(
                            B_sz,
                            T_raw_dim,
                            N_sz,
                            device=accelerator.device,
                            dtype=torch.bool,
                        )
                    else:
                        raw_point_mask = point_mask.unsqueeze(1).expand(
                            -1, T_raw_dim, -1
                        ).clone()
                    if valid_raw_frame_mask is not None:
                        raw_point_mask = raw_point_mask & valid_raw_frame_mask.unsqueeze(-1)

                    # Frame 0 is hard-conditioned and therefore excluded from
                    # trainable point losses, matching the existing raw-MSE behavior.
                    raw_point_mask[:, :1] = False

                    if args.use_physics_conditioned_decoder:
                        pred_x0_pos_raw = (
                            pred_x0_enc[..., :3]
                            if args.use_raw_transformer
                            else ae.decode(pred_x0_enc[..., :d_latent], T_raw_dim)
                        )
                        target_points = x_s_raw[..., :3].float()
                        canonical_points = x_s_raw[:, 0, :, :3].float()

                        physics_attrs = torch.ones(
                            B_sz,
                            args.max_objects,
                            4,
                            device=accelerator.device,
                            dtype=torch.float32,
                        )
                        physics_attrs[:, :, 0] = 1.0

                        decoder_out = physics_decoder(
                            latent=pred_x0_pos_raw.float(),
                            canonical_points=canonical_points,
                            physics_attrs=physics_attrs,
                            point_obj_idx=point_obj_idx,
                            point_mask=point_mask,
                        )

                        kabsch_diagnostics = {
                            name: value.detach()
                            for name, value in decoder_out["diagnostics"].items()
                        }
                        valid_object_frames = decoder_out["object_mask"][:, None, :].expand_as(
                            kabsch_diagnostics["fallback_count"]
                        )
                        if valid_raw_frame_mask is not None:
                            valid_object_frames = (
                                valid_object_frames
                                & valid_raw_frame_mask[:, :, None]
                            )
                        valid_object_frames_float = valid_object_frames.float()
                        valid_object_frame_count = valid_object_frames_float.sum().clamp_min(1)

                        fallback_rate = (
                            kabsch_diagnostics["fallback_count"]
                            * valid_object_frames_float
                        ).sum() / valid_object_frame_count
                        centroid_error = (
                            kabsch_diagnostics["centroid_error"]
                            * valid_object_frames_float
                        ).sum() / valid_object_frame_count

                        finite_svd = (
                            valid_object_frames
                            & torch.isfinite(kabsch_diagnostics["svd_condition_number"])
                        )
                        svd_condition = (
                            kabsch_diagnostics["svd_condition_number"]
                            * finite_svd.float()
                        ).sum() / finite_svd.float().sum().clamp_min(1)

                        valid_rotation_frames = valid_object_frames.clone()
                        valid_rotation_frames[:, :1] = False
                        relative_rotation_angle = (
                            kabsch_diagnostics["relative_rotation_angle"]
                            * valid_rotation_frames.float()
                        ).sum() / valid_rotation_frames.float().sum().clamp_min(1)

                        pred_points = torch.cat(
                            [
                                pred_x0_pos_raw[:, :1].float(),
                                decoder_out["positions"][:, 1:],
                            ],
                            dim=1,
                        )

                        # Direct pre-decoder supervision: every raw predicted
                        # coordinate receives a gradient from SimTransformer.
                        point_mse_raw = (
                            pred_x0_pos_raw.float() - target_points
                        ).square().sum(dim=-1)
                        loss_mse_raw = (
                            (point_mse_raw * raw_point_mask.float()).sum()
                            / raw_point_mask.float().sum().clamp_min(1)
                        )

                        # Kabsch-projected final-coordinate supervision.
                        point_mse_total = (
                            pred_points - target_points
                        ).square().sum(dim=-1)
                        loss_mse_total = (
                            (point_mse_total * raw_point_mask.float()).sum()
                            / raw_point_mask.float().sum().clamp_min(1)
                        )

                        loss_chamfer_coarse = _compute_chamfer_loss(
                            pos_pred=decoder_out["coarse_positions"],
                            pos_gt=target_points,
                            point_mask=point_mask,
                            valid_frame_mask=valid_raw_frame_mask,
                        )

                        stiffness = physics_attrs[:, :, 0]
                        expected_deformability = (
                            (1.0 - stiffness)
                            .unsqueeze(1)
                            .unsqueeze(-1)
                            .expand_as(decoder_out["deformability"])
                        )
                        loss_residual_reg = F.mse_loss(
                            decoder_out["deformability"],
                            expected_deformability,
                        )

                        residual_norm = decoder_out["residual"].norm(dim=-1)
                        residual_norm_per_sample = (
                            (residual_norm * raw_point_mask.float()).sum(dim=(1, 2))
                            / raw_point_mask.float().sum(dim=(1, 2)).clamp_min(1)
                        )
                        softness = (1.0 - stiffness).mean(dim=1)
                        loss_residual_mag = (
                            residual_norm_per_sample * softness
                        ).mean()

                        loss_point = (
                            loss_mse_raw
                            + loss_mse_total
                            + args.lambda_chamfer_coarse * loss_chamfer_coarse
                            + args.lambda_residual_reg * loss_residual_reg
                            + args.lambda_residual_mag * loss_residual_mag
                        )
                        # Separate direct coordinate supervision from Kabsch/decoder
                        # refinement so Direction A can warm up every physics term.
                        diffusion_loss = loss_mse_raw
                        decoder_refinement_loss = loss_point - loss_mse_raw
                        deformability_mean = decoder_out["deformability"].mean()

                    elif args.use_raw_point_mse_target:
                        pred_x0_pos_raw = (
                            pred_x0_enc[..., :3]
                            if args.use_raw_transformer
                            else ae.decode(pred_x0_enc[..., :d_latent], T_raw_dim)
                        )
                        target_x0_pos_raw = (
                            x_s_raw[..., :3]
                            if args.use_raw_transformer
                            else ae.decode(x_s_enc[..., :d_latent], T_raw_dim)
                        )
                        point_mse = (
                            pred_x0_pos_raw.float() - target_x0_pos_raw.float()
                        ).square().sum(dim=-1)
                        raw_point_mse_per_sample = (
                            (point_mse * raw_point_mask.float()).sum(dim=(1, 2))
                            / raw_point_mask.float().sum(dim=(1, 2)).clamp_min(1)
                        )
                        sample_weighting = weighting.float().reshape(B_sz, -1)[:, 0]
                        raw_point_mse_target_loss = (
                            raw_point_mse_per_sample * sample_weighting
                        ).mean()
                        diffusion_loss = raw_point_mse_target_loss

                    # Reuse the position decode above when available. Existing
                    # velocity, local-distance, momentum, floor, and diffusion
                    # losses remain unchanged below.
                    if pred_x0_pos_raw is None:
                        pred_x0_pos_raw = (
                            pred_x0_enc[..., :3]
                            if args.use_raw_transformer
                            else ae.decode(pred_x0_enc[..., :d_latent], T_raw_dim)
                        )
                    pos_pred = (
                        pred_points
                        if args.use_physics_conditioned_decoder
                        else pred_x0_pos_raw.float()
                    )
                    pos_gt = (
                        target_points
                        if target_points is not None
                        else x_s_raw[..., :3].float()
                    )
                    velocity_pred = None
                    velocity_gt = None
                    if any([
                        args.lambda_momentum > 0.0,
                        args.lambda_ang_vel > 0.0,
                        args.lambda_velocity_vector > 0.0,
                        args.lambda_velocity_accel > 0.0,
                    ]):
                        velocity_pred = (
                            pred_x0_enc[..., 3:6]
                            if args.use_raw_transformer
                            else ae.decode(pred_x0_enc[..., d_latent:], T_raw_dim)
                        ).float()
                        velocity_gt = x_s_raw[..., 3:6].float()

                    if args.lambda_vel > 0.0:
                        velocity_loss = _compute_velocity_consistency_loss(
                            pos_pred=pos_pred,
                            pos_gt=pos_gt,
                            point_mask=point_mask,
                            valid_frame_mask=valid_raw_frame_mask,
                        )
                    if args.lambda_ang_vel > 0.0:
                        angular_velocity_loss = _compute_angular_velocity_loss(
                            pos_pred=pred_x0_pos_raw.float(),
                            velocity_pred=velocity_pred,
                            pos_gt=x_s_raw[..., :3].float(),
                            velocity_gt=velocity_gt,
                            point_obj_idx=point_obj_idx,
                            point_mask=point_mask,
                            valid_frame_mask=valid_raw_frame_mask,
                        )
                    if args.lambda_velocity_vector > 0.0:
                        # Direction C: optional object-frame alignment ablation.
                        velocity_for_vector_loss = velocity_pred
                        if args.velocity_coord_align:
                            velocity_for_vector_loss = _align_velocity_to_gt_frame(
                                velocity_pred=velocity_pred,
                                pos_pred=pred_x0_pos_raw.float(),
                                pos_gt=x_s_raw[..., :3].float(),
                                canonical_points=x_s_raw[:, 0, :, :3].float(),
                                point_obj_idx=point_obj_idx,
                                point_mask=point_mask,
                                num_objects=args.max_objects,
                            )
                        velocity_vector_loss = _compute_velocity_vector_loss(
                            velocity_pred=velocity_for_vector_loss,
                            velocity_gt=velocity_gt,
                            point_mask=point_mask,
                            valid_frame_mask=valid_raw_frame_mask,
                        )
                    if args.lambda_velocity_accel > 0.0:
                        velocity_acceleration_loss = _compute_velocity_acceleration_loss(
                            velocity_pred=velocity_pred,
                            velocity_gt=velocity_gt,
                            point_mask=point_mask,
                            valid_frame_mask=valid_raw_frame_mask,
                        )
                    if args.lambda_centroid > 0.0:
                        centroid_loss = _compute_centroid_consistency_loss(
                            pos_pred=pred_x0_pos_raw.float(),
                            pos_gt=x_s_raw[..., :3].float(),
                            point_obj_idx=point_obj_idx,
                            point_mask=point_mask,
                            valid_frame_mask=valid_raw_frame_mask,
                        )
                    if args.lambda_rotation > 0.0 or args.lambda_rotation_temporal > 0.0:
                        rotation_loss, rotation_temporal_loss = _compute_rotation_consistency_losses(
                            pos_pred=pred_x0_pos_raw.float(),
                            pos_gt=x_s_raw[..., :3].float(),
                            canonical_points=x_s_raw[:, 0, :, :3].float(),
                            point_obj_idx=point_obj_idx,
                            point_mask=point_mask,
                            valid_frame_mask=valid_raw_frame_mask,
                            num_objects=args.max_objects,
                        )
                    if args.lambda_rotation_axis > 0.0:
                        rotation_axis_loss = _compute_rotation_axis_loss(
                            pos_pred=pred_x0_pos_raw.float(),
                            canonical_points=x_s_raw[:, 0, :, :3].float(),
                            point_obj_idx=point_obj_idx,
                            point_mask=point_mask,
                            valid_frame_mask=valid_raw_frame_mask,
                            num_objects=args.max_objects,
                        )
                    if args.lambda_contact_com_velocity > 0.0:
                        contact_com_velocity_loss = _compute_contact_com_velocity_loss(
                            pos_pred=pred_x0_pos_raw.float(),
                            pos_gt=x_s_raw[..., :3].float(),
                            c_force_raw=c_force_raw.float(),
                            point_obj_idx=point_obj_idx,
                            point_mask=point_mask,
                            valid_frame_mask=valid_raw_frame_mask,
                            num_objects=args.max_objects,
                            post_frames=args.contact_loss_post_frames,
                        )
                    if args.lambda_contact_pose_ang_speed > 0.0:
                        contact_pose_ang_speed_loss = _compute_contact_pose_ang_speed_loss(
                            pos_pred=pred_x0_pos_raw.float(),
                            pos_gt=x_s_raw[..., :3].float(),
                            c_force_raw=c_force_raw.float(),
                            point_obj_idx=point_obj_idx,
                            point_mask=point_mask,
                            valid_frame_mask=valid_raw_frame_mask,
                            num_objects=args.max_objects,
                            post_frames=args.contact_loss_post_frames,
                        )
                    if any([
                        args.lambda_deformation_gradient > 0.0,
                        args.lambda_local_volume > 0.0,
                        args.lambda_local_covariance > 0.0,
                    ]):
                        (
                            deformation_gradient_loss,
                            local_volume_loss,
                            local_covariance_loss,
                        ) = _compute_local_geometry_losses(
                            pos_pred=pred_x0_pos_raw.float(),
                            pos_gt=x_s_raw[..., :3].float(),
                            point_obj_idx=point_obj_idx,
                            point_mask=point_mask,
                            valid_frame_mask=valid_raw_frame_mask,
                            k=args.knn_k,
                        )
                    if args.lambda_local_dist > 0.0:
                        local_dist_loss = _compute_local_distance_loss(
                            pos_pred=pos_pred,
                            pos_gt=pos_gt,
                            point_obj_idx=point_obj_idx,
                            point_mask=point_mask,
                            valid_frame_mask=valid_raw_frame_mask,
                            k=args.knn_k,
                        )
                        if args.debug_raw_loss_gradients:
                            assert local_dist_loss.requires_grad and local_dist_loss.grad_fn is not None, (
                                "local_dist_loss is detached from the decoded prediction"
                            )
                    if args.lambda_covariance > 0.0:
                        covariance_loss = _compute_covariance_loss(
                            pos_pred=pos_pred,
                            pos_gt=pos_gt,
                            point_obj_idx=point_obj_idx,
                            point_mask=point_mask,
                            valid_frame_mask=valid_raw_frame_mask,
                        )
                    if args.lambda_chamfer > 0.0:
                        chamfer_loss = _compute_chamfer_loss(
                            pos_pred=pos_pred,
                            pos_gt=pos_gt,
                            point_mask=point_mask,
                            valid_frame_mask=valid_raw_frame_mask,
                        )
                    if args.lambda_momentum > 0.0:
                        momentum_loss = _compute_momentum_loss(
                            velocity_pred=velocity_pred,
                            velocity_gt=velocity_gt,
                            c_mass=c_mass,
                            point_obj_idx=point_obj_idx,
                            point_mask=point_mask,
                            valid_frame_mask=valid_raw_frame_mask,
                        )
                    if args.lambda_floor > 0.0:
                        floor_loss = _compute_floor_penetration_loss(
                            pos_pred=pos_pred,
                            c_floor=c_floor,
                            c_static=c_static,
                            point_obj_idx=point_obj_idx,
                            gravity_axis=args.gravity_axis,
                            point_mask=point_mask,
                            valid_frame_mask=valid_raw_frame_mask,
                        )

                    if should_log_sim_metrics:
                        with torch.no_grad():
                            sim_metric_values = {
                                "diag/knn_edge_error": knn_edge_error(
                                    pos_pred, pos_gt, point_obj_idx,
                                    k=args.knn_k,
                                    point_mask=point_mask,
                                    valid_frame_mask=valid_raw_frame_mask,
                                ),
                                "diag/velocity_drift": velocity_drift(
                                    pos_pred, pos_gt,
                                    point_mask=point_mask,
                                    valid_frame_mask=valid_raw_frame_mask,
                                ),
                                "diag/ae_reconstruction_chamfer": ae_reconstruction_chamfer(
                                    ae, x_s_raw,
                                    point_mask=point_mask,
                                    valid_frame_mask=valid_raw_frame_mask,
                                ),
                            }
                            sim_metric_values.update(frame0_error(
                                pos_pred,
                                x_s_raw[:, :1, ..., :3].float(),
                                point_mask=point_mask,
                            ))

                # Direction D: select a stable, independently testable physics subset.
                if args.physics_mode == "full":
                    effective_weights = {
                        "decoder_refinement": 1.0,
                        "vel": args.lambda_vel,
                        "ang_vel": args.lambda_ang_vel,
                        "velocity_vector": args.lambda_velocity_vector,
                        "velocity_accel": args.lambda_velocity_accel,
                        "centroid": args.lambda_centroid,
                        "rotation": args.lambda_rotation,
                        "rotation_temporal": args.lambda_rotation_temporal,
                        "rotation_axis": args.lambda_rotation_axis,
                        "contact_com_velocity": args.lambda_contact_com_velocity,
                        "contact_pose_ang_speed": args.lambda_contact_pose_ang_speed,
                        "deformation_gradient": args.lambda_deformation_gradient,
                        "local_volume": args.lambda_local_volume,
                        "local_covariance": args.lambda_local_covariance,
                        "local_dist": args.lambda_local_dist,
                        "covariance": args.lambda_covariance,
                        "chamfer": args.lambda_chamfer,
                        "momentum": args.lambda_momentum,
                        "floor": args.lambda_floor,
                    }
                elif args.physics_mode == "minimal":
                    effective_weights = {
                        "decoder_refinement": 0.0,
                        "vel": 0.0,
                        "ang_vel": 0.0,
                        "velocity_vector": 0.0,
                        "velocity_accel": 0.0,
                        "centroid": args.lambda_centroid,
                        "rotation": args.lambda_rotation,
                        "rotation_temporal": args.lambda_rotation_temporal,
                        "rotation_axis": args.lambda_rotation_axis,
                        # Contact coupling is intentionally available in minimal
                        # mode: it targets the diagnosed post-contact failure.
                        "contact_com_velocity": args.lambda_contact_com_velocity,
                        "contact_pose_ang_speed": args.lambda_contact_pose_ang_speed,
                        "deformation_gradient": 0.0,
                        "local_volume": 0.0,
                        "local_covariance": 0.0,
                        "local_dist": args.lambda_local_dist,
                        "covariance": 0.0,
                        "chamfer": 0.0,
                        "momentum": (
                            args.lambda_momentum if args.minimal_include_momentum else 0.0
                        ),
                        "floor": 0.0,
                    }
                else:  # shape_only
                    effective_weights = {name: 0.0 for name in (
                        "decoder_refinement",
                        "vel", "ang_vel", "velocity_vector", "velocity_accel",
                        "centroid", "rotation", "rotation_temporal", "rotation_axis",
                        "contact_com_velocity", "contact_pose_ang_speed",
                        "deformation_gradient", "local_volume", "local_covariance",
                        "local_dist", "covariance", "chamfer", "momentum", "floor",
                    )}

                # Direction A: warm up every non-diffusion physics term together.
                if args.physics_loss_warmup_steps > 0:
                    physics_loss_scale = min(
                        1.0,
                        global_step / args.physics_loss_warmup_steps,
                    )
                else:
                    physics_loss_scale = 1.0
                if args.training_stage == "shape_only":
                    physics_loss_scale = 0.0

                physics_loss = (
                    args.lambda_diffusion * effective_weights["decoder_refinement"] * decoder_refinement_loss
                    + effective_weights["vel"] * velocity_loss
                    + effective_weights["ang_vel"] * angular_velocity_loss
                    + effective_weights["velocity_vector"] * velocity_vector_loss
                    + effective_weights["velocity_accel"] * velocity_acceleration_loss
                    + effective_weights["centroid"] * centroid_loss
                    + effective_weights["rotation"] * rotation_loss
                    + effective_weights["rotation_temporal"] * rotation_temporal_loss
                    + effective_weights["rotation_axis"] * rotation_axis_loss
                    + effective_weights["contact_com_velocity"] * contact_com_velocity_loss
                    + effective_weights["contact_pose_ang_speed"] * contact_pose_ang_speed_loss
                    + effective_weights["deformation_gradient"] * deformation_gradient_loss
                    + effective_weights["local_volume"] * local_volume_loss
                    + effective_weights["local_covariance"] * local_covariance_loss
                    + effective_weights["local_dist"] * local_dist_loss
                    + effective_weights["covariance"] * covariance_loss
                    + effective_weights["chamfer"] * chamfer_loss
                    + effective_weights["momentum"] * momentum_loss
                    + effective_weights["floor"] * floor_loss
                )
                loss = args.lambda_diffusion * diffusion_loss + physics_loss_scale * physics_loss

                # --- Backward ---
                avg_loss = accelerator.gather(loss.repeat(bsz)).mean()
                avg_physics_loss = accelerator.gather(physics_loss.repeat(bsz)).mean()
                avg_diffusion_loss = accelerator.gather(diffusion_loss.repeat(bsz)).mean()
                avg_latent_diffusion_loss = accelerator.gather(
                    latent_diffusion_loss.repeat(bsz)
                ).mean()
                avg_raw_point_mse_target_loss = accelerator.gather(
                    raw_point_mse_target_loss.repeat(bsz)
                ).mean()
                avg_loss_mse_raw = accelerator.gather(
                    loss_mse_raw.repeat(bsz)
                ).mean()
                avg_loss_mse_total = accelerator.gather(
                    loss_mse_total.repeat(bsz)
                ).mean()
                avg_decoder_refinement_loss = accelerator.gather(
                    decoder_refinement_loss.repeat(bsz)
                ).mean()
                avg_loss_chamfer_coarse = accelerator.gather(
                    loss_chamfer_coarse.repeat(bsz)
                ).mean()
                avg_loss_residual_reg = accelerator.gather(
                    loss_residual_reg.repeat(bsz)
                ).mean()
                avg_loss_residual_mag = accelerator.gather(
                    loss_residual_mag.repeat(bsz)
                ).mean()
                avg_deformability_mean = accelerator.gather(
                    deformability_mean.detach().repeat(bsz)
                ).mean()
                avg_svd_condition = accelerator.gather(svd_condition.repeat(bsz)).mean()
                avg_fallback_rate = accelerator.gather(fallback_rate.repeat(bsz)).mean()
                avg_centroid_error = accelerator.gather(centroid_error.repeat(bsz)).mean()
                avg_relative_rotation_angle = accelerator.gather(
                    relative_rotation_angle.repeat(bsz)
                ).mean()
                avg_local_dist_loss = accelerator.gather(local_dist_loss.repeat(bsz)).mean()
                avg_covariance_loss = accelerator.gather(covariance_loss.repeat(bsz)).mean()
                avg_velocity_loss = accelerator.gather(velocity_loss.repeat(bsz)).mean()
                avg_angular_velocity_loss = accelerator.gather(
                    angular_velocity_loss.repeat(bsz)
                ).mean()
                avg_velocity_vector_loss = accelerator.gather(
                    velocity_vector_loss.repeat(bsz)
                ).mean()
                avg_velocity_acceleration_loss = accelerator.gather(
                    velocity_acceleration_loss.repeat(bsz)
                ).mean()
                avg_centroid_loss = accelerator.gather(centroid_loss.repeat(bsz)).mean()
                avg_rotation_loss = accelerator.gather(rotation_loss.repeat(bsz)).mean()
                avg_rotation_temporal_loss = accelerator.gather(
                    rotation_temporal_loss.repeat(bsz)
                ).mean()
                avg_rotation_axis_loss = accelerator.gather(
                    rotation_axis_loss.repeat(bsz)
                ).mean()
                avg_contact_com_velocity_loss = accelerator.gather(
                    contact_com_velocity_loss.repeat(bsz)
                ).mean()
                avg_contact_pose_ang_speed_loss = accelerator.gather(
                    contact_pose_ang_speed_loss.repeat(bsz)
                ).mean()
                avg_deformation_gradient_loss = accelerator.gather(
                    deformation_gradient_loss.repeat(bsz)
                ).mean()
                avg_local_volume_loss = accelerator.gather(
                    local_volume_loss.repeat(bsz)
                ).mean()
                avg_local_covariance_loss = accelerator.gather(
                    local_covariance_loss.repeat(bsz)
                ).mean()
                avg_chamfer_loss = accelerator.gather(chamfer_loss.repeat(bsz)).mean()
                avg_momentum_loss = accelerator.gather(momentum_loss.repeat(bsz)).mean()
                avg_floor_loss = accelerator.gather(floor_loss.repeat(bsz)).mean()
                train_loss += avg_loss.item()
                accum_count += 1

                accelerator.backward(loss)
                if args.debug_raw_loss_gradients and args.lambda_local_dist > 0.0:
                    local_grad_reaches_transformer = any(
                        parameter.grad is not None and torch.any(parameter.grad != 0).item()
                        for parameter in sim_transformer.parameters()
                        if parameter.requires_grad
                    )
                    assert local_grad_reaches_transformer, (
                        "No nonzero SimTransformer gradient after backward; raw local loss is disconnected"
                    )
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
                log_values = {
                        "train_loss": train_loss / max(accum_count, 1),
                        "diffusion_loss": avg_diffusion_loss.item(),
                        "latent_diffusion_loss": avg_latent_diffusion_loss.item(),
                        "raw_point_mse_target_loss": avg_raw_point_mse_target_loss.item(),
                        "weighted_diffusion_loss": args.lambda_diffusion * avg_diffusion_loss.item(),
                        "physics_loss": avg_physics_loss.item(),
                        "physics_loss_scale": physics_loss_scale,
                        "local_dist_loss": avg_local_dist_loss.item(),
                        "covariance_loss": avg_covariance_loss.item(),
                        "velocity_loss": avg_velocity_loss.item(),
                        "angular_velocity_loss": avg_angular_velocity_loss.item(),
                        "velocity_vector_loss": avg_velocity_vector_loss.item(),
                        "velocity_acceleration_loss": avg_velocity_acceleration_loss.item(),
                        "centroid_loss": avg_centroid_loss.item(),
                        "rotation_loss": avg_rotation_loss.item(),
                        "rotation_temporal_loss": avg_rotation_temporal_loss.item(),
                        "rotation_axis_loss": avg_rotation_axis_loss.item(),
                        "contact_com_velocity_loss": avg_contact_com_velocity_loss.item(),
                        "contact_pose_ang_speed_loss": avg_contact_pose_ang_speed_loss.item(),
                        "deformation_gradient_loss": avg_deformation_gradient_loss.item(),
                        "local_volume_loss": avg_local_volume_loss.item(),
                        "local_covariance_loss": avg_local_covariance_loss.item(),
                        "chamfer_loss": avg_chamfer_loss.item(),
                        "momentum_loss": avg_momentum_loss.item(),
                        "floor_loss": avg_floor_loss.item(),
                        "weighted_local_dist_loss": physics_loss_scale * effective_weights["local_dist"] * avg_local_dist_loss.item(),
                        "weighted_covariance_loss": physics_loss_scale * effective_weights["covariance"] * avg_covariance_loss.item(),
                        "weighted_velocity_loss": physics_loss_scale * effective_weights["vel"] * avg_velocity_loss.item(),
                        "weighted_angular_velocity_loss": (
                            physics_loss_scale * effective_weights["ang_vel"] * avg_angular_velocity_loss.item()
                        ),
                        "weighted_velocity_vector_loss": (
                            physics_loss_scale * effective_weights["velocity_vector"] * avg_velocity_vector_loss.item()
                        ),
                        "weighted_velocity_acceleration_loss": (
                            physics_loss_scale * effective_weights["velocity_accel"] * avg_velocity_acceleration_loss.item()
                        ),
                        "weighted_centroid_loss": physics_loss_scale * effective_weights["centroid"] * avg_centroid_loss.item(),
                        "weighted_rotation_loss": physics_loss_scale * effective_weights["rotation"] * avg_rotation_loss.item(),
                        "weighted_rotation_temporal_loss": (
                            physics_loss_scale * effective_weights["rotation_temporal"] * avg_rotation_temporal_loss.item()
                        ),
                        "weighted_rotation_axis_loss": (
                            physics_loss_scale * effective_weights["rotation_axis"] * avg_rotation_axis_loss.item()
                        ),
                        "weighted_contact_com_velocity_loss": (
                            physics_loss_scale * effective_weights["contact_com_velocity"]
                            * avg_contact_com_velocity_loss.item()
                        ),
                        "weighted_contact_pose_ang_speed_loss": (
                            physics_loss_scale * effective_weights["contact_pose_ang_speed"]
                            * avg_contact_pose_ang_speed_loss.item()
                        ),
                        "weighted_decoder_refinement_loss": (
                            physics_loss_scale * args.lambda_diffusion
                            * effective_weights["decoder_refinement"]
                            * avg_decoder_refinement_loss.item()
                        ),
                        "weighted_deformation_gradient_loss": (
                            physics_loss_scale * effective_weights["deformation_gradient"] * avg_deformation_gradient_loss.item()
                        ),
                        "weighted_local_volume_loss": (
                            physics_loss_scale * effective_weights["local_volume"] * avg_local_volume_loss.item()
                        ),
                        "weighted_local_covariance_loss": (
                            physics_loss_scale * effective_weights["local_covariance"] * avg_local_covariance_loss.item()
                        ),
                        "weighted_chamfer_loss": physics_loss_scale * effective_weights["chamfer"] * avg_chamfer_loss.item(),
                        "weighted_momentum_loss": physics_loss_scale * effective_weights["momentum"] * avg_momentum_loss.item(),
                        "weighted_floor_loss": physics_loss_scale * effective_weights["floor"] * avg_floor_loss.item(),
                        "lr": current_lr,
                        "epoch": epoch,
                        "global_step": global_step,
                }
                if args.use_physics_conditioned_decoder:
                    log_values.update({
                        "loss_mse_raw": avg_loss_mse_raw.item(),
                        "loss_mse_total": avg_loss_mse_total.item(),
                        "decoder_refinement_loss": avg_decoder_refinement_loss.item(),
                        "loss_chamfer_coarse": avg_loss_chamfer_coarse.item(),
                        "loss_residual_reg": avg_loss_residual_reg.item(),
                        "loss_residual_mag": avg_loss_residual_mag.item(),
                        "deformability_mean": avg_deformability_mean.item(),
                        "avg_svd_condition": avg_svd_condition.item(),
                        "avg_fallback_rate": avg_fallback_rate.item(),
                        "avg_centroid_error": avg_centroid_error.item(),
                        "avg_relative_rotation_angle": avg_relative_rotation_angle.item(),
                    })
                if sim_metric_values is not None:
                    log_values.update({
                        name: accelerator.gather(value.detach().repeat(bsz)).mean().item()
                        for name, value in sim_metric_values.items()
                    })
                accelerator.log(log_values, step=global_step)
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
                    unwrapped_physics = (
                        accelerator.unwrap_model(physics_decoder)
                        if physics_decoder is not None
                        else None
                    )
                    unwrapped_sim.eval()
                    unwrapped_cond.eval()
                    if unwrapped_physics is not None:
                        unwrapped_physics.eval()
                    try:
                        _run_vis(
                            vis_samples,
                            unwrapped_sim,
                            ae,
                            unwrapped_cond,
                            global_step,
                            accelerator.device,
                            weight_dtype,
                            args.vis_num_inference_steps,
                            args.vis_fps,
                            args.anchor_mode,
                            physics_decoder=unwrapped_physics,
                            use_raw_transformer=args.use_raw_transformer,
                        )
                    finally:
                        unwrapped_sim.train()
                        unwrapped_cond.train()
                        if unwrapped_physics is not None:
                            unwrapped_physics.train()

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
                        if physics_decoder is not None:
                            torch.save(
                                accelerator.unwrap_model(physics_decoder).state_dict(),
                                os.path.join(save_dir, "physics_decoder.pt"),
                            )
                    accelerator.wait_for_everyone()
                    # Includes optimizer, scheduler, RNG, and every prepared model,
                    # including physics_decoder when enabled.
                    accelerator.save_state(save_dir)
                    if accelerator.is_main_process:
                        logger.info(f"Saved checkpoint to {save_dir}")
                        gc.collect()
                        torch.cuda.empty_cache()

            logs = {
                "step_loss": loss.detach().item(),
                "diff_loss": diffusion_loss.detach().item(),
                "local_dist": local_dist_loss.detach().item(),
                "covariance": covariance_loss.detach().item(),
                "velocity": velocity_loss.detach().item(),
                "angular_velocity": angular_velocity_loss.detach().item(),
                "velocity_vector": velocity_vector_loss.detach().item(),
                "velocity_accel": velocity_acceleration_loss.detach().item(),
                "centroid": centroid_loss.detach().item(),
                "rotation": rotation_loss.detach().item(),
                "rotation_temporal": rotation_temporal_loss.detach().item(),
                "rotation_axis": rotation_axis_loss.detach().item(),
                "deformation_gradient": deformation_gradient_loss.detach().item(),
                "local_volume": local_volume_loss.detach().item(),
                "local_covariance": local_covariance_loss.detach().item(),
                "chamfer": chamfer_loss.detach().item(),
                "momentum": momentum_loss.detach().item(),
                "floor": floor_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
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
        if physics_decoder is not None:
            torch.save(
                accelerator.unwrap_model(physics_decoder).state_dict(),
                os.path.join(save_dir, "physics_decoder.pt"),
            )
        logger.info(f"Saved final model to {save_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()

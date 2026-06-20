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
    parser.add_argument("--lambda_local_dist", type=float, default=1e-3,
                        help="Weight for frame-0-relative local KNN edge deformation loss.")
    parser.add_argument("--lambda_covariance", type=float, default=0.0,
                        help="Weight for per-object raw-position covariance consistency loss.")
    parser.add_argument("--lambda_vel", type=float, default=0.1,
                        help="Weight for frame-to-frame raw-position velocity consistency loss.")
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
    if args.resume_from_checkpoint and args.init_from_model_dir:
        parser.error("Use only one of --resume_from_checkpoint and --init_from_model_dir")
    if args.initial_global_step and not args.init_from_model_dir:
        parser.error("--initial_global_step requires --init_from_model_dir")
    for name in (
        "lambda_diffusion", "lambda_vel", "lambda_local_dist", "lambda_covariance", "lambda_chamfer",
        "lambda_momentum", "lambda_floor",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name} must be >= 0")

    return args


def _run_vis(vis_samples, sim_transformer, ae, sim_cond_embedder,
             global_step, device, weight_dtype, num_inference_steps, fps, anchor_mode):
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


def _compute_momentum_loss(
    velocity_pred: torch.Tensor,
    velocity_gt: torch.Tensor,
    c_mass: torch.Tensor,
    point_obj_idx: torch.Tensor,
    point_mask: torch.Tensor | None,
    valid_frame_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Match total linear momentum per object after uniform per-point mass split."""
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
    """Symmetric squared Chamfer distance for each valid raw point-cloud frame."""
    B, T_raw, N, _ = pos_gt.shape
    total_loss = pos_gt.new_tensor(0.0)
    total_weight = pos_gt.new_tensor(0.0)

    for b in range(B):
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
        return pos_gt.new_tensor(0.0)
    return total_loss / total_weight


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
        d_anchor=0 if args.disable_point_anchor else 3,
        d_sim=args.d_sim,
        ffn_dim=args.sim_ffn_dim,
        num_heads=args.sim_num_heads,
        num_layers=args.sim_num_layers,
        use_factorized_attention=args.use_factorized_attention,
        use_temporal_correspondence=args.use_temporal_correspondence,
        use_temporal_rope=args.use_temporal_rope,
        use_object_local_attention=args.use_object_local_attention,
    )
    logger.info(
        "Temporal structure: sinusoidal token positions=on, correspondence=%s, RoPE=%s, "
        "factorized_attention=%s, object_local_attention=%s, "
        "lambda_diffusion=%g, lambda_local_dist=%g, lambda_covariance=%g, lambda_vel=%g, lambda_chamfer=%g, "
        "lambda_momentum=%g, lambda_floor=%g(axis=%s)",
        args.use_temporal_correspondence,
        args.use_temporal_rope,
        args.use_factorized_attention,
        args.use_object_local_attention,
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
        for checkpoint_path in (transformer_path, cond_embedder_path):
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f"Model-only initialization requires {checkpoint_path}"
                )
        sim_transformer.load_state_dict(torch.load(transformer_path, map_location="cpu"))
        sim_cond_embedder.load_state_dict(torch.load(cond_embedder_path, map_location="cpu"))
        logger.info(
            "Initialized Stage 1 model weights from %s at optimizer step %s; "
            "optimizer state will be reset.",
            args.init_from_model_dir,
            args.initial_global_step,
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
    )   # whether all the params are trainable

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

                # --- Encode x_s_raw and c_force_raw once via frozen AE (LDM-style) ---
                B_sz = x_s_raw.shape[0]
                T_raw_dim = x_s_raw.shape[1]
                N_sz = x_s_raw.shape[2]
                bsz = B_sz
                with torch.no_grad():
                    pos_enc = ae.encode(x_s_raw[..., :3])        # (B, T, N, d_latent)
                    vel_enc = ae.encode(x_s_raw[..., 3:6])       # (B, T, N, d_latent)
                    # c_force_raw: force(3) and contact(3) each → d_latent via frozen AE
                    force_enc   = ae.encode(c_force_raw[..., :3])
                    contact_enc = ae.encode(c_force_raw[..., 3:6])
                x_s_enc = torch.cat([pos_enc, vel_enc], dim=-1)  # (B, T, N, d_state)
                c_force_enc = torch.cat([force_enc, contact_enc], dim=-1)  # (B, T, N, d_force)
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

                # Get sigmas for flow matching
                sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=weight_dtype)
                schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
                step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]  # timestep到index的反向映射
                sigma = sigmas[step_indices].flatten()
                while len(sigma.shape) < x_s_enc.ndim:
                    sigma = sigma.unsqueeze(-1)

                # Noisy latents: zt = (1 - sigma) * z + sigma * noise.
                # Keep frame 0 fixed so Stage 1 conditions on the true initial state
                # instead of trying to regenerate it.
                noisy_x_s_enc = (1.0 - sigma) * x_s_enc + sigma * noise # (B, T, N, d_state)
                noisy_x_s_enc[:, :1] = x_s_enc[:, :1]

                # Flow matching target (latent space): velocity = noise - z
                target = noise - x_s_enc    # sigma(t) = t          (B, T, N, d_state)

                # --- Initial frame conditioning ---
                # Repeat frame-0 latent on every timestep token. Zero-padding the
                # future steps removed the strongest per-point identity signal and
                # forced the model to rely on a weak 3D anchor plus object-level
                # conditions, which is a direct source of identity ambiguity.
                init_enc_1 = x_s_enc[:, :1]  # (B, 1, N, d_state) — already encoded, no grad
                init_enc_padded = init_enc_1.expand(-1, T, -1, -1).contiguous()  # (B, T, N, d_state)
                init_mask = torch.ones(B_sz, T, N_sz, 1,
                                       device=accelerator.device, dtype=weight_dtype)
                init_mask[:, 0, :, :] = 0.0  # first latent frame is given
                point_anchor_1 = _compute_point_anchor(
                    x_s_raw[:, :1], point_obj_idx, args.anchor_mode
                ).to(device=accelerator.device, dtype=weight_dtype)
                if args.disable_point_anchor:
                    point_anchor_1 = point_anchor_1[..., :0]
                point_anchor = point_anchor_1.expand(-1, T, -1, -1).contiguous()  # (B, T, N, d_anchor)

                # --- Build valid sequence mask for DiT attention (padded batch mode) ---
                if args.padded_batch:
                    t_latent = (T_raw_tensor - 1) // 4 + 1   # (B,)
                    t_idx = torch.arange(T, device=accelerator.device).unsqueeze(0)  # (1, T)
                    t_valid = t_idx < t_latent.unsqueeze(1)   # (B, T)
                    latent_seq_mask = t_valid.unsqueeze(2) & point_mask.unsqueeze(1)
                    valid_seq_mask = latent_seq_mask.view(B_sz, T * N_sz)
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

                # --- Forward pass: DiT in latent space (LDM-style) ---
                # AE is fully detached — no decode needed during training.
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
                    )  # (B, T, N, d_state) — predicted latent velocity

                # --- Loss computation (latent space) ---
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, sigmas=sigma,
                )
                loss_per_elem = F.mse_loss(pred_enc.float(), target.float(), reduction='none')

                if args.padded_batch and point_mask is not None:
                    # Build latent-space mask: (B, T, N). Padded positions are excluded
                    # from both the numerator and denominator (zero contribution, not counted).
                    t_latent = (T_raw_tensor - 1) // 4 + 1                 # (B,)
                    t_idx = torch.arange(T, device=accelerator.device).unsqueeze(0)  # (1, T)
                    t_valid = t_idx < t_latent.unsqueeze(1)                # (B, T)
                    latent_mask = t_valid.unsqueeze(2) & point_mask.unsqueeze(1)  # (B, T, N)
                    latent_mask[:, :1] = False
                    loss_per_elem = loss_per_elem * latent_mask.unsqueeze(-1).float()
                    n_valid = latent_mask.float().sum() * args.d_state
                    diffusion_loss = (loss_per_elem * weighting.float()).sum() / n_valid.clamp(min=1)
                else:
                    latent_mask = torch.ones(B_sz, T, N_sz, device=accelerator.device, dtype=torch.bool)
                    latent_mask[:, :1] = False
                    loss_per_elem = loss_per_elem * latent_mask.unsqueeze(-1).float()
                    n_valid = latent_mask.float().sum() * args.d_state
                    diffusion_loss = (loss_per_elem * weighting.float()).sum() / n_valid.clamp(min=1)

                local_dist_loss = x_s_raw.new_tensor(0.0)
                covariance_loss = x_s_raw.new_tensor(0.0)
                velocity_loss = x_s_raw.new_tensor(0.0)
                chamfer_loss = x_s_raw.new_tensor(0.0)
                momentum_loss = x_s_raw.new_tensor(0.0)
                floor_loss = x_s_raw.new_tensor(0.0)
                should_log_sim_metrics = (
                    args.sim_metrics_steps > 0
                    and accelerator.sync_gradients
                    and (global_step + 1) % args.sim_metrics_steps == 0
                )
                use_raw_auxiliary_loss = any([
                    args.lambda_local_dist > 0.0,
                    args.lambda_covariance > 0.0,
                    args.lambda_vel > 0.0,
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
                    pred_x0_raw = torch.cat([
                        ae.decode(pred_x0_enc[..., :d_latent], T_raw_dim),
                        ae.decode(pred_x0_enc[..., d_latent:], T_raw_dim),
                    ], dim=-1)

                    if args.padded_batch:
                        raw_t_idx = torch.arange(T_raw_dim, device=accelerator.device).unsqueeze(0)
                        valid_raw_frame_mask = raw_t_idx < T_raw_tensor.unsqueeze(1)
                    else:
                        valid_raw_frame_mask = None

                    pos_pred = pred_x0_raw[..., :3].float()
                    pos_gt = x_s_raw[..., :3].float()
                    velocity_pred = pred_x0_raw[..., 3:6].float()
                    velocity_gt = x_s_raw[..., 3:6].float()

                    if args.lambda_vel > 0.0:
                        velocity_loss = _compute_velocity_consistency_loss(
                            pos_pred=pos_pred,
                            pos_gt=pos_gt,
                            point_mask=point_mask,
                            valid_frame_mask=valid_raw_frame_mask,
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
                                pred_x0_raw[..., :3].float(),
                                x_s_raw[:, :1, ..., :3].float(),
                                point_mask=point_mask,
                            ))

                loss = (
                    args.lambda_diffusion * diffusion_loss
                    + args.lambda_vel * velocity_loss
                    + args.lambda_local_dist * local_dist_loss
                    + args.lambda_covariance * covariance_loss
                    + args.lambda_chamfer * chamfer_loss
                    + args.lambda_momentum * momentum_loss
                    + args.lambda_floor * floor_loss
                )

                # --- Backward ---
                avg_loss = accelerator.gather(loss.repeat(bsz)).mean()
                avg_diffusion_loss = accelerator.gather(diffusion_loss.repeat(bsz)).mean()
                avg_local_dist_loss = accelerator.gather(local_dist_loss.repeat(bsz)).mean()
                avg_covariance_loss = accelerator.gather(covariance_loss.repeat(bsz)).mean()
                avg_velocity_loss = accelerator.gather(velocity_loss.repeat(bsz)).mean()
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
                        "weighted_diffusion_loss": args.lambda_diffusion * avg_diffusion_loss.item(),
                        "local_dist_loss": avg_local_dist_loss.item(),
                        "covariance_loss": avg_covariance_loss.item(),
                        "velocity_loss": avg_velocity_loss.item(),
                        "chamfer_loss": avg_chamfer_loss.item(),
                        "momentum_loss": avg_momentum_loss.item(),
                        "floor_loss": avg_floor_loss.item(),
                        "weighted_local_dist_loss": args.lambda_local_dist * avg_local_dist_loss.item(),
                        "weighted_covariance_loss": args.lambda_covariance * avg_covariance_loss.item(),
                        "weighted_velocity_loss": args.lambda_vel * avg_velocity_loss.item(),
                        "weighted_chamfer_loss": args.lambda_chamfer * avg_chamfer_loss.item(),
                        "weighted_momentum_loss": args.lambda_momentum * avg_momentum_loss.item(),
                        "weighted_floor_loss": args.lambda_floor * avg_floor_loss.item(),
                        "lr": current_lr,
                        "epoch": epoch,
                        "global_step": global_step,
                }
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
                    unwrapped_sim.eval()
                    unwrapped_cond.eval()
                    try:
                        _run_vis(
                            vis_samples, unwrapped_sim, ae, unwrapped_cond,
                            global_step, accelerator.device, weight_dtype,
                            args.vis_num_inference_steps, args.vis_fps, args.anchor_mode,
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
                    accelerator.wait_for_everyone()
                    # Includes optimizer, scheduler, RNG, and prepared model state so
                    # --resume_from_checkpoint can continue exactly at a later time.
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
        logger.info(f"Saved final model to {save_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()

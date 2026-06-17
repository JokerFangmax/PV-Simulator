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
    parser.add_argument("--lambda_local_dist", type=float, default=0.0,
                        help="Weight for local distance consistency loss in raw position space.")
    parser.add_argument("--local_dist_k", type=int, default=8,
                        help="Number of same-object neighbors from frame0 used in local distance loss.")
    parser.add_argument("--anchor_mode", type=str, default="local",
                        choices=["local"],
                        help="How to encode point anchors from the initial frame. "
                             "'local' uses centered world coordinates. "
                             "The earlier 'canonical_pca' variant is disabled because "
                             "its basis was unstable across samples.")
    parser.add_argument("--disable_point_anchor", action="store_true",
                        help="Disable point_anchor input entirely for ablation.")
    parser.add_argument("--use_temporal_correspondence", action="store_true",
                        help="Add an explicit same-point temporal attention block before the "
                             "flat Transformer. This tests whether point collapse is caused by "
                             "missing correspondence structure rather than by local distance loss.")
    parser.add_argument("--use_temporal_rope", action="store_true",
                        help="Apply 1D RoPE inside the same-point temporal attention block. "
                             "This adds explicit frame-order information on each point track.")

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
    if args.overfit_single_sample_idx is not None and args.overfit_repeat_length < 1:
        parser.error("--overfit_repeat_length must be >= 1")

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
    k: int,
) -> torch.Tensor:
    """Compare same-object local distances using frame0 KNN from ground truth.

    This is a soft, data-derived structural prior: no explicit rigid graph is
    provided, only the initial frame's local neighborhood geometry.
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

        obj_ids = torch.unique(point_obj_idx[b][valid_points])
        for obj_id in obj_ids.tolist():
            obj_mask = (point_obj_idx[b] == obj_id) & valid_points
            obj_idx = obj_mask.nonzero(as_tuple=True)[0]
            n_obj = obj_idx.numel()
            if n_obj < 2:
                continue

            obj_gt0 = pos_gt[b, 0, obj_idx]  # (n_obj, 3)
            obj_gt = pos_gt[b, :, obj_idx]   # (T_raw, n_obj, 3)
            obj_pred = pos_pred[b, :, obj_idx]

            pairwise = torch.cdist(obj_gt0, obj_gt0)
            pairwise.fill_diagonal_(torch.finfo(pairwise.dtype).max)
            k_eff = min(k, n_obj - 1)
            nbr_local = torch.topk(pairwise, k=k_eff, largest=False).indices  # (n_obj, k_eff)

            gt_ref = obj_gt.unsqueeze(2).expand(-1, -1, k_eff, -1)
            gt_nbr = obj_gt[:, nbr_local, :]
            pred_ref = obj_pred.unsqueeze(2).expand(-1, -1, k_eff, -1)
            pred_nbr = obj_pred[:, nbr_local, :]

            gt_dist = torch.norm(gt_ref - gt_nbr, dim=-1)
            pred_dist = torch.norm(pred_ref - pred_nbr, dim=-1)
            obj_scale = gt_dist[:, :].mean().clamp_min(1e-2)
            rel_err = (pred_dist - gt_dist) / obj_scale
            obj_loss = rel_err.pow(2).mean()

            total_loss = total_loss + obj_loss * (n_obj * k_eff)
            total_weight = total_weight + (n_obj * k_eff)

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
        use_temporal_correspondence=args.use_temporal_correspondence,
        use_temporal_rope=args.use_temporal_rope,
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
                        dtype=weight_dtype, valid_seq_mask=valid_seq_mask,
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
                if args.lambda_local_dist > 0.0:
                    pred_x0_enc = noise - pred_enc
                    pred_x0_enc[:, :1] = x_s_enc[:, :1]
                    pred_x0_raw = torch.cat([
                        ae.decode(pred_x0_enc[..., :d_latent], T_raw_dim),
                        ae.decode(pred_x0_enc[..., d_latent:], T_raw_dim),
                    ], dim=-1)
                    local_dist_loss = _compute_local_distance_loss(
                        pos_pred=pred_x0_raw[..., :3].float(),
                        pos_gt=x_s_raw[..., :3].float(),
                        point_obj_idx=point_obj_idx,
                        point_mask=point_mask,
                        k=args.local_dist_k,
                    )

                loss = diffusion_loss + args.lambda_local_dist * local_dist_loss

                # --- Backward ---
                avg_loss = accelerator.gather(loss.repeat(bsz)).mean()
                avg_diffusion_loss = accelerator.gather(diffusion_loss.repeat(bsz)).mean()
                avg_local_dist_loss = accelerator.gather(local_dist_loss.repeat(bsz)).mean()
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
                accelerator.log(
                    {
                        "train_loss": train_loss / max(accum_count, 1),
                        "diffusion_loss": avg_diffusion_loss.item(),
                        "local_dist_loss": avg_local_dist_loss.item(),
                        "lr": current_lr,
                        "epoch": epoch,
                        "global_step": global_step,
                    },
                    step=global_step,
                )
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
                        logger.info(f"Saved checkpoint to {save_dir}")

                        gc.collect()
                        torch.cuda.empty_cache()

            logs = {
                "step_loss": loss.detach().item(),
                "diff_loss": diffusion_loss.detach().item(),
                "local_dist": local_dist_loss.detach().item(),
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

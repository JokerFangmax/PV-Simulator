"""Stage 2 training for PV-Simulator: Joint MoT training.

Loads Stage 1 Simulation Branch checkpoint + pre-trained Wan2.1 Video Branch.
Adds LoRA to the video branch and trains:
  - Joint Attention modules (JointAttention)
  - LoRA adapters in the video branch
  - Simulation branch (fine-tune, optional)

Uses gated cross-modal attention with bias_scale that decays from 1.0→0.0
over `bias_decay_steps` training steps (gradual transition from independent
to joint attention).

Usage:
    accelerate launch --num_processes=4 scripts/pv_simulator/train_stage2.py \
        --stage1_ckpt outputs/stage1/final \
        --video_model_path /path/to/Wan2.1-Fun-V1.1-1.3B-InP \
        --ann_path /path/to/annotations.json \
        --data_root /path/to/data \
        --output_dir outputs/stage2 \
        --gradient_accumulation_steps 4
"""

import argparse
import gc
import logging
import math
import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (compute_density_for_timestep_sampling,
                                      compute_loss_weighting_for_sd3)
from PIL import Image
from tqdm.auto import tqdm

current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]
for project_root in project_roots:
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from videox_fun.data.dataset_simulation import SimulationDataset, sim_collate_fn
from videox_fun.models import AutoencoderKLWan, WanT5EncoderModel, WanTransformer3DModel
from videox_fun.models.sim_ae import CausalAE
from videox_fun.models.sim_condition import SimConditionEmbedder
from videox_fun.models.sim_transformer import SimTransformer
from videox_fun.models.mot_wrapper import MoTWrapper, DEFAULT_PAIRING
from videox_fun.utils.discrete_sampler import DiscreteSampling
from videox_fun.utils.lora_utils import create_network

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: Joint MoT Training")

    # Checkpoints
    parser.add_argument("--stage1_ckpt", type=str, required=True,
                        help="Path to Stage 1 checkpoint directory.")
    parser.add_argument("--ae_ckpt_dir", type=str, required=True,
                        help="Path to Stage 0 CausalAE checkpoint directory.")
    parser.add_argument("--video_model_path", type=str, required=True,
                        help="Path to Wan2.1-Fun-V1.1-1.3B-InP pretrained weights.")

    # Data
    parser.add_argument("--ann_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=None)

    # Model architecture (must match Stage 1)
    parser.add_argument("--d_state", type=int, default=64)
    parser.add_argument("--d_sim", type=int, default=512)
    parser.add_argument("--sim_ffn_dim", type=int, default=2048)
    parser.add_argument("--sim_num_heads", type=int, default=8)
    parser.add_argument("--sim_num_layers", type=int, default=10)
    parser.add_argument("--max_objects", type=int, default=16)
    parser.add_argument("--d_joint", type=int, default=256)
    parser.add_argument("--joint_num_heads", type=int, default=8)

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_target_modules", type=str,
                        default="to_q,to_k,to_v,to_out.0",
                        help="Comma-separated list of LoRA target module names.")
    parser.add_argument("--train_sim_branch", action="store_true", default=True,
                        help="Whether to fine-tune sim branch in Stage 2.")
    parser.add_argument("--sim_lr_scale", type=float, default=0.1,
                        help="Learning rate multiplier for sim branch relative to joint attn.")

    # Joint attention bias decay
    parser.add_argument("--bias_decay_steps", type=int, default=5000,
                        help="Steps over which bias_scale decays from 1.0 to 0.0.")

    # Training
    parser.add_argument("--output_dir", type=str, default="outputs/stage2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=50000)
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)

    # Loss weights
    parser.add_argument("--sim_loss_weight", type=float, default=1.0,
                        help="Weight for simulation denoising loss.")
    parser.add_argument("--vid_loss_weight", type=float, default=1.0,
                        help="Weight for video denoising loss.")

    # Diffusion
    parser.add_argument("--train_sampling_steps", type=int, default=1000)
    parser.add_argument("--uniform_sampling", action="store_true")
    parser.add_argument("--weighting_scheme", type=str, default="logit_normal")
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)

    # Checkpointing
    parser.add_argument("--checkpointing_steps", type=int, default=2000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=5)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    args = parser.parse_args()
    assert args.train_batch_size == 1
    return args


def main():
    args = parse_args()

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

    # --- Load Video Branch ---
    logger.info(f"Loading video transformer from {args.video_model_path}")
    video_transformer = WanTransformer3DModel.from_pretrained(
        args.video_model_path, subfolder="transformer"
    ).to(accelerator.device, dtype=weight_dtype)

    # Freeze video branch base weights
    for p in video_transformer.parameters():
        p.requires_grad_(False)

    # Load VAE (frozen, for encoding video latents)
    vae = AutoencoderKLWan.from_pretrained(
        args.video_model_path, subfolder="vae"
    ).to(accelerator.device, dtype=weight_dtype)
    for p in vae.parameters():
        p.requires_grad_(False)

    # Load text encoder (frozen)
    text_encoder = WanT5EncoderModel.from_pretrained(
        args.video_model_path, subfolder="text_encoder"
    ).to(accelerator.device, dtype=weight_dtype)
    for p in text_encoder.parameters():
        p.requires_grad_(False)

    # --- Load frozen CausalAE (from Stage 0) ---
    ae = CausalAE.load(args.ae_ckpt_dir)
    ae.to(accelerator.device, dtype=weight_dtype)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    logger.info(f"Loaded frozen CausalAE from {args.ae_ckpt_dir}")

    # --- Build Sim Components ---
    sim_cond_embedder = SimConditionEmbedder(max_objects=args.max_objects)
    d_cond = sim_cond_embedder.d_cond

    sim_transformer = SimTransformer(
        d_state=args.d_state,
        d_cond=d_cond,
        d_sim=args.d_sim,
        ffn_dim=args.sim_ffn_dim,
        num_heads=args.sim_num_heads,
        num_layers=args.sim_num_layers,
    )

    # Load Stage 1 weights
    logger.info(f"Loading Stage 1 checkpoint from {args.stage1_ckpt}")
    sim_transformer.load_state_dict(
        torch.load(os.path.join(args.stage1_ckpt, "sim_transformer.pt"), map_location="cpu")
    )
    sim_cond_embedder.load_state_dict(
        torch.load(os.path.join(args.stage1_ckpt, "sim_cond_embedder.pt"), map_location="cpu")
    )

    # Move to device
    sim_transformer.to(accelerator.device, dtype=weight_dtype)
    sim_cond_embedder.to(accelerator.device, dtype=weight_dtype)

    if not args.train_sim_branch:
        for p in sim_transformer.parameters():
            p.requires_grad_(False)
        for p in sim_cond_embedder.parameters():
            p.requires_grad_(False)

    # --- Build MoT Wrapper ---
    mot = MoTWrapper(
        video_transformer=video_transformer,
        d_state=args.d_state,
        d_cond=d_cond,
        d_sim=args.d_sim,
        sim_ffn_dim=args.sim_ffn_dim,
        sim_num_heads=args.sim_num_heads,
        sim_num_layers=args.sim_num_layers,
        d_joint=args.d_joint,
        joint_num_heads=args.joint_num_heads,
        max_objects=args.max_objects,
    )
    # Copy loaded sim components into mot
    mot.sim.load_state_dict(sim_transformer.state_dict())
    mot.sim_cond_embedder.load_state_dict(sim_cond_embedder.state_dict())
    mot.to(accelerator.device, dtype=weight_dtype)

    # --- Inject LoRA into Video Branch ---
    lora_network = create_network(
        1.0,
        args.lora_rank,
        args.lora_alpha,
        text_encoder=None,
        unet=video_transformer,
        target_name=args.lora_target_modules,
    )
    lora_network.to(accelerator.device, dtype=weight_dtype)
    lora_network.apply_to(
        text_encoder=None,
        unet=video_transformer,
        apply_text_encoder=False,
        apply_unet=True,
    )

    # --- Optimizer ---
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # Param groups: joint attention + LoRA at base lr; sim branch at lower lr
    joint_attn_params = list(mot.joint_attns.parameters())
    lora_params = list(filter(lambda p: p.requires_grad, lora_network.parameters()))

    param_groups = [
        {"params": joint_attn_params, "lr": args.learning_rate},
        {"params": lora_params, "lr": args.learning_rate},
    ]
    if args.train_sim_branch:
        sim_params = (
            list(mot.sim.parameters()) +
            list(mot.sim_cond_embedder.parameters())
        )
        param_groups.append({"params": sim_params, "lr": args.learning_rate * args.sim_lr_scale})

    optimizer = optimizer_cls(
        param_groups,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # --- Dataset ---
    train_dataset = SimulationDataset(
        ann_path=args.ann_path,
        data_root=args.data_root,
        load_video=True,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=sim_collate_fn,
        pin_memory=True,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=args.train_sampling_steps,
    )

    # Prepare with accelerate — wrap trainable components
    mot, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        mot, optimizer, train_dataloader, lr_scheduler,
    )

    # --- Resume ---
    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        global_step = int(os.path.basename(args.resume_from_checkpoint).split("-")[1])
        first_epoch = global_step // len(train_dataloader)

    idx_sampling = DiscreteSampling(args.train_sampling_steps, uniform_sampling=args.uniform_sampling)
    torch_rng = torch.Generator(device=accelerator.device).manual_seed(args.seed + accelerator.process_index)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    accelerator.init_trackers("pv_simulator_stage2", config=vars(args))

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(mot):
                # --- Sim inputs ---
                x_s_raw = batch['x_s_raw'].to(weight_dtype)          # (1, T_raw, N, 6)
                c_force_raw = batch['c_force_raw'].to(weight_dtype)  # (1, T_raw, N, 6)
                c_floor = batch['c_floor'].to(accelerator.device)
                c_id = batch['c_id'].squeeze(0).to(accelerator.device)
                c_mat = batch['c_mat'].squeeze(0).to(accelerator.device)
                c_mass = batch['c_mass'].squeeze(0).to(accelerator.device)
                c_static = batch['c_static'].squeeze(0).to(accelerator.device)
                point_obj_idx = batch['point_obj_idx'].squeeze(0).to(accelerator.device)
                T = batch['T']
                T_raw_val = batch['T_raw']
                N = batch['N']

                unwrapped_mot = accelerator.unwrap_model(mot)

                # --- Encode sim states via frozen AE ---
                with torch.no_grad():
                    pos_enc = ae.encode(x_s_raw[..., :3])    # (1, T, N, d_latent)
                    vel_enc = ae.encode(x_s_raw[..., 3:6])   # (1, T, N, d_latent)
                x_s_enc = torch.cat([pos_enc, vel_enc], dim=-1)  # (1, T, N, d_state)

                # --- Initial frame conditioning via frozen AE ---
                with torch.no_grad():
                    init_pos_enc = ae.encode(x_s_raw[:, :1, :, :3])
                    init_vel_enc = ae.encode(x_s_raw[:, :1, :, 3:6])
                init_enc_1 = torch.cat([init_pos_enc, init_vel_enc], dim=-1)
                init_enc_padded = torch.cat([
                    init_enc_1,
                    torch.zeros(1, T - 1, N, args.d_state,
                                device=accelerator.device, dtype=weight_dtype),
                ], dim=1)
                init_mask = torch.ones(1, T, N, 1,
                                       device=accelerator.device, dtype=weight_dtype)
                init_mask[:, 0, :, :] = 0.0

                c_sim = unwrapped_mot.sim_cond_embedder(
                    c_floor=c_floor, c_id=c_id, c_mat=c_mat, c_mass=c_mass,
                    c_static=c_static, c_force_raw=c_force_raw,
                    point_obj_idx=point_obj_idx, T=T,
                )  # (1, T, N, d_cond)

                # --- Sample noise & timestep ---
                noise_sim = torch.randn_like(x_s_enc)
                bsz = 1

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

                sigma_sim = sigma.clone()
                while len(sigma_sim.shape) < x_s_enc.ndim:
                    sigma_sim = sigma_sim.unsqueeze(-1)

                noisy_x_s_enc = (1.0 - sigma_sim) * x_s_enc + sigma_sim * noise_sim
                target_sim = noise_sim - x_s_enc

                # --- Compute bias_scale (decays 1.0 → 0.0 over bias_decay_steps) ---
                bias_scale = max(0.0, 1.0 - global_step / max(args.bias_decay_steps, 1))

                # --- Prepare sim tokens ---
                sim_x, sim_e0 = unwrapped_mot.prepare_sim_inputs(
                    noisy_x_s_enc, init_enc_padded, init_mask, c_sim, timesteps,
                )

                # --- Stage 2: joint forward pass ---
                # For Stage 2 we need video data. If video_path is available, load it;
                # otherwise run sim-only path (joint attn inactive while bias_scale=1.0).
                has_video = 'video_path' in batch and batch['video_path']

                if has_video:
                    # Load and encode video — deferred to full pipeline integration.
                    # For now, run sim-only path; video branch training requires
                    # a full video loading pipeline (out of scope for this script skeleton).
                    # TODO: integrate video loading and joint forward pass.
                    with torch.amp.autocast("cuda", dtype=weight_dtype):
                        pred_sim = unwrapped_mot.sim(
                            noisy_x_s_enc, init_enc_padded, init_mask, c_sim,
                            timesteps, dtype=weight_dtype)
                else:
                    with torch.amp.autocast("cuda", dtype=weight_dtype):
                        pred_sim = unwrapped_mot.sim(
                            noisy_x_s_enc, init_enc_padded, init_mask, c_sim,
                            timesteps, dtype=weight_dtype)

                # --- Sim loss ---
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, sigmas=sigma_sim,
                )
                loss_sim = (F.mse_loss(pred_sim.float(), target_sim.float(), reduction='none')
                            * weighting.float()).mean()
                loss = args.sim_loss_weight * loss_sim

                avg_loss = accelerator.gather(loss.repeat(1)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    all_params = joint_attn_params + lora_params
                    if args.train_sim_branch:
                        all_params += sim_params
                    accelerator.clip_grad_norm_(all_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({
                    "train_loss": train_loss,
                    "bias_scale": bias_scale,
                }, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = [
                                d for d in os.listdir(args.output_dir)
                                if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                for removing_ckpt in checkpoints[:len(checkpoints) - args.checkpoints_total_limit + 1]:
                                    shutil.rmtree(os.path.join(args.output_dir, removing_ckpt))

                        save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_dir, exist_ok=True)
                        unwrapped = accelerator.unwrap_model(mot)

                        # Save joint attention + sim components
                        torch.save(unwrapped.joint_attns.state_dict(),
                                   os.path.join(save_dir, "joint_attns.pt"))
                        torch.save(unwrapped.sim.state_dict(),
                                   os.path.join(save_dir, "sim_transformer.pt"))
                        torch.save(unwrapped.sim_cond_embedder.state_dict(),
                                   os.path.join(save_dir, "sim_cond_embedder.pt"))
                        # Save LoRA weights
                        lora_network.save_weights(
                            os.path.join(save_dir, "lora.safetensors"),
                            weight_dtype, None,
                        )
                        logger.info(f"Saved checkpoint to {save_dir}")
                        gc.collect()
                        torch.cuda.empty_cache()

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0],
                    "bias_scale": bias_scale}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_dir = os.path.join(args.output_dir, "final")
        os.makedirs(save_dir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(mot)
        torch.save(unwrapped.joint_attns.state_dict(), os.path.join(save_dir, "joint_attns.pt"))
        torch.save(unwrapped.sim.state_dict(), os.path.join(save_dir, "sim_transformer.pt"))
        torch.save(unwrapped.sim_cond_embedder.state_dict(), os.path.join(save_dir, "sim_cond_embedder.pt"))
        lora_network.save_weights(os.path.join(save_dir, "lora.safetensors"), weight_dtype, None)
        logger.info(f"Saved final model to {save_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()

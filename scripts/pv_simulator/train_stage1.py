"""Stage 1 training for PV-Simulator: Simulation Branch only.

Trains the SimTransformer + CausalTemporalEncoder/Decoder + SimConditionEmbedder
on physics trajectory data using flow matching diffusion loss.

No video branch is loaded. No Joint Attention is used.

Usage:
    accelerate launch --num_processes=4 scripts/pv_simulator/train_stage1.py \
        --data_root /path/to/sim_data \
        --ann_path /path/to/annotations.json \
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

from videox_fun.data.dataset_simulation import SimulationDataset, sim_collate_fn
from videox_fun.models.sim_causal_encoder import CausalTemporalEncoder, CausalTemporalDecoder
from videox_fun.models.sim_condition import SimConditionEmbedder
from videox_fun.models.sim_transformer import SimTransformer
from videox_fun.utils.discrete_sampler import DiscreteSampling

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Train Simulation Branch")

    # Data
    parser.add_argument("--ann_path", type=str, required=True, help="Path to annotation JSON file.")
    parser.add_argument("--data_root", type=str, default=None, help="Root directory for data files.")

    # Model architecture
    parser.add_argument("--d_state", type=int, default=256, help="Encoded point state dimension.")
    parser.add_argument("--d_sim", type=int, default=512, help="SimTransformer hidden dimension.")
    parser.add_argument("--sim_ffn_dim", type=int, default=2048, help="SimTransformer FFN dimension.")
    parser.add_argument("--sim_num_heads", type=int, default=8, help="SimTransformer attention heads.")
    parser.add_argument("--sim_num_layers", type=int, default=10, help="SimTransformer blocks.")
    parser.add_argument("--state_enc_mid", type=int, default=128, help="Causal encoder intermediate channels.")
    parser.add_argument("--max_objects", type=int, default=16, help="Maximum number of objects per scene.")

    # Training
    parser.add_argument("--output_dir", type=str, default="outputs/stage1", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_batch_size", type=int, default=1, help="Must be 1 (variable shapes).")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
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
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
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

    # Checkpointing & logging
    parser.add_argument("--checkpointing_steps", type=int, default=5000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=5)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    args = parser.parse_args()
    assert args.train_batch_size == 1, "SimulationDataset requires batch_size=1"
    return args


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

    # --- Build Models ---
    sim_cond_embedder = SimConditionEmbedder(max_objects=args.max_objects)
    d_cond = sim_cond_embedder.d_cond

    x_s_encoder = CausalTemporalEncoder(c_in=9, c_mid=args.state_enc_mid, d_out=args.d_state)
    x_s_decoder = CausalTemporalDecoder(d_feat=args.d_state, c_mid=args.state_enc_mid, c_out=6)

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
    x_s_encoder.to(accelerator.device, dtype=weight_dtype)
    x_s_decoder.to(accelerator.device, dtype=weight_dtype)
    sim_transformer.to(accelerator.device, dtype=weight_dtype)

    # Count parameters
    total_params = (
        sum(p.numel() for p in sim_transformer.parameters()) +
        sum(p.numel() for p in x_s_encoder.parameters()) +
        sum(p.numel() for p in x_s_decoder.parameters()) +
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
        list(x_s_encoder.parameters()) +
        list(x_s_decoder.parameters()) +
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
    train_dataset = SimulationDataset(
        ann_path=args.ann_path,
        data_root=args.data_root,
        load_video=False,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=sim_collate_fn,
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
    sim_transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        sim_transformer, optimizer, train_dataloader, lr_scheduler,
    )
    # These don't need gradient wrapping but should be on the right device
    x_s_encoder = x_s_encoder.to(accelerator.device)
    x_s_decoder = x_s_decoder.to(accelerator.device)
    sim_cond_embedder = sim_cond_embedder.to(accelerator.device)

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

    accelerator.init_trackers("pv_simulator_stage1", config=vars(args))

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(sim_transformer):
                # --- Unpack batch (bs=1) ---
                x_s_raw = batch['x_s_raw'].to(weight_dtype)         # (1, T_raw, N, 9)
                c_force_raw = batch['c_force_raw'].to(weight_dtype) # (1, T_raw, N, 6)
                c_floor = batch['c_floor'].to(accelerator.device)   # (1,)
                c_id = batch['c_id'].squeeze(0).to(accelerator.device)         # (n_objects,)
                c_mat = batch['c_mat'].squeeze(0).to(accelerator.device)       # (n_objects, 2)
                c_mass = batch['c_mass'].squeeze(0).to(accelerator.device)     # (n_objects,)
                c_static = batch['c_static'].squeeze(0).to(accelerator.device) # (n_objects,)
                c_init = batch['c_init'].squeeze(0).to(accelerator.device)     # (n_objects, 10)
                point_obj_idx = batch['point_obj_idx'].squeeze(0).to(accelerator.device)  # (N,)
                T = batch['T']
                T_raw = batch['T_raw']

                # --- Encode point states ---
                x_s_enc = x_s_encoder(x_s_raw)  # (1, T, N, d_state)

                # --- Encode conditions ---
                c_sim = sim_cond_embedder(
                    c_floor=c_floor,
                    c_id=c_id,
                    c_mat=c_mat,
                    c_mass=c_mass,
                    c_static=c_static,
                    c_force_raw=c_force_raw,
                    c_init=c_init,
                    point_obj_idx=point_obj_idx,
                    T=T,
                )  # (1, T, N, d_cond)

                # --- Flow matching noise ---
                noise = torch.randn_like(x_s_enc)
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

                # Get sigmas for flow matching
                sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=weight_dtype)
                schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
                step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
                sigma = sigmas[step_indices].flatten()
                # Expand sigma to match x_s_enc shape
                while len(sigma.shape) < x_s_enc.ndim:
                    sigma = sigma.unsqueeze(-1)

                # Noisy latents: zt = (1 - sigma) * x + sigma * noise
                noisy_x_s_enc = (1.0 - sigma) * x_s_enc + sigma * noise

                # Flow matching target: noise - x
                target = noise - x_s_enc

                # --- Forward pass ---
                with torch.cuda.amp.autocast(dtype=weight_dtype):
                    pred = sim_transformer(noisy_x_s_enc, c_sim, timesteps, dtype=weight_dtype)

                # --- Loss computation ---
                # Latent-space loss (primary)
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, sigmas=sigma,
                )
                loss_latent = F.mse_loss(pred.float(), target.float(), reduction='none')
                loss_latent = (loss_latent * weighting.float()).mean()

                loss = loss_latent

                # --- Backward ---
                avg_loss = accelerator.gather(loss.repeat(1)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

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
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

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

                        # Save all sim components
                        save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_dir, exist_ok=True)

                        unwrapped_sim = accelerator.unwrap_model(sim_transformer)
                        torch.save(unwrapped_sim.state_dict(), os.path.join(save_dir, "sim_transformer.pt"))
                        torch.save(x_s_encoder.state_dict(), os.path.join(save_dir, "x_s_encoder.pt"))
                        torch.save(x_s_decoder.state_dict(), os.path.join(save_dir, "x_s_decoder.pt"))
                        torch.save(sim_cond_embedder.state_dict(), os.path.join(save_dir, "sim_cond_embedder.pt"))
                        logger.info(f"Saved checkpoint to {save_dir}")

                        gc.collect()
                        torch.cuda.empty_cache()

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_dir = os.path.join(args.output_dir, "final")
        os.makedirs(save_dir, exist_ok=True)
        unwrapped_sim = accelerator.unwrap_model(sim_transformer)
        torch.save(unwrapped_sim.state_dict(), os.path.join(save_dir, "sim_transformer.pt"))
        torch.save(x_s_encoder.state_dict(), os.path.join(save_dir, "x_s_encoder.pt"))
        torch.save(x_s_decoder.state_dict(), os.path.join(save_dir, "x_s_decoder.pt"))
        torch.save(sim_cond_embedder.state_dict(), os.path.join(save_dir, "sim_cond_embedder.pt"))
        logger.info(f"Saved final model to {save_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()

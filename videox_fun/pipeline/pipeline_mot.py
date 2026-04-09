"""Stage 2 inference pipeline for PV-Simulator: Joint MoT (Video + Simulation).

Runs both the Video Branch (Wan2.1 + LoRA) and Simulation Branch together,
with JointAttention at paired blocks. Jointly denoises video latents and
physics trajectory latents from pure noise given a first-frame image,
text prompt, and physics conditions.

Usage:
    from videox_fun.pipeline.pipeline_mot import MoTPipeline

    pipeline = MoTPipeline.from_pretrained(
        video_model_path="path/to/Wan2.1-Fun-V1.1-1.3B-InP",
        stage2_ckpt="outputs/stage2/final",
        lora_path="outputs/stage2/final/lora.safetensors",
    )
    result = pipeline(
        prompt="A red ball bouncing on a floor",
        image=first_frame_pil,
        c_floor=0.0,
        c_id=torch.tensor([0]),
        c_mat=torch.tensor([[0.5, 0.3]]),
        c_mass=torch.tensor([1.0]),
        c_static=torch.tensor([0]),
        c_force_raw=torch.zeros(1, T_raw, N, 6),
        c_init=torch.zeros(1, 10),
        point_obj_idx=torch.zeros(N, dtype=torch.long),
        T=17,
        height=480,
        width=640,
        num_inference_steps=50,
    )
    video_frames = result['video']    # (T, H, W, 3) uint8
    x_s = result['x_s']              # (1, T_raw, N, 6)
"""

import math
import os
from typing import List, Optional, Union

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from videox_fun.models import AutoencoderKLWan, WanT5EncoderModel, WanTransformer3DModel
from videox_fun.models.mot_wrapper import MoTWrapper
from videox_fun.models.sim_condition import SimConditionEmbedder
from videox_fun.utils.lora_utils import merge_lora


class MoTPipeline:
    """Joint Video + Simulation inference pipeline.

    Performs joint denoising of video latents and physics trajectory latents
    using the MoTWrapper with JointAttention at paired blocks.

    The bias_scale during inference is set to (1 - t/T_max) so that:
    - At high noise (t ≈ T_max): bias_scale ≈ 0, gates open, strong joint attention
    - At low noise (t ≈ 0): bias_scale ≈ 1, gates push toward closed, detail refinement

    Args:
        mot: MoTWrapper with both branches.
        vae: AutoencoderKLWan for video encoding/decoding.
        text_encoder: WanT5EncoderModel.
        tokenizer: AutoTokenizer.
        scheduler: FlowMatchEulerDiscreteScheduler.
    """

    def __init__(
        self,
        mot: MoTWrapper,
        vae: AutoencoderKLWan,
        text_encoder: WanT5EncoderModel,
        tokenizer: AutoTokenizer,
        scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None,
    ):
        self.mot = mot
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler or FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
        self.device = next(mot.parameters()).device
        self.dtype = next(mot.parameters()).dtype

    @classmethod
    def from_pretrained(
        cls,
        video_model_path: str,
        stage2_ckpt: str,
        lora_path: Optional[str] = None,
        device: str = "cuda",
        dtype=torch.bfloat16,
        max_objects: int = 16,
        d_state: int = 256,
        d_sim: int = 512,
        sim_ffn_dim: int = 2048,
        sim_num_heads: int = 8,
        sim_num_layers: int = 10,
        d_joint: int = 256,
        joint_num_heads: int = 8,
        state_enc_mid: int = 128,
    ):
        """Load a MoTPipeline from pretrained checkpoints."""
        # Load video components
        tokenizer = AutoTokenizer.from_pretrained(video_model_path, subfolder="tokenizer")
        text_encoder = WanT5EncoderModel.from_pretrained(
            video_model_path, subfolder="text_encoder",
        ).to(device, dtype=dtype)
        vae = AutoencoderKLWan.from_pretrained(
            video_model_path, subfolder="vae",
        ).to(device, dtype=dtype)
        video_transformer = WanTransformer3DModel.from_pretrained(
            video_model_path, subfolder="transformer",
        ).to(device, dtype=dtype)

        # Build sim components for d_cond calculation
        sim_cond_embedder_tmp = SimConditionEmbedder(max_objects=max_objects)
        d_cond = sim_cond_embedder_tmp.d_cond

        # Build MoT
        mot = MoTWrapper(
            video_transformer=video_transformer,
            d_state=d_state,
            d_cond=d_cond,
            d_sim=d_sim,
            sim_ffn_dim=sim_ffn_dim,
            sim_num_heads=sim_num_heads,
            sim_num_layers=sim_num_layers,
            d_joint=d_joint,
            joint_num_heads=joint_num_heads,
            max_objects=max_objects,
            state_enc_mid=state_enc_mid,
        )

        # Load Stage 2 weights
        mot.joint_attns.load_state_dict(
            torch.load(os.path.join(stage2_ckpt, "joint_attns.pt"), map_location="cpu"))
        mot.sim.load_state_dict(
            torch.load(os.path.join(stage2_ckpt, "sim_transformer.pt"), map_location="cpu"))
        mot.x_s_encoder.load_state_dict(
            torch.load(os.path.join(stage2_ckpt, "x_s_encoder.pt"), map_location="cpu"))
        mot.x_s_decoder.load_state_dict(
            torch.load(os.path.join(stage2_ckpt, "x_s_decoder.pt"), map_location="cpu"))
        mot.sim_cond_embedder.load_state_dict(
            torch.load(os.path.join(stage2_ckpt, "sim_cond_embedder.pt"), map_location="cpu"))

        # Optionally merge LoRA
        if lora_path is not None:
            merge_lora(video_transformer, lora_path)

        mot = mot.to(device, dtype=dtype)

        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
        return cls(mot, vae, text_encoder, tokenizer, scheduler)

    def to(self, device=None, dtype=None):
        self.device = device or self.device
        self.dtype = dtype or self.dtype
        for m in [self.mot, self.vae, self.text_encoder]:
            m.to(device=device, dtype=dtype)
        return self

    def _encode_text(self, prompt: str, max_length: int = 512):
        """Encode text prompt to embeddings."""
        ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = ids.input_ids.to(self.device)
        attention_mask = ids.attention_mask.to(self.device)
        seq_len = attention_mask.gt(0).sum(dim=1).long()
        with torch.no_grad():
            embeds = self.text_encoder(input_ids, attention_mask=attention_mask)[0]
        return [embeds[0, :seq_len[0]]]

    def _preprocess_image(self, image: Union[Image.Image, torch.Tensor],
                          height: int, width: int) -> torch.Tensor:
        """Preprocess PIL image to (C, H, W) tensor normalized to [-1, 1]."""
        if isinstance(image, Image.Image):
            image = image.convert("RGB").resize((width, height), Image.LANCZOS)
            image = torch.from_numpy(np.array(image)).float() / 255.0
            image = (image - 0.5) / 0.5
            image = image.permute(2, 0, 1)  # (3, H, W)
        return image.to(self.device, dtype=self.dtype)

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, torch.Tensor]],  # first-frame condition
        c_floor: torch.Tensor,          # (1,) float
        c_id: torch.Tensor,              # (n_objects,) int
        c_mat: torch.Tensor,             # (n_objects, 2) float
        c_mass: torch.Tensor,            # (n_objects,) float
        c_static: torch.Tensor,          # (n_objects,) int
        c_force_raw: torch.Tensor,       # (1, T_raw, N, 6)
        c_init: torch.Tensor,            # (n_objects, 10)
        point_obj_idx: torch.Tensor,     # (N,) int
        T: int,                          # number of latent frames
        height: int = 480,
        width: int = 640,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        negative_prompt: str = "",
        generator: Optional[torch.Generator] = None,
        show_progress: bool = True,
        max_sequence_length: int = 512,
    ):
        """Run joint video + simulation denoising.

        Returns:
            dict with:
              'video': (T, H, W, 3) uint8 numpy array — decoded video frames
              'x_s': (1, T_raw, N, 6) tensor — decoded point states (pos + vel)
        """
        device = self.device
        dtype = self.dtype
        T_raw = 4 * (T - 1) + 1
        N = point_obj_idx.shape[0]
        do_cfg = guidance_scale > 1.0

        # --- Text embeddings ---
        prompt_embeds = self._encode_text(prompt, max_sequence_length)
        if do_cfg:
            neg_embeds = self._encode_text(negative_prompt or "", max_sequence_length)

        # --- First-frame image conditioning ---
        if image is not None:
            img_tensor = self._preprocess_image(image, height, width)
            img_tensor = img_tensor.unsqueeze(1)  # (C, 1, H, W) — single frame
            # Mask: 0 = condition, 1 = generate
            mask = torch.ones(1, 1, T, height // 8, width // 8, device=device, dtype=dtype)
            mask[:, :, 0] = 0.0  # first latent frame is conditioned

            # Encode condition image to latent
            img_for_vae = img_tensor.unsqueeze(0)  # (1, C, 1, H, W)
            cond_latent = self.vae.encode(img_for_vae).latent_dist.sample()
            # Repeat to fill all frames (will be masked)
            masked_latent = cond_latent.expand(-1, -1, T, -1, -1)
            inpaint_latents = torch.cat([mask, masked_latent], dim=1)  # (1, C+1, T, H', W')
        else:
            inpaint_latents = None

        # --- Sim conditions (constant) ---
        c_floor = c_floor.to(device)
        c_id = c_id.to(device)
        c_mat = c_mat.to(device, dtype=dtype)
        c_mass = c_mass.to(device, dtype=dtype)
        c_static = c_static.to(device)
        c_force_raw = c_force_raw.to(device, dtype=dtype)
        c_init = c_init.to(device, dtype=dtype)
        point_obj_idx = point_obj_idx.to(device)

        c_sim = self.mot.sim_cond_embedder(
            c_floor=c_floor, c_id=c_id, c_mat=c_mat, c_mass=c_mass,
            c_static=c_static, c_force_raw=c_force_raw, c_init=c_init,
            point_obj_idx=point_obj_idx, T=T,
        )  # (1, T, N, d_cond)

        # --- Initialize latents from noise ---
        latent_h = height // (self.vae.config.spatial_compression_ratio or 8)
        latent_w = width // (self.vae.config.spatial_compression_ratio or 8)
        latent_c = self.vae.config.latent_channels or 16

        vid_latents = torch.randn(
            1, latent_c, T, latent_h, latent_w,
            device=device, dtype=dtype, generator=generator,
        )
        sim_latents = torch.randn(
            1, T, N, self.mot.sim.d_state,
            device=device, dtype=dtype, generator=generator,
        )

        # --- Compute seq_len for video branch ---
        patch_size = self.mot.video.config.patch_size  # (1, 2, 2)
        seq_len = math.ceil(
            (latent_h * latent_w) / (patch_size[1] * patch_size[2]) * T
        )

        # --- Denoising loop ---
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        T_max = self.scheduler.config.num_train_timesteps

        progress_bar = tqdm(timesteps, desc="Joint denoising", disable=not show_progress)

        for t in progress_bar:
            t_batch = t.unsqueeze(0)

            # bias_scale: open gates at high noise, close at low noise
            bias_scale = float(t.item()) / float(T_max)

            # Prepare sim tokens & modulation
            sim_x, sim_e0 = self.mot.prepare_sim_inputs(sim_latents, c_sim, t_batch)

            # CFG: run with and without conditioning
            if do_cfg:
                # Video latents doubled for CFG
                vid_input = torch.cat([vid_latents] * 2, dim=0)
                context_list = prompt_embeds + neg_embeds if do_cfg else prompt_embeds
            else:
                vid_input = vid_latents
                context_list = prompt_embeds

            # Reshape video latents to list format expected by transformer
            x_list = [vid_input[i] for i in range(vid_input.shape[0])]
            context_list_fmt = [context_list[0]] * len(x_list)

            with torch.cuda.amp.autocast(dtype=dtype):
                vid_pred, sim_out = self.mot.forward_joint(
                    x=x_list,
                    t=t_batch.expand(len(x_list)),
                    context=context_list_fmt,
                    seq_len=seq_len,
                    y=inpaint_latents,
                    sim_x=sim_x,
                    sim_e0=sim_e0,
                    bias_scale=bias_scale,
                    dtype=dtype,
                )

            # CFG on video
            if do_cfg:
                vid_pred_cond, vid_pred_uncond = vid_pred.chunk(2)
                vid_pred_final = vid_pred_uncond + guidance_scale * (vid_pred_cond - vid_pred_uncond)
            else:
                vid_pred_final = vid_pred

            # Decode sim tokens back to latent space
            sim_pred = self.mot.decode_sim_output(sim_out, T, N, T_raw)

            # Scheduler step
            vid_latents = self.scheduler.step(vid_pred_final, t, vid_latents).prev_sample
            sim_latents = self.scheduler.step(sim_pred, t, sim_latents).prev_sample

        # --- Decode outputs ---
        # Video: VAE decode
        video_decoded = self.vae.decode(vid_latents).sample  # (1, C, T, H, W)
        video_decoded = video_decoded.clamp(-1, 1)
        video_frames = ((video_decoded[0].permute(1, 2, 3, 0) + 1) / 2 * 255).byte()  # (T, H, W, 3)

        # Simulation: causal decoder
        x_s = self.mot.x_s_decoder(sim_latents, T_raw)  # (1, T_raw, N, 6)

        return {
            'video': video_frames.cpu().numpy(),
            'x_s': x_s,
        }

# Run Summary

## Current state

- Status: running
- Environment: `videox`
- tmux session: `stage1_50k_padded_videox_gpu4_7`
- Log file:
  `experiments/stage1_50k/logs/stage1_50k_padded_videox_gpu4_7.log`
- Output directory:
  `outputs/stage1/stage1_50k_padded`
- W&B project/run:
  `pv-simulator-stage1` / `stage1_50k_padded_videox_gpu4_7`
- W&B mode: `offline`
- Offline run dir:
  `wandb/offline-run-20260507_105930-a0s31117`

## Previous run closed by request

- Previous tmux session: `stage1_50k_padded_videox`
- Previous log:
  `experiments/stage1_50k/logs/stage1_50k_padded_videox.log`
- Previous observed progress before stop:
  - `24/100000`
  - `step_loss=1.68`
  - `lr=1.2e-5`

## Relaunch target

- New tmux session: `stage1_50k_padded_videox_gpu4_7`
- New log:
  `experiments/stage1_50k/logs/stage1_50k_padded_videox_gpu4_7.log`
- Launcher: `python -m accelerate.commands.launch`
- GPU binding: `CUDA_VISIBLE_DEVICES=4,7`
- Processes: `2`
- Per-process batch size: `4`
- Effective global batch size: `8`

## Latest observed progress

- Dataset load: passed
- Multi-GPU init: passed
- W&B init: passed
- Training progress observed in new log:
  - `7/100000`
  - `step_loss=1.53`
  - `lr=3.5e-6`

## Remaining TODOs

- Keep monitoring until checkpoint creation is confirmed
- Record the first checkpoint path once it is written

## Concurrent run requested later

- New tmux session target: `stage1_50k_padded_videox_gpu3_6_vis200`
- New output directory:
  `outputs/stage1/stage1_50k_padded_gpu3_6_vis200`
- New log:
  `experiments/stage1_50k/logs/stage1_50k_padded_videox_gpu3_6_vis200.log`
- GPUs: `3,6`
- W&B mode: `online`
- W&B run:
  `stage1_50k_padded_videox_gpu3_6_vis200`
- W&B URL:
  `https://wandb.ai/raineggplant-tsinghua-university/pv-simulator-stage1/runs/vsum2f8u`
- Visualization flags:
  - `--vis_steps 200`
  - `--num_vis_samples 3`
- Current status: running
- Latest observed progress:
  - `1/100000`
  - `step_loss=1.53`
  - `lr=5e-7`

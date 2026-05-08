# Stage 1 on MOVI-AB 50k Shards

This experiment bundle tracks the Stage 1 SimDiT run against the new 50k MOVI-AB sharded dataset at `datasets/movi_ab_50k_shards`.

## Goal

- Continue Stage 1 training with the 50k dataset without modifying existing Stage 0 sweep artifacts in `sweep/`.
- Reuse the completed Stage 0 autoencoder checkpoint recommended by the Stage 0 sweep analysis.
- Keep the code path as close as possible to the existing Stage 1 `movi` training flow.

## Dataset

- Primary dataset root: `datasets/movi_ab_50k_shards`
- Internal format: `webdataset` shards under `datasets/movi_ab_50k_shards/webdataset/shards`
- Samples discovered from: `datasets/movi_ab_50k_shards/webdataset/manifest.jsonl`
- Shard manifest: `datasets/movi_ab_50k_shards/webdataset/dataset_manifest.json`

## Environment

- Repo guidance still mentions `asr`
- Initial requested runtime environment was `pv`, but that env was missing core ML/runtime packages
- Sanity checks in this record were therefore run in conda env `videox`
- The full Stage 1 run in this record is launched in conda env `videox`
- `pv` is no longer used for this Stage 1 run

## Stage 0 Checkpoint

- Recommended AE from Stage 0 sweep analysis:
  `outputs/stage0/sweeps_mae/training/lr1e-3-cosine/final`

Rationale:

- `sweep/analysis.md` explicitly recommends the MAE-trained AE for downstream Stage 1/2 work because of better interpolation behavior and better per-category MAE.

## Main Scripts

- Training entrypoint: `scripts/pv_simulator/train_stage1.py`
- Inference entrypoint: `scripts/pv_simulator/infer_stage1.py`
- Dataset loader updated for sharded MOVI input:
  `videox_fun/data/dataset_simulation.py`

## Files in This Folder

- `commands.md`: exact commands for setup, sanity checks, and full training
- `configs/`: copies of the intended sanity/full-run argument sets
- `logs/`: suggested location for stdout/stderr captures from sanity/full runs
- `notes.md`: assumptions, code changes, issues, and follow-up items
- `wandb.md`: W&B setup, offline/online mode, run naming, and sync instructions
- `run_summary.md`: current run state, latest observed progress, and remaining TODOs
- `patches/`: saved git diff and a short code-change summary for reproducibility

## Current Status

- Full Stage 1 training is running in tmux session `stage1_50k_padded_videox_gpu4_7`
- Runtime environment: `videox`
- Dataset: `datasets/movi_ab_50k_shards`
- AE checkpoint: `outputs/stage0/sweeps_mae/training/lr1e-3-cosine/final`
- Output directory: `outputs/stage1/stage1_50k_padded`
- W&B mode: `offline`
- W&B project/run: `pv-simulator-stage1` / `stage1_50k_padded_videox_gpu4_7`
- Visible GPUs for the active run: `4,7`
- Main log file:
  `experiments/stage1_50k/logs/stage1_50k_padded_videox_gpu4_7.log`

Concurrent run requested afterward:

- tmux session: `stage1_50k_padded_videox_gpu3_6_vis200`
- Output directory:
  `outputs/stage1/stage1_50k_padded_gpu3_6_vis200`
- W&B mode: `online`
- W&B project/run:
  `pv-simulator-stage1` / `stage1_50k_padded_videox_gpu3_6_vis200`
- W&B URL:
  `https://wandb.ai/raineggplant-tsinghua-university/pv-simulator-stage1/runs/vsum2f8u`
- Visible GPUs: `3,6`
- Visualization logging:
  `--vis_steps 200 --num_vis_samples 3`
- Current status: running

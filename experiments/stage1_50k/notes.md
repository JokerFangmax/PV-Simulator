# Notes

## Scope

- Do not modify or overwrite anything under `sweep/`.
- Do not move or delete files from `datasets/movi_ab_50k_shards`.
- Keep Stage 1 changes minimal and focused on enabling the new sharded MOVI dataset.

## Code change summary

- Added sharded MOVI (`webdataset`) support to `MoviSimulationDataset`.
- Preserved the existing `--dataset_type movi` Stage 1 interface so the new dataset can be passed via `--data_root datasets/movi_ab_50k_shards`.

## Sharded dataset format observed

- Root manifest:
  `datasets/movi_ab_50k_shards/webdataset/dataset_manifest.json`
- Sample manifest:
  `datasets/movi_ab_50k_shards/webdataset/manifest.jsonl`
- Each outer shard stores per-sample pairs:
  - `<sample_id>.manifest.json`
  - `<sample_id>.payload.tar`
- Each inner payload tar contains the usual MOVI files, including:
  - `metadata.json`
  - `point_cloud_states.pkl`
  - `events.json`

## Assumptions

- `max_objects=5` remains valid for the new 50k dataset because the current Stage 1 setup already assumes this cap.
- `max_points_per_object=200` remains compatible with the observed sample structure (`N=1000` for 5 objects in inspected samples).
- The MAE Stage 0 AE winner is the intended downstream checkpoint unless the user wants to compare against another AE later.

## Environment issue

- The existing conda env `pv` was present but initially contained only a minimal package set.
- The sanity work in this record was switched to `videox`, which already had `torch`, `numpy`, `accelerate`, and `diffusers`.
- If future work must use `pv`, it needs dependency installation first.

## Observed sanity-check results

- Dataset discovery: passed
  - `MoviSimulationDataset` found `50000` sharded samples under `datasets/movi_ab_50k_shards/webdataset`
- Sample read: passed
  - Sample 0 yielded the expected Stage 1 keys
  - Observed shapes: `x_s_raw=(21, 1000, 6)`, `c_force_raw=(21, 1000, 6)`, `c_id=(5,)`, `c_mat=(5, 2)`, `point_obj_idx=(1000,)`
- Padded batch: passed
  - Observed batch shapes: `x_s_raw=(2, 21, 1000, 6)`, `c_id=(2, 5)`, `point_mask=(2, 1000)`, `obj_mask=(2, 5)`
- Stage 1 inference pipeline import: passed
  - `from videox_fun.pipeline.pipeline_simulation import SimulationPipeline` succeeded in `videox`
- Stage 1 training startup: partial pass
  - `train_stage1.py` successfully loaded the AE checkpoint, created the 50k sharded dataset, built the dataloader, and entered the training loop
  - On this machine the one-step run did not complete promptly after `Steps: 0/1`
  - The script warns that kernel `5.4.0` is below the recommended minimum `5.5.0`, and this is a plausible cause of the observed stall

## TODO

- Confirm the dependency install in `pv` completed successfully.
- Re-run dataset and padded-batch sanity checks in `pv`.
- Run the one-step `train_stage1.py` startup check in `pv`.
- Optionally run a lightweight `pipeline_simulation` import smoke test after the environment is ready.

## 2026-05-07 Full-run update in `videox`

- The Stage 1 continuation work is now standardized on `videox`; `pv` is not used anymore for this run
- Built-in W&B support already existed in `scripts/pv_simulator/train_stage1.py` through:
  - `--report_to wandb`
  - `--wandb_project`
  - `--wandb_run_name`
- A small training-script change was still useful so W&B receives explicit `epoch` and `global_step` fields together with `train_loss` and `lr`
- `wandb` had to be installed into `videox`
- `python -m wandb status` showed `api_key: null`, so the run was launched with `WANDB_MODE=offline`
- The bare `wandb` shell entrypoint was not reliable on this machine because it resolved to a different user-level Python environment; `python -m wandb` in `videox` is the safe form
- `nohup` was not a reliable launch method in this agent/session context, so the persistent full run was launched in tmux instead
- The launch wrapper at `experiments/stage1_50k/logs/launch_full_stage1_50k_padded_videox.sh` intentionally uses `set -eo pipefail` rather than `set -euo pipefail`
  because `conda activate videox` triggered an `MKL_INTERFACE_LAYER` unbound-variable error with `set -u`

## Current full-run observations

- tmux session: `stage1_50k_padded_videox`
- Main log: `experiments/stage1_50k/logs/stage1_50k_padded_videox.log`
- Output dir: `outputs/stage1/stage1_50k_padded`
- Active offline W&B run:
  `wandb/offline-run-20260507_102455-8czrt9zp`
- Latest observed progress during this update:
  - step `24/100000`
  - `step_loss=1.68`
  - `lr=1.2e-5`
- No checkpoint files had been written yet at the time of inspection

## Updated TODO

- Monitor whether the run continues past the slow early steps on this kernel
- Sync the offline W&B run later if cloud tracking is needed
- Once the first checkpoint appears, record the checkpoint path for a concrete resume target
- If the host continues to show instability, rerun on a machine with kernel `>= 5.5`

## 2026-05-07 Relaunch on GPU 4 and 7

- The previous full run was deliberately stopped on user request after reaching step `24/100000`
- GPU checks showed GPU `4` and GPU `7` were idle, while the active Stage 1 run was still on GPU `0`
- The relaunch uses `CUDA_VISIBLE_DEVICES=4,7` plus `python -m accelerate.commands.launch --multi_gpu --num_processes 2`
- The bare `accelerate` executable is not safe on this machine because it resolves to a user-level Python without `torch`; the module form in `videox` is required
- Per-process `train_batch_size` was reduced from `8` to `4` so the effective global batch stays at `8` across 2 processes
- Output directory remains `outputs/stage1/stage1_50k_padded`
- New log/run identity for the relaunch:
  - tmux session: `stage1_50k_padded_videox_gpu4_7`
  - log file: `experiments/stage1_50k/logs/stage1_50k_padded_videox_gpu4_7.log`
  - W&B run name: `stage1_50k_padded_videox_gpu4_7`
- Verified after relaunch:
  - rank `0` and rank `1` both initialized under NCCL
  - the two `videox` worker processes landed on physical GPU `4` and GPU `7`
  - W&B offline run directory:
    `wandb/offline-run-20260507_105930-a0s31117`
  - the run reached step `7/100000`
  - latest observed `step_loss=1.53`, `lr=3.5e-6`
- `accelerate launch` printed a default-warning line about launcher `--mixed_precision`; this is benign for the current run because the training script itself reports `Mixed precision type: bf16`

## 2026-05-07 Concurrent visualization run on GPU 3 and 6

- User requested a second Stage 1 run without stopping the active GPU `4,7` job
- To avoid checkpoint collisions, this concurrent run uses a separate output directory:
  `outputs/stage1/stage1_50k_padded_gpu3_6_vis200`
- Requested visualization settings were enabled:
  - `--vis_steps 200`
  - `--num_vis_samples 3`
- The run is intended to use online W&B
- The raw W&B API key provided by the user is not stored anywhere in the repo
- The secure relaunch uses a temporary file in `/tmp`, read and deleted by the wrapper before the training process starts
- GPU `3` and GPU `6` were not empty at launch time, but both had enough free memory margin for another Stage 1 copy
- Verified after secure relaunch:
  - tmux session: `stage1_50k_padded_videox_gpu3_6_vis200`
  - log file:
    `experiments/stage1_50k/logs/stage1_50k_padded_videox_gpu3_6_vis200.log`
  - output directory:
    `outputs/stage1/stage1_50k_padded_gpu3_6_vis200`
  - the temporary key file `/tmp/wandb_stage1_gpu3_6.key` was deleted after wrapper startup
  - online W&B run id: `vsum2f8u`
  - online W&B URL:
    `https://wandb.ai/raineggplant-tsinghua-university/pv-simulator-stage1/runs/vsum2f8u`
  - first observed progress:
    - step `1/100000`
    - `step_loss=1.53`
    - `lr=5e-7`

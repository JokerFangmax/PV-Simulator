# W&B Record

## Status

- Environment: `videox`
- SDK availability: installed and importable
- Checked with:
  `source /data/fhr/miniconda3/etc/profile.d/conda.sh && conda activate videox && python -m wandb status`
- Machine login state during launch: no API key configured (`api_key: null`)
- Effective mode for this Stage 1 run: `offline`

## Run Identity

- Project: `pv-simulator-stage1`
- Run name: `stage1_50k_padded_videox`
- Offline run directory:
  `wandb/offline-run-20260507_102455-8czrt9zp`

## Built-in support used

Stage 1 already had native tracker wiring through `accelerate`:

- `--report_to wandb`
- `--wandb_project`
- `--wandb_run_name`

No refactor was needed. The only Stage 1 W&B code tweak was adding explicit
`epoch` and `global_step` fields to the existing `accelerator.log(...)` call.

## Logged information

Configured through CLI / tracker config:

- project name
- run name
- dataset type
- dataset path
- AE checkpoint path
- output directory
- padded-batch settings
- batch size
- learning rate
- mixed precision

Logged during training:

- `train_loss`
- `lr`
- `epoch`
- `global_step`

Runtime paths visible in logs:

- training log:
  `experiments/stage1_50k/logs/stage1_50k_padded_videox.log`
- output dir:
  `outputs/stage1/stage1_50k_padded`

## Sync later

Use the `videox` environment and the module form of the CLI:

```bash
source /data/fhr/miniconda3/etc/profile.d/conda.sh && conda activate videox && python -m wandb sync wandb/offline-run-20260507_102455-8czrt9zp
```

If a future machine has a configured W&B login and online syncing is desired,
remove `WANDB_MODE=offline` from the launch environment before starting a new run.

## GPU 4 and 7 relaunch

- Previous offline run:
  `wandb/offline-run-20260507_102455-8czrt9zp`
- Current active offline run after the GPU `4,7` relaunch:
  `wandb/offline-run-20260507_105930-a0s31117`
- Current active run name:
  `stage1_50k_padded_videox_gpu4_7`

Sync the active GPU `4,7` run later with:

```bash
source /data/fhr/miniconda3/etc/profile.d/conda.sh && conda activate videox && python -m wandb sync wandb/offline-run-20260507_105930-a0s31117
```

## GPU 3 and 6 online visualization run

- Active online run name:
  `stage1_50k_padded_videox_gpu3_6_vis200`
- Active online run id:
  `vsum2f8u`
- Project URL:
  `https://wandb.ai/raineggplant-tsinghua-university/pv-simulator-stage1`
- Run URL:
  `https://wandb.ai/raineggplant-tsinghua-university/pv-simulator-stage1/runs/vsum2f8u`
- Local run directory:
  `wandb/run-20260507_121141-vsum2f8u`
- Visualization settings:
  - `vis_steps=200`
  - `num_vis_samples=3`
- Secret handling:
  - the raw API key is not stored in this repo
  - the secure launch used a temporary `/tmp` key file that the wrapper deleted before training continued

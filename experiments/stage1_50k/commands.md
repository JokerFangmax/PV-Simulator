# Commands

## 1. Environment inspection

```bash
conda env list
conda run -n pv python -c "import sys, site; print(sys.executable); print(sys.version); print(site.getsitepackages())"
conda run -n pv pip list
source /data/fhr/miniconda3/etc/profile.d/conda.sh && conda activate videox && python -c "import torch, numpy, accelerate, diffusers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('numpy', numpy.__version__); print('accelerate', accelerate.__version__); print('diffusers', diffusers.__version__)"
```

## 2. Optional dependency install for `pv`

This was needed because the existing `pv` env did not contain the project runtime packages.

```bash
conda run -n pv python -m pip install -r requirements.txt tqdm matplotlib wandb
```

## 3. Dataset sanity check

```bash
source /data/fhr/miniconda3/etc/profile.d/conda.sh && conda activate videox && python -c "from videox_fun.data.dataset_simulation import MoviSimulationDataset; ds=MoviSimulationDataset('datasets/movi_ab_50k_shards', max_objects=5); s=ds[0]; print('len', len(ds)); print('keys', sorted(s.keys())); print('x_s_raw', tuple(s['x_s_raw'].shape)); print('c_mat', tuple(s['c_mat'].shape)); print('point_obj_idx', tuple(s['point_obj_idx'].shape))"
```

## 4. Padded-batch sanity check

```bash
source /data/fhr/miniconda3/etc/profile.d/conda.sh && conda activate videox && python -c "from functools import partial; from torch.utils.data import DataLoader; from videox_fun.data.dataset_simulation import MoviSimulationDataset, sim_collate_fn_padded; ds=MoviSimulationDataset('datasets/movi_ab_50k_shards', max_objects=5); dl=DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=partial(sim_collate_fn_padded, max_T_raw=21, max_objects=5, max_points_per_object=200)); b=next(iter(dl)); print('batch_x_s_raw', tuple(b['x_s_raw'].shape)); print('batch_c_id', tuple(b['c_id'].shape)); print('batch_point_mask', tuple(b['point_mask'].shape)); print('batch_T_raw', tuple(b['T_raw'].shape), b['T_raw'].tolist()); print('batch_N', tuple(b['N'].shape), b['N'].tolist())"
```

## 5. Stage 1 training startup sanity check

```bash
source /data/fhr/miniconda3/etc/profile.d/conda.sh && conda activate videox && python scripts/pv_simulator/train_stage1.py \
  --dataset_type movi \
  --data_root datasets/movi_ab_50k_shards \
  --ae_ckpt_dir outputs/stage0/sweeps_mae/training/lr1e-3-cosine/final \
  --output_dir experiments/stage1_50k/logs/sanity_train_start \
  --padded_batch \
  --max_objects 5 \
  --max_T_raw 21 \
  --max_points_per_object 200 \
  --train_batch_size 2 \
  --dataloader_num_workers 0 \
  --num_train_epochs 1 \
  --max_train_steps 1 \
  --learning_rate 1e-4 \
  --lr_warmup_steps 1 \
  --mixed_precision no \
  --vis_steps 0 \
  --checkpointing_steps 1000 \
  --report_to tensorboard
```

Observed result:

- The script initializes successfully, loads the MAE AE checkpoint, constructs the 50k sharded dataset, and enters the training loop.
- On this machine it may stall after `Steps: 0/1`, consistent with the kernel warning emitted by `accelerate` (`kernel 5.4.0` below the recommended minimum `5.5.0`).

## 6. Recommended full Stage 1 run

```bash
source /data/fhr/miniconda3/etc/profile.d/conda.sh && conda activate videox && python scripts/pv_simulator/train_stage1.py \
  --dataset_type movi \
  --data_root datasets/movi_ab_50k_shards \
  --ae_ckpt_dir outputs/stage0/sweeps_mae/training/lr1e-3-cosine/final \
  --output_dir outputs/stage1/stage1_50k_padded \
  --padded_batch \
  --max_objects 5 \
  --max_T_raw 21 \
  --max_points_per_object 200 \
  --num_train_epochs 1000 \
  --max_train_steps 100000 \
  --lr_warmup_steps 200 \
  --train_batch_size 8 \
  --learning_rate 1e-4 \
  --mixed_precision bf16 \
  --report_to tensorboard \
  --vis_steps 0
```

## 7. Optional inference import smoke test

```bash
source /data/fhr/miniconda3/etc/profile.d/conda.sh && conda activate videox && python -c "from videox_fun.pipeline.pipeline_simulation import SimulationPipeline; print('pipeline_import_ok', SimulationPipeline.__name__)"
```

## 8. W&B setup in `videox`

```bash
source /data/fhr/miniconda3/etc/profile.d/conda.sh && conda activate videox && python -m pip install wandb
source /data/fhr/miniconda3/etc/profile.d/conda.sh && conda activate videox && python -m wandb status
```

Observed result:

- `wandb` is importable in `videox`
- No API key was configured on this machine, so Stage 1 was launched with `WANDB_MODE=offline`
- Use `python -m wandb ...` from the activated `videox` env instead of the bare `wandb` shell entrypoint

## 9. Stage 1 startup check with offline W&B

```bash
source /data/fhr/miniconda3/etc/profile.d/conda.sh && conda activate videox && WANDB_MODE=offline python scripts/pv_simulator/train_stage1.py \
  --dataset_type movi \
  --data_root datasets/movi_ab_50k_shards \
  --ae_ckpt_dir outputs/stage0/sweeps_mae/training/lr1e-3-cosine/final \
  --output_dir experiments/stage1_50k/logs/startup_wandb_offline \
  --padded_batch \
  --max_objects 5 \
  --max_T_raw 21 \
  --max_points_per_object 200 \
  --train_batch_size 2 \
  --dataloader_num_workers 0 \
  --num_train_epochs 1 \
  --max_train_steps 1 \
  --learning_rate 1e-4 \
  --lr_warmup_steps 1 \
  --mixed_precision no \
  --vis_steps 0 \
  --checkpointing_steps 1000 \
  --report_to wandb \
  --wandb_project pv-simulator-stage1 \
  --wandb_run_name stage1_50k_padded_videox_startup
```

Observed result:

- Dataset loading, model initialization, and W&B initialization all succeeded in `videox`
- Offline W&B run data was written locally under `wandb/`
- The script reached the first training step without immediate startup errors

## 10. Full Stage 1 launch script and persistent tmux launch

Wrapper script:

```bash
bash experiments/stage1_50k/logs/launch_full_stage1_50k_padded_videox.sh
```

Persistent launch used for the actual run:

```bash
tmux new-session -d -s stage1_50k_padded_videox 'cd /data/fhr/projects/PV-Simulator && bash experiments/stage1_50k/logs/launch_full_stage1_50k_padded_videox.sh > experiments/stage1_50k/logs/stage1_50k_padded_videox.log 2>&1'
```

Inner training command executed by the wrapper:

```bash
source /data/fhr/miniconda3/etc/profile.d/conda.sh && conda activate videox && export WANDB_MODE=offline && python -u scripts/pv_simulator/train_stage1.py \
  --dataset_type movi \
  --data_root datasets/movi_ab_50k_shards \
  --ae_ckpt_dir outputs/stage0/sweeps_mae/training/lr1e-3-cosine/final \
  --output_dir outputs/stage1/stage1_50k_padded \
  --padded_batch \
  --max_objects 5 \
  --max_T_raw 21 \
  --max_points_per_object 200 \
  --num_train_epochs 1000 \
  --max_train_steps 100000 \
  --lr_warmup_steps 200 \
  --train_batch_size 8 \
  --learning_rate 1e-4 \
  --mixed_precision bf16 \
  --dataloader_num_workers 0 \
  --report_to wandb \
  --wandb_project pv-simulator-stage1 \
  --wandb_run_name stage1_50k_padded_videox \
  --vis_steps 0
```

## 11. Run monitoring

```bash
tmux list-sessions
tmux attach -t stage1_50k_padded_videox
tmux capture-pane -t stage1_50k_padded_videox -p | tail -n 60
tail -f experiments/stage1_50k/logs/stage1_50k_padded_videox.log
```

## 12. Resume and offline sync

Resume template:

```bash
tmux new-session -d -s stage1_50k_padded_videox_resume 'cd /data/fhr/projects/PV-Simulator && source /data/fhr/miniconda3/etc/profile.d/conda.sh && conda activate videox && export WANDB_MODE=offline && python -u scripts/pv_simulator/train_stage1.py --dataset_type movi --data_root datasets/movi_ab_50k_shards --ae_ckpt_dir outputs/stage0/sweeps_mae/training/lr1e-3-cosine/final --output_dir outputs/stage1/stage1_50k_padded --padded_batch --max_objects 5 --max_T_raw 21 --max_points_per_object 200 --num_train_epochs 1000 --max_train_steps 100000 --lr_warmup_steps 200 --train_batch_size 8 --learning_rate 1e-4 --mixed_precision bf16 --dataloader_num_workers 0 --report_to wandb --wandb_project pv-simulator-stage1 --wandb_run_name stage1_50k_padded_videox --vis_steps 0 --resume_from_checkpoint outputs/stage1/stage1_50k_padded/checkpoint-<N> > experiments/stage1_50k/logs/stage1_50k_padded_videox_resume.log 2>&1'
```

Offline sync command for the active full run:

```bash
source /data/fhr/miniconda3/etc/profile.d/conda.sh && conda activate videox && python -m wandb sync wandb/offline-run-20260507_102455-8czrt9zp
```

## 13. Kill the single-GPU run and relaunch on GPU 4 and 7

Availability check used before relaunch:

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader
```

Observed result at relaunch time:

- GPU `4` and GPU `7` were idle
- The active Stage 1 training process was still running on GPU `0`

Stop the previous tmux-backed run:

```bash
tmux kill-session -t stage1_50k_padded_videox
kill 269893
```

Relaunch wrapper:

```bash
bash experiments/stage1_50k/logs/launch_full_stage1_50k_padded_videox_gpu4_7.sh
```

Persistent relaunch used:

```bash
tmux new-session -d -s stage1_50k_padded_videox_gpu4_7 'cd /data/fhr/projects/PV-Simulator && bash experiments/stage1_50k/logs/launch_full_stage1_50k_padded_videox_gpu4_7.sh > experiments/stage1_50k/logs/stage1_50k_padded_videox_gpu4_7.log 2>&1'
```

Inner command executed by the wrapper:

```bash
source /data/fhr/miniconda3/etc/profile.d/conda.sh && conda activate videox && export WANDB_MODE=offline && export CUDA_VISIBLE_DEVICES=4,7 && python -m accelerate.commands.launch --multi_gpu --num_processes 2 --main_process_port 29607 scripts/pv_simulator/train_stage1.py --dataset_type movi --data_root datasets/movi_ab_50k_shards --ae_ckpt_dir outputs/stage0/sweeps_mae/training/lr1e-3-cosine/final --output_dir outputs/stage1/stage1_50k_padded --padded_batch --max_objects 5 --max_T_raw 21 --max_points_per_object 200 --num_train_epochs 1000 --max_train_steps 100000 --lr_warmup_steps 200 --train_batch_size 4 --learning_rate 1e-4 --mixed_precision bf16 --dataloader_num_workers 0 --report_to wandb --wandb_project pv-simulator-stage1 --wandb_run_name stage1_50k_padded_videox_gpu4_7 --vis_steps 0
```

Monitoring:

```bash
tmux attach -t stage1_50k_padded_videox_gpu4_7
tmux capture-pane -t stage1_50k_padded_videox_gpu4_7 -p | tail -n 80
tail -f experiments/stage1_50k/logs/stage1_50k_padded_videox_gpu4_7.log
```

## 14. Concurrent online W&B run on GPU 3 and 6 with visualization

Live checks before the new run:

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader
```

Observed result:

- GPU `3` and GPU `6` were not idle, but each had only ~3 GiB already allocated
- The existing Stage 1 run on GPU `4,7` was left untouched

New wrapper:

```bash
bash experiments/stage1_50k/logs/launch_full_stage1_50k_padded_videox_gpu3_6_vis200.sh
```

Persistent launch used:

```bash
tmux new-session -d -s stage1_50k_padded_videox_gpu3_6_vis200 'cd /data/fhr/projects/PV-Simulator && export WANDB_API_KEY_FILE=/tmp/wandb_stage1_gpu3_6.key && bash experiments/stage1_50k/logs/launch_full_stage1_50k_padded_videox_gpu3_6_vis200.sh > experiments/stage1_50k/logs/stage1_50k_padded_videox_gpu3_6_vis200.log 2>&1'
```

Inner command executed by the wrapper:

```bash
source /data/fhr/miniconda3/etc/profile.d/conda.sh && conda activate videox && export CUDA_VISIBLE_DEVICES=3,6 && export WANDB_MODE=online && python -m accelerate.commands.launch --multi_gpu --num_processes 2 --main_process_port 29636 scripts/pv_simulator/train_stage1.py --dataset_type movi --data_root datasets/movi_ab_50k_shards --ae_ckpt_dir outputs/stage0/sweeps_mae/training/lr1e-3-cosine/final --output_dir outputs/stage1/stage1_50k_padded_gpu3_6_vis200 --padded_batch --max_objects 5 --max_T_raw 21 --max_points_per_object 200 --num_train_epochs 1000 --max_train_steps 100000 --lr_warmup_steps 200 --train_batch_size 4 --learning_rate 1e-4 --mixed_precision bf16 --dataloader_num_workers 0 --report_to wandb --wandb_project pv-simulator-stage1 --wandb_run_name stage1_50k_padded_videox_gpu3_6_vis200 --vis_steps 200 --num_vis_samples 3
```

Monitoring:

```bash
tmux attach -t stage1_50k_padded_videox_gpu3_6_vis200
tmux capture-pane -t stage1_50k_padded_videox_gpu3_6_vis200 -p | tail -n 120
tail -f experiments/stage1_50k/logs/stage1_50k_padded_videox_gpu3_6_vis200.log
```

Observed startup result:

- rank `0` and rank `1` initialized successfully under NCCL
- the 3 fixed visualization samples loaded successfully
- online W&B tracking started
- first training step completed

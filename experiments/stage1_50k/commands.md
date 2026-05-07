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

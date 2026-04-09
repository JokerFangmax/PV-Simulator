```shell
# accelerate launch --num_processes=1 scripts/pv_simulator/train_stage1.py \
python scripts/pv_simulator/train_stage1.py \
    --dataset_type movi \
    --data_root datasets/movi_ab_10k \
    --output_dir outputs/stage1_padded \
    --padded_batch \
    --max_objects 5 \
    --max_T_raw 21 \
    --max_points_per_object 200 \
    --num_train_epochs 1000 \
    --max_train_steps 100000 \
    --lr_warmup_steps 200 \
    --train_batch_size 8 \
    --mixed_precision bf16 \
    --report_to wandb \
    --wandb_run_name stage1_padded \
    --vis_steps 100 \
    --num_vis_samples 3


python scripts/pv_simulator/train_stage1.py \
    --dataset_type movi \
    --data_root datasets/movi_ab_10k \
    --output_dir outputs/stage1 \
    --max_objects 5 \
    --num_train_epochs 1000 \
    --max_train_steps 100000 \
    --lr_warmup_steps 200 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --mixed_precision bf16 \
    --report_to wandb \
    --wandb_run_name stage1 \
    --vis_steps 100 \
    --num_vis_samples 3
```

lightweight model:
```shell
python scripts/pv_simulator/train_stage1.py \
    --dataset_type movi \
    --data_root datasets/movi_ab_10k \
    --output_dir outputs/stage1_padded-light \
    --padded_batch \
    --sim_num_layers 4 \
    --d_state 128 \
    --d_sim 256 \
    --sim_ffn_dim 512 \
    --sim_num_heads 4 \
    --max_objects 5 \
    --max_T_raw 21 \
    --max_points_per_object 200 \
    --num_train_epochs 1000 \
    --max_train_steps 100000 \
    --lr_warmup_steps 200 \
    --train_batch_size 16 \
    --mixed_precision bf16 \
    --report_to wandb \
    --wandb_run_name stage1_padded-light \
    --vis_steps 200 \
    --num_vis_samples 3


python scripts/pv_simulator/train_stage1.py \
    --dataset_type movi \
    --data_root datasets/movi_ab_10k \
    --output_dir outputs/stage1-light \
    --sim_num_layers 4 \
    --d_state 128 \
    --d_sim 256 \
    --sim_ffn_dim 512 \
    --sim_num_heads 4 \
    --max_objects 5 \
    --max_T_raw 21 \
    --max_points_per_object 200 \
    --num_train_epochs 1000 \
    --max_train_steps 100000 \
    --lr_warmup_steps 200 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --mixed_precision bf16 \
    --report_to wandb \
    --wandb_run_name stage1-light \
    --vis_steps 200 \
    --num_vis_samples 3
```

# Sweep fixes / crash log

Crashes from the orchestrator are appended below with a tail of the train log.
Code edits applied by Claude to recover failed runs are also logged here.

## 2026-04-12 10:15:15  Phase E  run mae-res3 — CRASH
- category: stage_e
- cfg: {"max_train_steps": 5000, "val_every": 1000, "checkpointing_steps": 5000, "batch_size": 256, "seed": 42, "mixed_precision": "bf16", "lambda_smooth": 0.0, "lr_total_steps": 5000, "lr_warmup_steps": 200, "lambda_mmd": 0.1, "lambda_interp": 0.1, "c_mid": 128, "d_latent": 16, "n_res_blocks": 3, "lr": 0.001, "lr_scheduler": "cosine", "loss_type": "mae"}
- message: exit=1, retrying (1/2)
- tail of train.log:
```
    x = block(x, feat_cache, feat_idx)
  File "/data/szy/miniconda3/envs/asr/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data/szy/miniconda3/envs/asr/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data/szy/projects/phys_video/PV-Simulator/videox_fun/models/sim_causal_encoder.py", line 98, in forward
    x = layer(x, feat_cache[idx])
  File "/data/szy/miniconda3/envs/asr/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data/szy/miniconda3/envs/asr/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data/szy/projects/phys_video/PV-Simulator/videox_fun/models/sim_causal_encoder.py", line 42, in forward
    return super().forward(x)
  File "/data/szy/miniconda3/envs/asr/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 375, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/data/szy/miniconda3/envs/asr/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 370, in _conv_forward
    return F.conv1d(
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 24.00 MiB. GPU 0 has a total capacity of 23.57 GiB of which 20.81 MiB is free. Including non-PyTorch memory, this process has 23.54 GiB memory in use. Of the allocated memory 22.64 GiB is allocated by PyTorch, and 596.00 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
Traceback (most recent call last):
  File "/data/szy/projects/phys_video/PV-Simulator/scripts/pv_simulator/train_stage0.py", line 994, in <module>
    main()
  File "/data/szy/projects/phys_video/PV-Simulator/scripts/pv_simulator/train_stage0.py", line 918, in main
    _, z_interp = ae(x_interp)
  File "/data/szy/miniconda3/envs/asr/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data/szy/miniconda3/envs/asr/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data/szy/miniconda3/envs/asr/lib/python3.10/site-packages/accelerate/utils/operations.py", line 823, in forward
    return model_forward(*args, **kwargs)
  File "/data/szy/miniconda3/envs/asr/lib/python3.10/site-packages/accelerate/utils/operations.py", line 811, in __call__
    return convert_to_fp32(self.model_forward(*args, **kwargs))
  File "/data/szy/miniconda3/envs/asr/lib/python3.10/site-packages/torch/amp/autocast_mode.py", line 44, in decorate_autocast
    return func(*args, **kwargs)
  File "/data/szy/projects/phys_video/PV-Simulator/videox_fun/models/sim_ae.py", line 381, in forward
    x_hat = self.decode(z, t_raw)
  File "/data/szy/projects/phys_video/PV-Simulator/videox_fun/models/sim_ae.py", line 354, in decode
    out_ = self.decoder(
  File "/data/szy/miniconda3/envs/asr/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data/szy/miniconda3/envs/asr/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data/szy/projects/phys_video/PV-Simulator/videox_fun/models/sim_ae.py", line 223, in forward
    x = block(x, feat_cache, feat_idx)
  File "/data/szy/miniconda3/envs/asr/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data/szy/miniconda3/envs/asr/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data/szy/projects/phys_video/PV-Simulator/videox_fun/models/sim_causal_encoder.py", line 98, in forward
    x = layer(x, feat_cache[idx])
  File "/data/szy/miniconda3/envs/asr/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data/szy/miniconda3/envs/asr/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data/szy/projects/phys_video/PV-Simulator/videox_fun/models/sim_causal_encoder.py", line 42, in forward
    return super().forward(x)
  File "/data/szy/miniconda3/envs/asr/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 375, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/data/szy/miniconda3/envs/asr/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 370, in _conv_forward
    return F.conv1d(
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 24.00 MiB. GPU 0 has a total capacity of 23.57 GiB of which 20.81 MiB is free. Including non-PyTorch memory, this process has 23.54 GiB memory in use. Of the allocated memory 22.64 GiB is allocated by PyTorch, and 596.00 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

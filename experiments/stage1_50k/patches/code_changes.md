# Code Change Summary

This folder stores reproducibility notes for the minimal code changes needed
to run Stage 1 on the 50k sharded MOVI dataset with W&B tracking in `videox`.

## Current live git diff

- `git_diff_stage1_50k_videox_wandb.patch`
- At the time of this run, the live uncommitted diff contains the Stage 1
  logging change in `scripts/pv_simulator/train_stage1.py`

## Relevant Stage 1 code adjustments

- `videox_fun/data/dataset_simulation.py`
  - Added support for the new sharded MOVI layout under
    `datasets/movi_ab_50k_shards/webdataset`
  - Keeps the existing `--dataset_type movi` interface unchanged
- `videox_fun/models/__init__.py`
  - Made `TurboWanTransformer3DModel` import optional so unrelated missing
    dependencies in `videox` do not block Stage 1 imports
- `scripts/pv_simulator/train_stage1.py`
  - Preserved built-in W&B support
  - Added explicit `epoch` and `global_step` keys to the tracker log payload

## Notes

- The dataset-loader and optional-import adjustments were prepared during the
  earlier Stage 1 50k setup pass
- The live git diff file should be treated as the authoritative patch for any
  still-uncommitted code at the time this run was launched

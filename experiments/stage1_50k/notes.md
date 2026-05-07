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

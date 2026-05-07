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
- If `pv` is still desired later, it needs dependency provisioning first

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

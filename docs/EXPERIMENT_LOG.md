# Experiment Log

This log records each modification and experiment using the required project
format.

## Step 001: Bootstrap project documentation

Date: 2026-06-27

Goal: Create project context and experiment tracking documents for
diagnosis-first Stage 1 stabilization.

Hypothesis: Clear documentation and a required logging protocol will prevent
premature architecture changes and keep work focused on reproducible Stage 1
diagnostics.

Files changed:
- docs/PV_SIMULATOR_CONTEXT.md
- docs/EXPERIMENT_ROADMAP.md
- docs/EXPERIMENT_LOG.md
- docs/DECISION_LOG.md
- docs/CODEX_PROTOCOL.md

Commands run:
- ls docs
- git status --short
- test -f docs/PV_SIMULATOR_CONTEXT.md
- test -f docs/EXPERIMENT_ROADMAP.md
- test -f docs/EXPERIMENT_LOG.md
- test -f docs/DECISION_LOG.md
- test -f docs/CODEX_PROTOCOL.md
- rg -n "Step 001|Phase 0|Stage 1|Do not assume|PBD|XPBD" docs
- git diff -- docs

Results: Documentation files were planned after inspecting the existing docs
layout. The target files did not exist before this step. After creation, all
five file-existence checks passed, and `rg` found the expected documentation
markers. `git diff -- docs` produced no output because the `docs/` directory is
untracked in this worktree.

Pass/Fail: Pass

Problems found: `README.md` already had unrelated local modifications before
this step. The existing `docs/` directory is currently untracked as a whole in
`git status`, including the pre-existing `docs/METHOD.md`.

Next action: Run the requested documentation smoke checks and then begin Phase 0
sanity-check implementation in a later task.

## Step 002: Add Phase 0 Stage 1 sanity diagnostics

Date: 2026-06-27

Goal: Implement diagnosis-first Phase 0 tooling for Stage 1 without modifying
SimTransformer architecture, losses, Rigid-Residual, Contact-Aware modules,
PBD, XPBD, Warp, or Stage 2 code.

Hypothesis: A standalone diagnostic runner can verify the current Stage 1 data,
AE, baseline, and one-batch training paths before any architecture changes are
considered.

Files changed:
- scripts/pv_simulator/diagnostics/run_phase0_sanity.py
- docs/EXPERIMENT_LOG.md

Commands run:
- sed -n '1,260p' scripts/pv_simulator/train_stage1.py
- sed -n '260,620p' scripts/pv_simulator/train_stage1.py
- sed -n '1,260p' scripts/pv_simulator/infer_stage1.py
- sed -n '1,300p' videox_fun/data/dataset_simulation.py
- sed -n '300,760p' videox_fun/data/dataset_simulation.py
- sed -n '1,320p' videox_fun/models/sim_ae.py
- sed -n '320,720p' videox_fun/models/sim_ae.py
- sed -n '1,460p' videox_fun/models/sim_transformer.py
- sed -n '460,700p' videox_fun/models/sim_transformer.py
- sed -n '1,360p' videox_fun/pipeline/pipeline_simulation.py
- sed -n '1,260p' videox_fun/models/sim_condition.py
- sed -n '1,320p' videox_fun/utils/sim_metrics.py
- rg --files scripts/pv_simulator videox_fun | rg 'visual|plot|diagnose|metric'
- python -m py_compile scripts/pv_simulator/diagnostics/run_phase0_sanity.py
- conda run -n videox python -m py_compile scripts/pv_simulator/diagnostics/run_phase0_sanity.py
- conda run -n videox python scripts/pv_simulator/diagnostics/run_phase0_sanity.py --help
- conda run -n videox python scripts/pv_simulator/diagnostics/run_phase0_sanity.py --dataset_type movi --data_root datasets/movi_ab_50k_shards --output_root /tmp/pv_phase0_smoke --run_name smoke_phase0_final --padded_batch --max_objects 5 --max_T_raw 21 --max_points_per_object 200 --modes audit baselines --sample_idx 0 --num_samples 1 --batch_size 1 --device cpu --max_chamfer_points 32
- conda run -n videox python scripts/pv_simulator/diagnostics/run_phase0_sanity.py --dataset_type movi --data_root datasets/movi_ab_50k_shards --ae_ckpt_dir outputs/stage0/sweeps_mae/training/lr1e-3-cosine/final --output_root /tmp/pv_phase0_smoke --run_name smoke_phase0_ae --padded_batch --max_objects 5 --max_T_raw 21 --max_points_per_object 200 --modes ae --sample_idx 0 --num_samples 1 --batch_size 1 --device cpu --max_chamfer_points 32
- conda run -n videox python scripts/pv_simulator/diagnostics/run_phase0_sanity.py --dataset_type movi --data_root datasets/movi_ab_50k_shards --ae_ckpt_dir outputs/stage0/sweeps_mae/training/lr1e-3-cosine/final --output_root /tmp/pv_phase0_smoke --run_name smoke_phase0_overfit --padded_batch --max_objects 5 --max_T_raw 5 --max_points_per_object 2 --modes deterministic_overfit --sample_idx 0 --num_samples 1 --batch_size 1 --device cpu --overfit_steps 1 --d_sim 32 --sim_ffn_dim 64 --sim_num_heads 4 --sim_num_layers 1 --log_every 1 --max_chamfer_points 16
- conda run -n videox python scripts/pv_simulator/diagnostics/run_phase0_sanity.py --dataset_type movi --data_root datasets/movi_ab_50k_shards --ae_ckpt_dir outputs/stage0/sweeps_mae/training/lr1e-3-cosine/final --output_root /tmp/pv_phase0_smoke --run_name smoke_phase0_stochastic --padded_batch --max_objects 5 --max_T_raw 5 --max_points_per_object 2 --modes stochastic_overfit --sample_idx 0 --num_samples 1 --batch_size 1 --device cpu --overfit_steps 1 --d_sim 32 --sim_ffn_dim 64 --sim_num_heads 4 --sim_num_layers 1 --log_every 1 --max_chamfer_points 16
- find /tmp/pv_phase0_smoke/smoke_phase0_final -maxdepth 1 -type f -print
- git status --short

Results: Added `run_phase0_sanity.py` with modes for AE reconstruction,
deterministic one-batch overfit, stochastic one-batch overfit, naive baselines,
and data loader / coordinate / mask audit. The script writes
`metrics.json`, `metrics.csv`, `audit.txt`, and `config.json` under a default
timestamped `experiments/stage1_diagnostics/<YYYYMMDD_HHMMSS>_phase0/`
directory, with optional npz snapshots and visualization videos. Smoke checks
passed in the `videox` environment. The default `python` environment lacks
`torch`, so direct `python ... --help` failed before rerunning successfully with
`conda run -n videox`.

Pass/Fail: Pass

Problems found: CPU overfit smoke emits a warning from the existing
SimTransformer autocast path because it requests CUDA autocast while CUDA is not
available; PyTorch disables it and the diagnostic still completes. Sample 0 of
`datasets/movi_ab_50k_shards` had zero `c_force_raw`, so contact-window metrics
are reported as unavailable for that sample.

Next action: Run full Phase 0 diagnostics on the intended GPU/runtime:

```bash
conda run -n videox python scripts/pv_simulator/diagnostics/run_phase0_sanity.py \
  --dataset_type movi \
  --data_root datasets/movi_ab_50k_shards \
  --ae_ckpt_dir outputs/stage0/sweeps_mae/training/lr1e-3-cosine/final \
  --padded_batch \
  --max_objects 5 \
  --max_T_raw 21 \
  --max_points_per_object 200 \
  --modes all \
  --sample_idx 0 \
  --num_samples 1 \
  --batch_size 1 \
  --device cuda \
  --dtype bf16 \
  --overfit_steps 100 \
  --save_npz_snapshots
```

Expected outputs: `metrics.json`, `metrics.csv`, `audit.txt`, and
`config.json` in `experiments/stage1_diagnostics/<YYYYMMDD_HHMMSS>_phase0/`,
plus optional `.npz` snapshots and visualization videos when requested.

## Step 003: Add Stage 1 six-case diagnostic tooling

Date: 2026-06-27

Goal: Implement Stage 1 six-case diagnostic tooling to evaluate the current
Simulation Branch on representative physical scenarios before adding
Contact-Aware Sampling, Rigid-Residual Representation, Data Augmentation,
Violation Feedback, PBD, XPBD, Warp, or Stage 2 video demos.

Hypothesis: Heuristic case selection plus consistent per-case metrics,
snapshots, and visualizations will expose which physical regimes the current
Stage 1 pipeline handles or fails without requiring architecture changes.

Files changed:
- scripts/pv_simulator/diagnostics/audit_dataset_distribution.py
- scripts/pv_simulator/diagnostics/run_stage1_cases.py
- scripts/pv_simulator/diagnostics/visualize_stage1_predictions.py
- docs/EXPERIMENT_LOG.md

Commands run:
- sed -n '1,260p' scripts/pv_simulator/diagnostics/run_phase0_sanity.py
- sed -n '260,620p' scripts/pv_simulator/diagnostics/run_phase0_sanity.py
- sed -n '220,520p' scripts/pv_simulator/infer_stage1.py
- sed -n '240,560p' scripts/pv_simulator/visualize.py
- conda run -n videox python -m py_compile scripts/pv_simulator/diagnostics/audit_dataset_distribution.py
- conda run -n videox python -m py_compile scripts/pv_simulator/diagnostics/visualize_stage1_predictions.py
- conda run -n videox python -m py_compile scripts/pv_simulator/diagnostics/run_stage1_cases.py
- conda run -n videox python scripts/pv_simulator/diagnostics/audit_dataset_distribution.py --help
- conda run -n videox python scripts/pv_simulator/diagnostics/visualize_stage1_predictions.py --help
- conda run -n videox python scripts/pv_simulator/diagnostics/run_stage1_cases.py --help
- conda run -n videox python scripts/pv_simulator/diagnostics/audit_dataset_distribution.py --dataset_type movi --data_root datasets/movi_ab_50k_shards --output_root /tmp/pv_stage1_cases_smoke --run_name smoke_distribution --padded_batch --max_objects 5 --max_T_raw 21 --max_points_per_object 20 --max_samples 4
- conda run -n videox python scripts/pv_simulator/diagnostics/run_stage1_cases.py --dataset_type movi --data_root datasets/movi_ab_50k_shards --ae_ckpt_dir outputs/stage0/sweeps_mae/training/lr1e-3-cosine/final --output_root /tmp/pv_stage1_cases_smoke --run_name smoke_cases --padded_batch --max_objects 5 --max_T_raw 5 --max_points_per_object 2 --cases multi_object_floor collision_heavy --case_sample_idx multi_object_floor:0 --case_sample_idx collision_heavy:0 --skip_stage1_pred --device cpu --max_chamfer_points 16
- conda run -n videox python scripts/pv_simulator/diagnostics/visualize_stage1_predictions.py --snapshot_npz /tmp/pv_stage1_cases_smoke/smoke_cases/multi_object_floor/case_snapshot.npz --output_dir /tmp/pv_stage1_cases_smoke/smoke_vis --fps 4 --max_points_per_object 2 --auto_select_keypoints 2
- conda run -n videox python scripts/pv_simulator/diagnostics/run_stage1_cases.py --dataset_type movi --data_root datasets/movi_ab_50k_shards --ae_ckpt_dir outputs/stage0/sweeps_mae/training/lr1e-3-cosine/final --output_root /tmp/pv_stage1_cases_smoke --run_name smoke_cases_vis --padded_batch --max_objects 5 --max_T_raw 5 --max_points_per_object 2 --cases multi_object_floor --case_sample_idx multi_object_floor:0 --skip_stage1_pred --device cpu --max_chamfer_points 16 --save_visualizations --vis_fps 4 --vis_max_points_per_object 2
- conda run -n videox python -m py_compile scripts/pv_simulator/diagnostics/audit_dataset_distribution.py scripts/pv_simulator/diagnostics/run_stage1_cases.py scripts/pv_simulator/diagnostics/visualize_stage1_predictions.py
- find /tmp/pv_stage1_cases_smoke/smoke_cases -maxdepth 2 -type f -print
- find /tmp/pv_stage1_cases_smoke/smoke_distribution -maxdepth 1 -type f -print
- find /tmp/pv_stage1_cases_smoke/smoke_cases_vis -maxdepth 3 -type f -print
- git status --short

Results: Added dataset distribution auditing, six-case diagnostic execution,
and snapshot visualization tooling. `audit_dataset_distribution.py` scans
dataset samples and saves contact frame ratio, rolling/sliding ratio, oblique
impact ratio, collision-heavy ratio, multi-object ratio, friction/restitution/
mass distributions, object-count distribution, and force/contact availability.
`run_stage1_cases.py` selects or accepts explicit samples for free fall,
vertical bounce, oblique impact, rolling/sliding, multi-object floor, and
collision-heavy/contact-dense cases, then writes per-case `gt.npy`,
`ae_recon.npy`, `naive_baseline.npy`, optional `simdit_pred.npy`,
`case_snapshot.npz`, `metrics.json`, `metrics.csv`, and optional visualization
videos. `visualize_stage1_predictions.py` renders saved snapshots. Smoke runs
passed in `videox` using `--skip_stage1_pred`; this validates dataset
heuristics, AE reconstruction, naive baseline, metrics, snapshots, README
writing, and visualization paths without requiring a Stage 1 checkpoint.

Pass/Fail: Pass

Problems found: Exact case labels are not available in the current sample dict,
so the scripts use documented heuristics. Contact can be inferred from nonzero
force/contact channels when available, otherwise from near-floor points and
vertical velocity sign changes near the floor. The smoke visualization still
emits harmless Matplotlib marker warnings for unfilled `x` markers. Full
SimDiT prediction was not smoke-tested because this step used
`--skip_stage1_pred`; use `--ckpt_dir` for the intended evaluation run.

Next action: Run the full six-case diagnostic with a real Stage 1 checkpoint:

```bash
conda run -n videox python scripts/pv_simulator/diagnostics/run_stage1_cases.py \
  --dataset_type movi \
  --data_root datasets/movi_ab_50k_shards \
  --ae_ckpt_dir outputs/stage0/sweeps_mae/training/lr1e-3-cosine/final \
  --ckpt_dir outputs/stage1/stage1_50k_padded/final \
  --padded_batch \
  --max_objects 5 \
  --max_T_raw 21 \
  --max_points_per_object 200 \
  --max_scan_samples 512 \
  --device cuda \
  --dtype bf16 \
  --num_inference_steps 50 \
  --save_visualizations
```

Expected outputs: `experiments/stage1_diagnostics/<YYYYMMDD_HHMMSS>_six_cases/`
with top-level `README.md`, `summary.json`, candidate distribution files, and
one subdirectory per found case containing trajectories, metrics, snapshots,
and optional videos.

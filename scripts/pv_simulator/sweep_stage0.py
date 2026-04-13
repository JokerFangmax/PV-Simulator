"""Automated Stage 0 AE hyperparameter sweep.

Runs three sequential phases with winner-picking between them:
  A. loss weights   — lambda_mmd x lambda_interp
  B. model capacity — c_mid x d_latent      (uses winner from A)
  C. training       — lr x lr_scheduler     (uses winner from A+B)

Two GPUs are used concurrently. Each run trains for 5000 steps, then is
evaluated via eval_ae.py --json. Results are written to
sweep/experiments.md after every status change; crashes are appended to
sweep/fixes.md with log tails.

Usage:
    python scripts/pv_simulator/sweep_stage0.py
"""

import argparse
import datetime as dt
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SWEEP_DIR = ROOT / "sweep"
FIXES     = SWEEP_DIR / "fixes.md"

TRAIN_SCRIPT = ROOT / "scripts" / "pv_simulator" / "train_stage0.py"
EVAL_SCRIPT  = ROOT / "scripts" / "pv_simulator" / "eval_ae.py"

# These are rebound in main() once --loss_type is parsed. They have defaults
# here so the module-level helper functions can reference them without care.
LOSS_TYPE    = "mse"
OUT_ROOT     = ROOT / "outputs" / "stage0" / "sweeps"
TRACKER      = SWEEP_DIR / "experiments.md"
WANDB_PREFIX = "pv-ae-sweep"

# Recon metric used by compute_composite_scores — rebound from LOSS_TYPE in main().
RECON_METRIC = "mse"


def configure_paths_for_loss_type(loss_type: str):
    """Set module-level paths and wandb project prefix for a given loss type.

    Called once from main(). Keeps MSE sweep (the default) at its original
    locations so backward compatibility is preserved.
    """
    global LOSS_TYPE, OUT_ROOT, TRACKER, WANDB_PREFIX, RECON_METRIC
    LOSS_TYPE = loss_type
    if loss_type == "mse":
        OUT_ROOT     = ROOT / "outputs" / "stage0" / "sweeps"
        TRACKER      = SWEEP_DIR / "experiments.md"
        WANDB_PREFIX = "pv-ae-sweep"
    else:
        OUT_ROOT     = ROOT / "outputs" / "stage0" / "sweeps_mae"
        TRACKER      = SWEEP_DIR / "experiments_mae.md"
        WANDB_PREFIX = "pv-ae-sweep-mae"
    RECON_METRIC = loss_type  # "mse" or "mae"

# ---------------------------------------------------------------------------
# Defaults shared across all runs
# ---------------------------------------------------------------------------

DEFAULTS = {
    "max_train_steps": 5000,
    "val_every":       1000,
    "checkpointing_steps": 5000,
    "batch_size":      256,
    "seed":            42,
    "mixed_precision": "bf16",
    "lambda_smooth":   0.0,
    "lr_total_steps":  5000,
    "lr_warmup_steps": 200,
    # knobs that phases override:
    "lambda_mmd":      0.1,
    "lambda_interp":   0.1,
    "c_mid":           64,
    "d_latent":        16,
    "n_res_blocks":    1,
    "lr":              3e-4,
    "lr_scheduler":    "cosine",
}

MAX_RUN_WALL_SECONDS = 120 * 60  # 120 min hard cap per run
MAX_RETRIES          = 2
N_GPUS               = 2

# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------

def phase_a_configs():
    """Phase A: sweep loss weights at fixed defaults."""
    mmds    = [0.0, 0.1, 1.0]
    interps = [0.0, 0.1, 1.0]
    runs = []
    for mmd, interp in itertools.product(mmds, interps):
        name = f"mmd{_fmt(mmd)}-int{_fmt(interp)}"
        cfg  = dict(DEFAULTS)
        cfg["lambda_mmd"]    = mmd
        cfg["lambda_interp"] = interp
        runs.append({"name": name, "category": "loss", "cfg": cfg})
    return runs


def phase_b_configs(winner_a):
    """Phase B: sweep model capacity, using Phase A winner for loss weights."""
    c_mids    = [32, 64, 128]
    d_latents = [8, 16, 32]
    runs = []
    for c_mid, d_lat in itertools.product(c_mids, d_latents):
        name = f"cmid{c_mid}-dlat{d_lat}"
        cfg  = dict(DEFAULTS)
        cfg["lambda_mmd"]    = winner_a["cfg"]["lambda_mmd"]
        cfg["lambda_interp"] = winner_a["cfg"]["lambda_interp"]
        cfg["c_mid"]         = c_mid
        cfg["d_latent"]      = d_lat
        runs.append({"name": name, "category": "capacity", "cfg": cfg})
    return runs


def phase_c_configs(winner_a, winner_b):
    """Phase C: sweep lr/scheduler using winners from A and B."""
    lrs    = [1e-4, 3e-4, 1e-3]
    scheds = ["cosine", "constant"]
    runs = []
    for lr, sched in itertools.product(lrs, scheds):
        name = f"lr{_fmt(lr)}-{sched}"
        cfg  = dict(DEFAULTS)
        cfg["lambda_mmd"]    = winner_a["cfg"]["lambda_mmd"]
        cfg["lambda_interp"] = winner_a["cfg"]["lambda_interp"]
        cfg["c_mid"]         = winner_b["cfg"]["c_mid"]
        cfg["d_latent"]      = winner_b["cfg"]["d_latent"]
        cfg["lr"]            = lr
        cfg["lr_scheduler"]  = sched
        runs.append({"name": name, "category": "training", "cfg": cfg})
    return runs


def phase_e_configs():
    """Stage E — depth + budget diagnostic (hardcoded 4 runs).

    Answers: is the MAE gap a width, depth, or training-budget problem?
      E1 mae-res2    — MAE winner cfg + n_res_blocks=2   (depth test)
      E2 mae-res3    — MAE winner cfg + n_res_blocks=3   (depth test)
      E3 mae-long    — MAE winner cfg at 15k steps       (budget test)
      E4 mse-res2    — MSE winner cfg + n_res_blocks=2   (depth helps MSE too?)
    """
    common = dict(DEFAULTS)
    common["lambda_mmd"]    = 0.1
    common["lambda_interp"] = 0.1
    common["lr"]            = 1e-3
    common["lr_scheduler"]  = "cosine"

    def mae_cfg(**overrides):
        cfg = dict(common)
        cfg["loss_type"] = "mae"
        cfg["c_mid"]     = 128
        cfg["d_latent"]  = 16
        cfg.update(overrides)
        return cfg

    def mse_cfg(**overrides):
        cfg = dict(common)
        cfg["loss_type"] = "mse"
        cfg["c_mid"]     = 128
        cfg["d_latent"]  = 32
        cfg.update(overrides)
        return cfg

    # Depth>1 runs OOM at bs=256 on a 24GB GPU because the interp-consistency
    # loss runs the AE three times per step (x, x_interp, x2); halve batch size.
    runs = [
        {"name": "mae-res2", "category": "stage_e",
         "cfg": mae_cfg(n_res_blocks=2, max_train_steps=5000, batch_size=128)},
        {"name": "mae-res3", "category": "stage_e",
         "cfg": mae_cfg(n_res_blocks=3, max_train_steps=5000, batch_size=128)},
        {"name": "mae-long", "category": "stage_e",
         "cfg": mae_cfg(n_res_blocks=1, max_train_steps=15000)},
        {"name": "mse-res2", "category": "stage_e",
         "cfg": mse_cfg(n_res_blocks=2, max_train_steps=5000, batch_size=128)},
    ]
    return runs


def phase_f_configs():
    """Stage F — Huber (SmoothL1) loss sweep at the mae-long winner config.

    Tests whether Huber gives MSE's convergence speed + MAE's detail accuracy.
    All runs use 15k steps (budget matters per Stage E findings).
    delta controls L1→L2 transition: small delta ≈ MAE, large delta ≈ MSE.
    """
    common = dict(DEFAULTS)
    common["lambda_mmd"]    = 0.1
    common["lambda_interp"] = 0.1
    common["lr"]            = 1e-3
    common["lr_scheduler"]  = "cosine"
    common["c_mid"]         = 128
    common["d_latent"]      = 16
    common["n_res_blocks"]  = 1
    common["max_train_steps"] = 15000
    common["loss_type"]     = "huber"

    runs = []
    for delta in [0.01, 0.05, 0.1, 0.3]:
        cfg = dict(common)
        cfg["huber_delta"] = delta
        cfg["lr_total_steps_override"] = 15000
        name = f"huber-d{delta}"
        runs.append({"name": name, "category": "stage_f", "cfg": cfg})
    return runs


def phase_g_configs():
    """Stage G — cosine_floor LR schedule sweep.

    Tests whether fast cosine decay + long floor training beats plain cosine-15k.
    Grid: lr_decay_steps ∈ {3000, 5000, 7000} × lr_min_ratio ∈ {0.01, 0.1}
    All runs use MAE loss at the mae-long winner config, 15k total steps.
    """
    common = dict(DEFAULTS)
    common["lambda_mmd"]    = 0.1
    common["lambda_interp"] = 0.1
    common["lr"]            = 1e-3
    common["lr_scheduler"]  = "cosine_floor"
    common["c_mid"]         = 128
    common["d_latent"]      = 16
    common["n_res_blocks"]  = 1
    common["max_train_steps"] = 15000
    common["loss_type"]     = "mae"
    common["lr_total_steps_override"] = 15000

    runs = []
    for decay in [3000, 5000, 7000]:
        for min_ratio in [0.01, 0.1]:
            cfg = dict(common)
            cfg["lr_decay_steps"] = decay
            cfg["lr_min_ratio"]   = min_ratio
            name = f"decay{decay}-floor{min_ratio}"
            runs.append({"name": name, "category": "stage_g", "cfg": cfg})
    return runs


def phase_d_configs(winner_abc, depths=(1, 2, 3)):
    """Phase D: sweep n_res_blocks (depth) at the A+B+C winner config.

    Runs are named `res{k}` and share every other hyperparameter with the
    current winner. Used to test whether per-level depth (stacking more
    ResBlocks) improves reconstruction on top of the already-swept width.
    """
    runs = []
    base = dict(winner_abc["cfg"])
    for k in depths:
        cfg = dict(base)
        cfg["n_res_blocks"] = k
        runs.append({"name": f"res{k}", "category": "depth", "cfg": cfg})
    return runs


def _fmt(x):
    if isinstance(x, float):
        if x == 0:
            return "0"
        if abs(x) < 1e-2 or abs(x) >= 1000:
            return f"{x:.0e}".replace("-0", "-")
        return f"{x:g}"
    return str(x)


# ---------------------------------------------------------------------------
# Winner selection
# ---------------------------------------------------------------------------
#
# MSE alone is insufficient: the AE's latent space is consumed by a
# flow-matching diffusion model in Stages 1/2, so the latent must also be
# close to N(0, I) (MMD) and locally linear under interpolation (interp).
# We combine the three into a *normalized composite score*: each metric is
# divided by its minimum across the phase's runs, then summed. This is
# scale-invariant (no hand-tuned weights) and penalizes being far from best
# on any axis.

def compute_composite_scores(results, metric_keys=None):
    """Compute normalized composite score for each successful run.

    Writes r["score_norm"] (dict: metric -> normalized value) and
    r["score"] (float: sum) in-place. Returns the list of successful runs.

    `metric_keys` defaults to (RECON_METRIC, "mmd", "interp"), so an MAE
    sweep scores by MAE and an MSE sweep by MSE — on top of the shared
    MMD and interp latent-quality terms.
    """
    if metric_keys is None:
        metric_keys = (RECON_METRIC, "mmd", "interp")
    good = [r for r in results if r.get("status") == "done" and r.get("metrics")]
    if not good:
        return []
    mins = {}
    for m in metric_keys:
        vals = [r["metrics"]["overall"].get(m) for r in good]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        mins[m] = max(1e-12, min(vals))
    for r in good:
        norm = {}
        for m, mn in mins.items():
            v = r["metrics"]["overall"].get(m)
            if v is None:
                continue
            norm[m] = v / mn
        r["score_norm"] = norm
        r["score"] = sum(norm.values())
    return good


def pick_winner_best_score(results):
    """Pick the run with the lowest normalized composite score.

    Composite = mse/mse_min + mmd/mmd_min + interp/interp_min.
    """
    good = compute_composite_scores(results)
    if not good:
        return None
    return min(good, key=lambda r: r["score"])


def pick_winner_smallest_model(results, slack=0.10):
    """Pick the smallest (c_mid*d_latent) model within +10% of best composite score.

    'Best' is the normalized composite score so that smaller-is-better
    respects latent quality (MMD, interp) as well as reconstruction (MSE).
    """
    good = compute_composite_scores(results)
    if not good:
        return None
    best_score = min(r["score"] for r in good)
    cutoff     = best_score * (1.0 + slack)
    eligible   = [r for r in good if r["score"] <= cutoff]
    return min(eligible, key=lambda r: r["cfg"]["c_mid"] * r["cfg"]["d_latent"])


# ---------------------------------------------------------------------------
# Subprocess orchestration
# ---------------------------------------------------------------------------

def build_train_command(run, out_dir, wandb_project):
    cfg = run["cfg"]
    # Per-run loss_type override; falls back to the module-global LOSS_TYPE
    # for callers (Phase A/B/C/D) that don't set it.
    loss_type = cfg.get("loss_type", LOSS_TYPE)
    max_steps = cfg["max_train_steps"]
    # Keep the LR schedule aligned with the actual step budget for this run
    # (so e.g. a 15k-step Stage E run gets its cosine curve across 15k steps,
    # not the DEFAULTS 5k). Explicit cfg["lr_total_steps"] still wins.
    lr_total_steps = cfg.get("lr_total_steps_override", max_steps)
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--output_dir", str(out_dir),
        "--wandb_project", wandb_project,
        "--wandb_run_name", run["name"],
        "--report_to", "wandb",
        "--max_train_steps",    str(max_steps),
        "--val_every",          str(cfg["val_every"]),
        "--checkpointing_steps",str(cfg["checkpointing_steps"]),
        "--batch_size",         str(cfg["batch_size"]),
        "--seed",               str(cfg["seed"]),
        "--mixed_precision",    cfg["mixed_precision"],
        "--loss_type",          loss_type,
        "--lambda_mmd",         str(cfg["lambda_mmd"]),
        "--lambda_interp",      str(cfg["lambda_interp"]),
        "--lambda_smooth",      str(cfg["lambda_smooth"]),
        "--c_mid",              str(cfg["c_mid"]),
        "--d_latent",           str(cfg["d_latent"]),
        "--n_res_blocks",       str(cfg["n_res_blocks"]),
        "--lr",                 str(cfg["lr"]),
        "--lr_scheduler",       cfg["lr_scheduler"],
        "--lr_warmup_steps",    str(cfg["lr_warmup_steps"]),
        "--lr_total_steps",     str(lr_total_steps),
    ]
    if loss_type == "huber":
        cmd += ["--huber_delta", str(cfg.get("huber_delta", 0.1))]
    if cfg["lr_scheduler"] == "cosine_floor":
        if "lr_decay_steps" in cfg:
            cmd += ["--lr_decay_steps", str(cfg["lr_decay_steps"])]
        if "lr_min_ratio" in cfg:
            cmd += ["--lr_min_ratio", str(cfg["lr_min_ratio"])]
    if cfg.get("loss_reweight", False):
        cmd.append("--loss_reweight")
    return cmd


def launch_training(run, gpu_idx, wandb_project):
    out_dir = OUT_ROOT / run["category"] / run["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    cmd = build_train_command(run, out_dir, wandb_project)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    env["PYTHONUNBUFFERED"] = "1"
    env["WANDB_SILENT"] = "true"

    log_file = open(log_path, "w")
    log_file.write(f"# cmd: {' '.join(cmd)}\n")
    log_file.write(f"# gpu: {gpu_idx}\n")
    log_file.write(f"# start: {dt.datetime.now().isoformat()}\n")
    log_file.flush()

    proc = subprocess.Popen(
        cmd, cwd=str(ROOT), env=env,
        stdout=log_file, stderr=subprocess.STDOUT,
    )
    return {
        "proc": proc,
        "log_file": log_file,
        "log_path": log_path,
        "out_dir": out_dir,
        "started": time.time(),
        "gpu_idx": gpu_idx,
    }


def run_eval(final_dir, gpu_idx):
    """Run eval_ae.py --json on a final/ directory. Returns parsed dict or None."""
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        str(final_dir),
        "--json",
        "--n_eval", "256",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    try:
        result = subprocess.run(
            cmd, cwd=str(ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=600,
        )
        if result.returncode != 0:
            return None, f"eval exit={result.returncode}: {result.stderr.decode()[:500]}"
        data = json.loads(result.stdout.decode())
        if not data:
            return None, "eval json empty"
        return data[0], None
    except subprocess.TimeoutExpired:
        return None, "eval timeout"
    except Exception as e:
        return None, f"eval error: {e}"


# ---------------------------------------------------------------------------
# Tracker markdown
# ---------------------------------------------------------------------------

def write_tracker(state):
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append(f"# Stage 0 AE — Sweep Tracker  (loss_type={LOSS_TYPE}, updated: {now})\n")
    lines.append("")
    lines.append("## Plan\n")
    lines.append(f"- **Phase A** (loss weights, `{WANDB_PREFIX}-loss`): 9 runs, λ_mmd × λ_interp.")
    lines.append(f"- **Phase B** (width, `{WANDB_PREFIX}-capacity`): 9 runs, c_mid × d_latent (uses winner from A).")
    lines.append(f"- **Phase C** (training dynamics, `{WANDB_PREFIX}-training`): 6 runs, lr × scheduler (uses winner from A+B).")
    lines.append(f"- **Phase D** (depth, `{WANDB_PREFIX}-depth`): sweep n_res_blocks at the A+B+C winner.")
    lines.append(f"- Each run: 5000 steps, bs=256, bf16, **loss_type={LOSS_TYPE}** (recon objective).")
    lines.append("- Validation logs BOTH MSE and MAE every 1000 steps regardless of training objective.")
    lines.append("- λ_smooth pinned to 0 per user feedback.")
    lines.append("")

    for phase_key in ["A", "B", "C", "D", "E", "F", "G"]:
        phase = state["phases"].get(phase_key)
        if phase is None:
            continue
        title = {
            "A": "Phase A — Loss weights",
            "B": "Phase B — Width (c_mid × d_latent)",
            "C": "Phase C — Training dynamics",
            "D": "Phase D — Depth (n_res_blocks)",
            "E": "Stage E — Depth + budget diagnostic",
            "F": "Stage F — Huber loss sweep",
            "G": "Stage G — cosine_floor LR schedule",
        }[phase_key]
        lines.append(f"## {title}  (status: {phase['status']})\n")
        if phase.get("note"):
            lines.append(f"> {phase['note']}\n")
        # Refresh composite scores so the table reflects the latest ranking.
        compute_composite_scores(phase["runs"])
        lines.append("| Run | λ_mmd | λ_interp | c_mid | d_latent | n_res | lr | sched | status | **score** | MSE | MAE | MMD | interp | smooth_cat | sharp_cat | sparse_cat | GPU |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in phase["runs"]:
            cfg = r["cfg"]
            m = r.get("metrics")
            if m:
                mse    = f"{m['overall']['mse']:.4f}"
                mae    = f"{m['overall'].get('mae', float('nan')):.4f}"
                mmd    = f"{m['overall']['mmd']:.4f}"
                interp = f"{m['overall']['interp']:.4f}"
                cat_key = RECON_METRIC if RECON_METRIC in m['per_category']['smooth'] else 'mse'
                sm     = f"{m['per_category']['smooth'][cat_key]:.4f}"
                sh     = f"{m['per_category']['sharp'][cat_key]:.4f}"
                sp     = f"{m['per_category']['sparse'][cat_key]:.4f}"
                sc     = f"{r.get('score', float('nan')):.3f}" if r.get("score") is not None else "—"
            else:
                mse = mae = mmd = interp = sm = sh = sp = sc = "—"
            status_icon = {
                "pending":  "⏳ pending",
                "running":  "🏃 running",
                "done":     "✅ done",
                "failed":   "❌ failed",
                "skipped":  "⚪ skipped",
            }.get(r.get("status", "pending"), r.get("status", "?"))
            gpu = r.get("gpu_idx", "")
            gpu = f"{gpu}" if gpu != "" else ""
            n_res = cfg.get("n_res_blocks", 1)
            lines.append(
                f"| {r['name']} | {_fmt(cfg['lambda_mmd'])} | {_fmt(cfg['lambda_interp'])} "
                f"| {cfg['c_mid']} | {cfg['d_latent']} | {n_res} | {_fmt(cfg['lr'])} | {cfg['lr_scheduler']} "
                f"| {status_icon} | **{sc}** | {mse} | {mae} | {mmd} | {interp} | {sm} | {sh} | {sp} | {gpu} |"
            )
        winner = phase.get("winner")
        if winner:
            lines.append("")
            m = winner["metrics"]["overall"]
            lines.append(
                f"**Winner**: `{winner['name']}` — "
                f"score={winner.get('score', float('nan')):.3f} "
                f"(MSE={m['mse']:.4f}, MAE={m.get('mae', float('nan')):.4f}, "
                f"MMD={m['mmd']:.4f}, interp={m['interp']:.4f})"
            )
        lines.append("")
        lines.append(
            f"*`score` = {RECON_METRIC}/{RECON_METRIC}_min + mmd/mmd_min + interp/interp_min "
            f"(lower is better, ≈3.0 means tied on all axes with the per-metric best). "
            f"`smooth/sharp/sparse_cat` show per-category {RECON_METRIC.upper()}.*"
        )
        lines.append("")

    lines.append("## Summary\n")
    for line in state.get("summary", []):
        lines.append(f"- {line}")
    lines.append("")
    TRACKER.write_text("\n".join(lines))


def append_fix_log(run, phase_key, message):
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tail = ""
    try:
        with open(run["log_path"], "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 8000))
            tail = f.read().decode(errors="replace")
    except Exception:
        pass
    tail_lines = tail.splitlines()[-60:]
    with open(FIXES, "a") as f:
        f.write(f"\n## {now}  Phase {phase_key}  run {run['name']} — CRASH\n")
        f.write(f"- category: {run['category']}\n")
        f.write(f"- cfg: {json.dumps(run['cfg'])}\n")
        f.write(f"- message: {message}\n")
        f.write("- tail of train.log:\n")
        f.write("```\n")
        f.write("\n".join(tail_lines))
        f.write("\n```\n")


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------

def run_phase(phase_key, runs, state, wandb_project):
    """Run all configs in a phase using a 2-GPU pool. Returns completed run list."""
    phase = state["phases"][phase_key]
    phase["runs"] = [dict(r, status="pending") for r in runs]
    phase["status"] = "running"
    write_tracker(state)

    queue   = list(range(len(runs)))
    active  = {}  # gpu_idx -> (idx, handle)
    retries = {i: 0 for i in range(len(runs))}

    def launch_next():
        if not queue:
            return False
        free_gpus = [g for g in range(N_GPUS) if g not in active]
        if not free_gpus:
            return False
        idx = queue.pop(0)
        gpu = free_gpus[0]
        run_entry = phase["runs"][idx]
        run_entry["status"]  = "running"
        run_entry["gpu_idx"] = gpu
        handle = launch_training(run_entry, gpu, wandb_project)
        active[gpu] = (idx, handle)
        write_tracker(state)
        print(f"[{phase_key}] launched {run_entry['name']} on GPU {gpu}")
        return True

    # Kick off up to N_GPUS runs
    while launch_next():
        pass

    while active:
        time.sleep(5)
        for gpu in list(active.keys()):
            idx, handle = active[gpu]
            run_entry = phase["runs"][idx]
            proc = handle["proc"]
            elapsed = time.time() - handle["started"]
            rc = proc.poll()

            if rc is None:
                # still running; enforce wall time
                if elapsed > MAX_RUN_WALL_SECONDS:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    handle["log_file"].close()
                    append_fix_log({**run_entry, "log_path": handle["log_path"]},
                                   phase_key, "wall-time cap exceeded")
                    run_entry["status"] = "failed"
                    run_entry["error"]  = "timeout"
                    del active[gpu]
                    write_tracker(state)
                continue

            # finished
            handle["log_file"].close()
            if rc != 0:
                # attempt retry
                retries[idx] += 1
                if retries[idx] <= MAX_RETRIES:
                    run_entry["status"] = "pending"
                    queue.append(idx)
                    append_fix_log({**run_entry, "log_path": handle["log_path"]},
                                   phase_key,
                                   f"exit={rc}, retrying ({retries[idx]}/{MAX_RETRIES})")
                    del active[gpu]
                    write_tracker(state)
                    while launch_next():
                        pass
                    continue
                else:
                    append_fix_log({**run_entry, "log_path": handle["log_path"]},
                                   phase_key,
                                   f"exit={rc}, retries exhausted")
                    run_entry["status"] = "failed"
                    run_entry["error"]  = f"exit {rc}"
                    del active[gpu]
                    write_tracker(state)
                    # still try to launch next
                    while launch_next():
                        pass
                    continue

            # success — run eval
            final_dir = handle["out_dir"] / "final"
            if not final_dir.exists():
                append_fix_log({**run_entry, "log_path": handle["log_path"]},
                               phase_key, "final/ missing after training")
                run_entry["status"] = "failed"
                run_entry["error"]  = "no final dir"
                del active[gpu]
                write_tracker(state)
                while launch_next():
                    pass
                continue

            metrics, err = run_eval(final_dir, gpu)
            if metrics is None:
                append_fix_log({**run_entry, "log_path": handle["log_path"]},
                               phase_key, f"eval failed: {err}")
                run_entry["status"] = "failed"
                run_entry["error"]  = err
            else:
                run_entry["status"]  = "done"
                run_entry["metrics"] = metrics
                print(f"[{phase_key}] done {run_entry['name']}  "
                      f"MSE={metrics['overall']['mse']:.4f}")
            del active[gpu]
            write_tracker(state)
            while launch_next():
                pass

    phase["status"] = "finished"
    write_tracker(state)
    return phase["runs"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def reload_phase_a_from_disk():
    """Rebuild Phase A run entries from completed dirs under outputs/stage0/sweeps/loss.

    Used when --skip_a is set so the tracker still shows Phase A results.
    Parses run names of the form `mmd<X>-int<Y>` to reconstruct cfg, then
    re-evaluates each final/ via eval_ae.py --json.
    """
    import re
    loss_dir = OUT_ROOT / "loss"
    if not loss_dir.exists():
        return []
    pat = re.compile(r"^mmd([0-9.]+)-int([0-9.]+)$")
    runs = []
    for d in sorted(loss_dir.iterdir()):
        if not d.is_dir():
            continue
        m = pat.match(d.name)
        if not m:
            continue
        final = d / "final"
        if not final.exists():
            continue
        cfg = dict(DEFAULTS)
        cfg["lambda_mmd"]    = float(m.group(1))
        cfg["lambda_interp"] = float(m.group(2))
        metrics, err = run_eval(final, gpu_idx=0)
        run_entry = {
            "name": d.name,
            "category": "loss",
            "cfg": cfg,
            "status": "done" if metrics else "failed",
            "metrics": metrics,
        }
        runs.append(run_entry)
        if metrics:
            print(f"[A-reload] {d.name}  MSE={metrics['overall']['mse']:.4f}")
        else:
            print(f"[A-reload] {d.name}  eval failed: {err}")
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "mae"],
                        help="Reconstruction objective for training. MSE sweep writes to "
                             "outputs/stage0/sweeps/ + sweep/experiments.md; MAE sweep writes to "
                             "outputs/stage0/sweeps_mae/ + sweep/experiments_mae.md with wandb "
                             "project prefix 'pv-ae-sweep-mae'.")
    parser.add_argument("--skip_a", action="store_true")
    parser.add_argument("--skip_b", action="store_true")
    parser.add_argument("--skip_c", action="store_true")
    parser.add_argument("--run_d", action="store_true",
                        help="Run Phase D: sweep n_res_blocks (depth) at the A+B+C winner.")
    parser.add_argument("--run_stage_e", action="store_true",
                        help="Run Stage E diagnostic: 4 runs that test width vs depth vs budget "
                             "for the MAE detail gap. Ignores all other phase flags, writes to "
                             "outputs/stage0/diag/stage_e and sweep/experiments_stage_e.md.")
    parser.add_argument("--run_stage_f", action="store_true",
                        help="Run Stage F: Huber loss sweep (delta in {0.01,0.05,0.1,0.3}) "
                             "at 15k steps. Writes to outputs/stage0/diag/stage_f and "
                             "sweep/experiments_stage_f.md.")
    parser.add_argument("--run_stage_g", action="store_true",
                        help="Run Stage G: cosine_floor LR schedule sweep "
                             "(decay_steps x min_ratio) at 15k steps.")
    parser.add_argument("--d_depths", type=int, nargs="+", default=[1, 2, 3],
                        help="n_res_blocks values to sweep in Phase D.")
    parser.add_argument("--force_c_lr", type=float, default=None,
                        help="Override lr used for Phase D (ignores Phase C winner).")
    parser.add_argument("--force_c_scheduler", type=str, default=None,
                        choices=["cosine", "constant"],
                        help="Override lr_scheduler used for Phase D.")
    parser.add_argument("--force_lambda_mmd", type=float, default=None,
                        help="Override the loss weight used for Phase B/C (ignores Phase A winner for this axis).")
    parser.add_argument("--force_lambda_interp", type=float, default=None,
                        help="Override the loss weight used for Phase B/C.")
    parser.add_argument("--force_c_mid", type=int, default=None,
                        help="Override the c_mid used for Phase C (ignores Phase B winner).")
    parser.add_argument("--force_d_latent", type=int, default=None,
                        help="Override the d_latent used for Phase C.")
    args = parser.parse_args()

    configure_paths_for_loss_type(args.loss_type)

    if args.run_stage_e or args.run_stage_f or args.run_stage_g:
        global OUT_ROOT, TRACKER, WANDB_PREFIX, RECON_METRIC
        RECON_METRIC = "mae"

    if args.run_stage_e:
        OUT_ROOT     = ROOT / "outputs" / "stage0" / "diag"
        TRACKER      = SWEEP_DIR / "experiments_stage_e.md"
        WANDB_PREFIX = "pv-ae-stage-e"

    if args.run_stage_f:
        OUT_ROOT     = ROOT / "outputs" / "stage0" / "diag"
        TRACKER      = SWEEP_DIR / "experiments_stage_f.md"
        WANDB_PREFIX = "pv-ae-stage-f"

    if args.run_stage_g:
        OUT_ROOT     = ROOT / "outputs" / "stage0" / "diag"
        TRACKER      = SWEEP_DIR / "experiments_stage_g.md"
        WANDB_PREFIX = "pv-ae-stage-g"

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not FIXES.exists():
        FIXES.write_text("# Sweep fixes / crash log\n")

    if args.run_stage_e:
        state = {
            "phases": {
                "E": {"runs": [], "status": "pending", "winner": None,
                      "note": "Stage E: depth + budget diagnostic (mixed MSE/MAE)"},
            },
            "summary": [],
        }
        write_tracker(state)
        runs_e = run_phase("E", phase_e_configs(), state, WANDB_PREFIX)
        winner_e = pick_winner_best_score(runs_e)
        state["phases"]["E"]["winner"] = winner_e
        # Report all successful runs in the summary — winner is informative
        # but E is a diagnostic, not a pick-one-and-proceed phase.
        for r in runs_e:
            if r.get("status") != "done":
                continue
            m = r["metrics"]["overall"]
            state["summary"].append(
                f"`{r['name']}` (loss={r['cfg']['loss_type']}, "
                f"c_mid={r['cfg']['c_mid']}, dlat={r['cfg']['d_latent']}, "
                f"n_res={r['cfg']['n_res_blocks']}, steps={r['cfg']['max_train_steps']}): "
                f"MSE={m['mse']:.4f}, MAE={m.get('mae', float('nan')):.4f}, "
                f"MMD={m['mmd']:.4f}, interp={m['interp']:.4f}, "
                f"score={r.get('score', float('nan')):.3f}"
            )
        if winner_e is not None:
            state["summary"].append(
                f"Best by MAE-composite: `{winner_e['name']}` — "
                f"use it to seed Stage F."
            )
        state["summary"].append("Stage E complete.")
        write_tracker(state)
        return

    if args.run_stage_f:
        state = {
            "phases": {
                "F": {"runs": [], "status": "pending", "winner": None,
                      "note": "Stage F: Huber loss sweep (delta ∈ {0.01, 0.05, 0.1, 0.3}), 15k steps"},
            },
            "summary": [],
        }
        write_tracker(state)
        runs_f = run_phase("F", phase_f_configs(), state, WANDB_PREFIX)
        winner_f = pick_winner_best_score(runs_f)
        state["phases"]["F"]["winner"] = winner_f
        for r in runs_f:
            if r.get("status") != "done":
                continue
            m = r["metrics"]["overall"]
            state["summary"].append(
                f"`{r['name']}` (delta={r['cfg'].get('huber_delta', '?')}): "
                f"MSE={m['mse']:.6f}, MAE={m.get('mae', float('nan')):.4f}, "
                f"MMD={m['mmd']:.4f}, interp={m['interp']:.4f}, "
                f"score={r.get('score', float('nan')):.3f}"
            )
        if winner_f is not None:
            state["summary"].append(
                f"Best by composite: `{winner_f['name']}` "
                f"(delta={winner_f['cfg'].get('huber_delta', '?')})"
            )
        state["summary"].append("Stage F complete.")
        write_tracker(state)
        return

    if args.run_stage_g:
        state = {
            "phases": {
                "G": {"runs": [], "status": "pending", "winner": None,
                      "note": "Stage G: cosine_floor LR sweep "
                              "(decay_steps ∈ {3k,5k,7k} × min_ratio ∈ {0.01,0.1}), "
                              "MAE loss, 15k steps"},
            },
            "summary": [],
        }
        write_tracker(state)
        runs_g = run_phase("G", phase_g_configs(), state, WANDB_PREFIX)
        winner_g = pick_winner_best_score(runs_g)
        state["phases"]["G"]["winner"] = winner_g
        for r in runs_g:
            if r.get("status") != "done":
                continue
            m = r["metrics"]["overall"]
            state["summary"].append(
                f"`{r['name']}` (decay={r['cfg'].get('lr_decay_steps', '?')}, "
                f"floor={r['cfg'].get('lr_min_ratio', '?')}): "
                f"MSE={m['mse']:.6f}, MAE={m.get('mae', float('nan')):.4f}, "
                f"MMD={m['mmd']:.4f}, interp={m['interp']:.4f}, "
                f"score={r.get('score', float('nan')):.3f}"
            )
        if winner_g is not None:
            state["summary"].append(
                f"Best by composite: `{winner_g['name']}` "
                f"(decay={winner_g['cfg'].get('lr_decay_steps', '?')}, "
                f"floor={winner_g['cfg'].get('lr_min_ratio', '?')})"
            )
        state["summary"].append("Stage G complete.")
        write_tracker(state)
        return

    state = {
        "phases": {
            "A": {"runs": [], "status": "pending", "winner": None},
            "B": {"runs": [], "status": "pending", "winner": None, "note": "waiting for Phase A"},
            "C": {"runs": [], "status": "pending", "winner": None, "note": "waiting for Phase A+B"},
            "D": {"runs": [], "status": "pending", "winner": None, "note": "waiting for Phase A+B+C"},
        },
        "summary": [],
    }
    write_tracker(state)

    # ---- Phase A ----
    if not args.skip_a:
        runs_a = run_phase("A", phase_a_configs(), state, f"{WANDB_PREFIX}-loss")
        winner_a = pick_winner_best_score(runs_a)
        state["phases"]["A"]["winner"] = winner_a
        if winner_a is None:
            state["summary"].append("Phase A: no successful runs; aborting subsequent phases.")
            write_tracker(state)
            return
        state["summary"].append(
            f"Phase A winner: `{winner_a['name']}` "
            f"(λ_mmd={winner_a['cfg']['lambda_mmd']}, λ_interp={winner_a['cfg']['lambda_interp']}) "
            f"score={winner_a['score']:.3f}  "
            f"[MSE={winner_a['metrics']['overall']['mse']:.4f}, "
            f"MMD={winner_a['metrics']['overall']['mmd']:.4f}, "
            f"interp={winner_a['metrics']['overall']['interp']:.4f}]"
        )
        write_tracker(state)
    else:
        print("[A] --skip_a set; reloading Phase A results from disk ...")
        reloaded_a = reload_phase_a_from_disk()
        state["phases"]["A"]["runs"]   = reloaded_a
        state["phases"]["A"]["status"] = "finished (reloaded)"
        compute_composite_scores(reloaded_a)
        prior_winner = pick_winner_best_score(reloaded_a)
        state["phases"]["A"]["winner"] = prior_winner
        winner_a = {"cfg": dict(DEFAULTS), "name": "forced/defaults"}
        if prior_winner is not None:
            state["summary"].append(
                f"Phase A reloaded from disk. Composite-best would have been "
                f"`{prior_winner['name']}` (score={prior_winner['score']:.3f}). "
                f"Per user direction this is overridden."
            )
        write_tracker(state)

    # Apply forced loss-weight overrides (e.g. user requested 0.1/0.1 for Phase B).
    if args.force_lambda_mmd is not None:
        winner_a = dict(winner_a, cfg=dict(winner_a["cfg"], lambda_mmd=args.force_lambda_mmd))
    if args.force_lambda_interp is not None:
        winner_a = dict(winner_a, cfg=dict(winner_a["cfg"], lambda_interp=args.force_lambda_interp))
    if args.force_lambda_mmd is not None or args.force_lambda_interp is not None:
        state["summary"].append(
            f"Phase B+C loss weights forced by user: "
            f"λ_mmd={winner_a['cfg']['lambda_mmd']}, λ_interp={winner_a['cfg']['lambda_interp']}."
        )
        winner_a["name"] = (f"forced_mmd{_fmt(winner_a['cfg']['lambda_mmd'])}"
                            f"-int{_fmt(winner_a['cfg']['lambda_interp'])}")
        write_tracker(state)

    # ---- Phase B ----
    if not args.skip_b:
        state["phases"]["B"]["note"] = None
        runs_b = run_phase("B", phase_b_configs(winner_a), state, f"{WANDB_PREFIX}-capacity")
        winner_b = pick_winner_smallest_model(runs_b)
        state["phases"]["B"]["winner"] = winner_b
        if winner_b is None:
            state["summary"].append("Phase B: no successful runs; aborting Phase C.")
            write_tracker(state)
            return
        state["summary"].append(
            f"Phase B winner (smallest within +10% of best composite score): `{winner_b['name']}` "
            f"(c_mid={winner_b['cfg']['c_mid']}, d_latent={winner_b['cfg']['d_latent']}) "
            f"score={winner_b['score']:.3f}  "
            f"[MSE={winner_b['metrics']['overall']['mse']:.4f}, "
            f"MMD={winner_b['metrics']['overall']['mmd']:.4f}, "
            f"interp={winner_b['metrics']['overall']['interp']:.4f}]"
        )
        write_tracker(state)
    else:
        winner_b = {"cfg": dict(DEFAULTS), "name": "forced/defaults"}

    # Apply forced capacity overrides (Phase C will use these).
    if args.force_c_mid is not None:
        winner_b = dict(winner_b, cfg=dict(winner_b["cfg"], c_mid=args.force_c_mid))
    if args.force_d_latent is not None:
        winner_b = dict(winner_b, cfg=dict(winner_b["cfg"], d_latent=args.force_d_latent))
    if args.force_c_mid is not None or args.force_d_latent is not None:
        state["summary"].append(
            f"Phase C capacity forced by user: "
            f"c_mid={winner_b['cfg']['c_mid']}, d_latent={winner_b['cfg']['d_latent']}."
        )
        winner_b["name"] = f"forced_cmid{winner_b['cfg']['c_mid']}-dlat{winner_b['cfg']['d_latent']}"
        write_tracker(state)

    # ---- Phase C ----
    if not args.skip_c:
        state["phases"]["C"]["note"] = None
        runs_c = run_phase("C", phase_c_configs(winner_a, winner_b), state, f"{WANDB_PREFIX}-training")
        winner_c = pick_winner_best_score(runs_c)
        state["phases"]["C"]["winner"] = winner_c
        if winner_c is not None:
            state["summary"].append(
                f"Phase C winner: `{winner_c['name']}` "
                f"(lr={winner_c['cfg']['lr']}, sched={winner_c['cfg']['lr_scheduler']}) "
                f"score={winner_c['score']:.3f}  "
                f"[MSE={winner_c['metrics']['overall']['mse']:.4f}, "
                f"MMD={winner_c['metrics']['overall']['mmd']:.4f}, "
                f"interp={winner_c['metrics']['overall']['interp']:.4f}]"
            )
            state["summary"].append(
                f"**Recommended Stage 0 config**: "
                f"λ_mmd={winner_a['cfg']['lambda_mmd']}, "
                f"λ_interp={winner_a['cfg']['lambda_interp']}, "
                f"λ_smooth=0, "
                f"c_mid={winner_b['cfg']['c_mid']}, "
                f"d_latent={winner_b['cfg']['d_latent']}, "
                f"lr={winner_c['cfg']['lr']}, "
                f"scheduler={winner_c['cfg']['lr_scheduler']}"
            )
        else:
            state["summary"].append("Phase C: no successful runs.")
        write_tracker(state)
    else:
        winner_c = {"cfg": dict(DEFAULTS), "name": "forced/defaults"}

    # ---- Phase D (depth sweep, opt-in) ----
    if args.run_d:
        # Build the starting config for Phase D from A+B+C winners, with
        # optional per-knob overrides for standalone use.
        winner_abc_cfg = dict(DEFAULTS)
        winner_abc_cfg["lambda_mmd"]    = winner_a["cfg"]["lambda_mmd"]
        winner_abc_cfg["lambda_interp"] = winner_a["cfg"]["lambda_interp"]
        winner_abc_cfg["c_mid"]         = winner_b["cfg"]["c_mid"]
        winner_abc_cfg["d_latent"]      = winner_b["cfg"]["d_latent"]
        winner_abc_cfg["lr"]            = winner_c["cfg"]["lr"]
        winner_abc_cfg["lr_scheduler"]  = winner_c["cfg"]["lr_scheduler"]
        if args.force_c_lr is not None:
            winner_abc_cfg["lr"] = args.force_c_lr
        if args.force_c_scheduler is not None:
            winner_abc_cfg["lr_scheduler"] = args.force_c_scheduler
        winner_abc = {
            "cfg": winner_abc_cfg,
            "name": (f"abc_mmd{_fmt(winner_abc_cfg['lambda_mmd'])}"
                     f"-int{_fmt(winner_abc_cfg['lambda_interp'])}"
                     f"-cmid{winner_abc_cfg['c_mid']}"
                     f"-dlat{winner_abc_cfg['d_latent']}"
                     f"-lr{_fmt(winner_abc_cfg['lr'])}"
                     f"-{winner_abc_cfg['lr_scheduler']}"),
        }
        state["phases"]["D"]["note"] = None
        runs_d = run_phase(
            "D", phase_d_configs(winner_abc, depths=tuple(args.d_depths)),
            state, f"{WANDB_PREFIX}-depth",
        )
        winner_d = pick_winner_best_score(runs_d)
        state["phases"]["D"]["winner"] = winner_d
        if winner_d is not None:
            m = winner_d["metrics"]["overall"]
            state["summary"].append(
                f"Phase D winner: `{winner_d['name']}` "
                f"(n_res_blocks={winner_d['cfg']['n_res_blocks']}) "
                f"score={winner_d['score']:.3f}  "
                f"[MSE={m['mse']:.4f}, MAE={m.get('mae', float('nan')):.4f}, "
                f"MMD={m['mmd']:.4f}, interp={m['interp']:.4f}]"
            )
            state["summary"].append(
                f"**Recommended Stage 0 config (with depth)**: "
                f"λ_mmd={winner_abc_cfg['lambda_mmd']}, "
                f"λ_interp={winner_abc_cfg['lambda_interp']}, "
                f"λ_smooth=0, "
                f"c_mid={winner_abc_cfg['c_mid']}, "
                f"d_latent={winner_abc_cfg['d_latent']}, "
                f"n_res_blocks={winner_d['cfg']['n_res_blocks']}, "
                f"lr={winner_abc_cfg['lr']}, "
                f"scheduler={winner_abc_cfg['lr_scheduler']}"
            )
        else:
            state["summary"].append("Phase D: no successful runs.")
        write_tracker(state)

    state["summary"].append("Sweep complete.")
    write_tracker(state)


if __name__ == "__main__":
    main()

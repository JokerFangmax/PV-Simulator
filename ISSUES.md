# PV-Simulator: Known Issues

Issues found during Stage 0 development. Stages 1 and 2 likely share the same patterns and should be audited for the same problems.

---

## [FIXED] Stage 0 — `train_stage0.py` and `orchestrate_stage0.py`

### 1. Resume does not account for already-completed steps

**File:** `scripts/pv_simulator/train_stage0.py`

**Problem:** When resuming from a checkpoint saved at step N with `--resume_from`, `global_step` resets to 0 and the loop runs for `max_train_steps` *additional* steps. Passing `--max_train_steps 50000` after a 20k-step checkpoint results in 70k total steps, not 50k.

The LR scheduler also restarts from relative step 0, so the cosine schedule restarts near peak LR instead of continuing from the mid-curve position it should be at.

**Fix (applied):**
- Added `--start_step` argument (default 0). Pass `--start_step N` when resuming from a checkpoint at step N.
- Training loop now runs `max_train_steps - start_step` iterations.
- `global_step` initialized to `start_step`.
- `lr_lambda` converts relative step → absolute step by adding `start_step`, so the cosine/linear schedule continues from the correct position.

**Applies to Stages 1 & 2:** Yes — `train_stage1.py` and `train_stage2.py` should get the same `--start_step` argument and loop/scheduler fix.

---

### 2. LR warmup argument existed but was never implemented

**File:** `scripts/pv_simulator/train_stage0.py`

**Problem:** `--lr_warmup_steps` was accepted as an argument but the code created `CosineAnnealingLR` directly, ignoring warmup entirely.

**Fix (applied):** Replaced `CosineAnnealingLR` with a `LambdaLR` that does linear warmup for `lr_warmup_steps`, then cosine/linear/constant decay for the remainder.

**Applies to Stages 1 & 2:** Likely yes — check whether warmup is actually applied in those scripts.

---

### 3. LR scheduler created before `accelerator.prepare(optimizer)`

**File:** `scripts/pv_simulator/train_stage0.py`

**Problem:** The scheduler was constructed with the raw (un-wrapped) optimizer, then `accelerator.prepare(optimizer)` returned a new wrapped optimizer. The scheduler held a stale reference to the original optimizer object.

**Fix (applied):** Moved scheduler construction to after `accelerator.prepare(ae, optimizer)`.

**Applies to Stages 1 & 2:** Yes — check scheduler construction order in both scripts.

---

### 4. No gradient clipping

**File:** `scripts/pv_simulator/train_stage0.py`

**Problem:** No gradient norm clipping, which can cause instability especially early in training with high learning rates.

**Fix (applied):** Added `--grad_clip 1.0` (default). Applied via `accelerator.clip_grad_norm_()` before `optimizer.step()`.

**Applies to Stages 1 & 2:** Yes — add `--grad_clip` to both.

---

### 5. Logged loss is a single noisy batch, not a smoothed metric

**File:** `scripts/pv_simulator/train_stage0.py`

**Problem:** Each synthetic batch is drawn from a randomly chosen generator (e.g., `gen_gaussian_noise` with std up to 10, `gen_piecewise_constant`). A single unlucky batch causes large MSE spikes in the log (e.g., 3 → 26 → 3) that look like training instability but are just sampling noise.

**Fix (applied):** Added an EMA (α=0.98, ~50-step effective window) over all loss components. Logged values now reflect the recent trend rather than one noisy sample.

**Applies to Stages 1 & 2:** Stages 1 & 2 use real data with a DataLoader, so batch-to-batch variance is lower, but EMA logging is still good practice.

---

### 6. Deprecated `torch.cuda.amp.autocast` API

**File:** `scripts/pv_simulator/train_stage0.py`

**Problem:** Used `torch.cuda.amp.autocast(dtype=...)` which is deprecated in newer PyTorch versions.

**Fix (applied):** Updated to `torch.amp.autocast("cuda", dtype=...)`.

**Applies to Stages 1 & 2:** Check both scripts for the same deprecated call.

---

### 7. LR schedule discontinuity when resuming from a screening run

**File:** `scripts/pv_simulator/train_stage0.py`, `scripts/pv_simulator/orchestrate_stage0.py`

**Problem:** The screening run (20k steps) uses `max_train_steps=20000` as the cosine cycle length, so its LR decays to ≈0 by step 20k. When the continuation resumes with `max_train_steps=50000`, the schedule treats step 20k as being 40% through a 50k cycle — LR ≈ 65% of peak. This causes a large upward LR jump at the resume boundary; the two runs don't share the same cosine curve.

Concretely:
- Screening end:    `progress = 19500/19500 = 1.0`  → LR ≈ 0
- Continuation start: `progress = 19500/49500 ≈ 0.39` → LR ≈ 65% of peak

**Fix (applied):** Added `--lr_total_steps` (default: `max_train_steps`). Setting it to the **full planned training length** on both the screening and continuation runs makes them share one coherent cosine curve.

```bash
# Screening run — treat as first 20k of a planned 50k cosine:
python train_stage0.py --max_train_steps 20000 --lr_total_steps 50000 ...

# Continuation — same cosine, picks up at step 20k:
python train_stage0.py --max_train_steps 50000 --lr_total_steps 50000 \
    --start_step 20000 --resume_from <ckpt> ...
```

The orchestrator now always passes `lr_total_steps=FINAL_STEPS` to all three launch calls.

**Applies to Stages 1 & 2:** Yes — same pattern applies whenever training is done in phases or resumed.

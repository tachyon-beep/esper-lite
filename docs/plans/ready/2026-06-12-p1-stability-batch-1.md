# P1 Stability Batch 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the first post-baseline P1 stability batch so PPO update accounting, entropy/KL metrics, rollout telemetry, and gradient drift tracking cannot silently report healthy data for invalid training states.

**Architecture:** This is a bounded stabilization slice on top of merged recovery PR #72. Keep changes in Simic PPO math/coordinator modules, add focused regression tests, and close each Filigree issue only after the code is committed and verified.

**Tech Stack:** Python 3.11, PyTorch, pytest, Ruff, mypy, Filigree.

**Prerequisites:**
- Branch from current `origin/main`: `codex/p1-stability-batch-1`.
- Claimed Filigree issues: `esper-lite-18eb2f`, `esper-lite-2fcc87`, `esper-lite-5f7f67`, `esper-lite-1bbfb2`, `esper-lite-c82e50`, `esper-lite-ee44b1`.
- Preserve unrelated dirty files in `/home/john/esper-lite`; do all code edits in `/home/john/.config/superpowers/worktrees/esper-lite/codex-steady-state-recovery`.

---

## Task 1: PPO Entropy Schedule Bounds

**Files:**
- Modify: `src/esper/simic/agent/ppo_agent.py`
- Test: `tests/simic/agent/test_ppo_entropy_floor.py`

**Steps:**
1. Add tests proving `total_train_steps=0` is rejected and progress beyond 100% still produces the late-training floor, not a negative multiplier.
2. Validate `total_train_steps > 0` in `PPOAgent.__init__`.
3. Clamp progress at the schedule boundary used by `update()`.

**Definition of Done:**
- `esper-lite-18eb2f` cannot reproduce.
- Schedule factor remains in `[0.5, 1.5]` even when `train_steps > total_train_steps`.

## Task 2: PPO KL No-Step Contract

**Files:**
- Inspect/modify if needed: `src/esper/simic/agent/ppo_agent.py`
- Inspect/modify if needed: `src/esper/simic/agent/ppo_metrics.py`
- Test: `tests/simic/test_ppo.py`
- Test: `tests/simic/training/test_ppo_coordinator.py`

**Steps:**
1. Verify the current epoch-0 KL abort path counts optimizer steps, not ratio-stat epochs.
2. If already implemented, add tracker comments and close `esper-lite-2fcc87` as fixed by existing code on `main`.
3. If gaps remain, preserve the no-step contract: `ppo_update_performed=False`, loss fields `NaN`, no `ppo_grad_norm` requirement in coordinator.

**Definition of Done:**
- Existing or new regression proves epoch-0 KL abort does not advance as a performed PPO update.

## Task 3: Zero-Availability Entropy Floor

**Files:**
- Modify: `src/esper/simic/agent/ppo_update.py`
- Test: `tests/simic/agent/test_ppo_entropy_floor.py`

**Steps:**
1. Write a failing `compute_losses()` regression where a head has zero availability and low entropy.
2. Route per-step entropy plus effective availability masks into `compute_entropy_floor_penalty()` without precomputing a clamped zero mean.
3. Keep forced-action masking behavior intact.

**Definition of Done:**
- `esper-lite-5f7f67` cannot reproduce.
- Unavailable heads add no entropy-floor penalty.

## Task 4: Zero-Mask KL Weighting

**Files:**
- Modify: `src/esper/simic/agent/ppo_update.py`
- Test: `tests/simic/agent/test_ppo_ratio_metrics.py` or an existing PPO math test file.

**Steps:**
1. Add a regression with one valid high-KL head and one all-zero mask head.
2. Remove the artificial denominator contribution from zero-mask heads in `compute_ratio_metrics()`.
3. Keep the all-zero-mask case finite and explicit.

**Definition of Done:**
- `esper-lite-1bbfb2` cannot reproduce.
- All-zero heads contribute zero weight, not zero KL with positive weight.

## Task 5: Rollout Total Steps

**Files:**
- Modify: `src/esper/simic/training/ppo_coordinator.py`
- Test: `tests/simic/training/test_ppo_coordinator.py`

**Steps:**
1. Capture `rollout_total_steps` before `run_ppo_updates_fn()` can clear the buffer.
2. Preserve existing skipped-update behavior.
3. Add a regression where the update function clears the buffer and the metric still reports pre-update length.

**Definition of Done:**
- `esper-lite-c82e50` cannot reproduce.

## Task 6: Non-Finite Gradient Drift

**Files:**
- Modify: `src/esper/simic/training/ppo_coordinator.py`
- Modify: `src/esper/simic/telemetry/gradient_ema.py`
- Test: `tests/simic/training/test_ppo_coordinator.py`
- Test: `tests/simic/test_gradient_ema.py`

**Steps:**
1. Add tests proving `NaN` and `Inf` gradient norms do not update EMA state.
2. Return unhealthy drift metrics for non-finite norms without corrupting the tracker.
3. Reject non-finite inputs directly inside `GradientEMATracker.update()`.

**Definition of Done:**
- `esper-lite-ee44b1` cannot reproduce.
- EMA state remains finite after non-finite gradient observations.

## Verification

Run the focused gate first:

```bash
PYTHONPATH=src uv run pytest tests/simic/agent/test_ppo_entropy_floor.py tests/simic/test_ppo.py tests/simic/training/test_ppo_coordinator.py tests/simic/test_gradient_ema.py -q
uv run ruff check src/esper/simic/agent/ppo_agent.py src/esper/simic/agent/ppo_update.py src/esper/simic/training/ppo_coordinator.py src/esper/simic/telemetry/gradient_ema.py tests/simic/agent/test_ppo_entropy_floor.py tests/simic/test_ppo.py tests/simic/training/test_ppo_coordinator.py tests/simic/test_gradient_ema.py
MYPYPATH=src uv run mypy -p esper
uv run python scripts/lint_defensive_patterns.py
```

Then run the broader recovery gate before opening the PR:

```bash
uv run ruff check src/ tests/
uv run python scripts/lint_leyline_types.py
uv run python scripts/lint_gpu_sync.py
PYTHONPATH=src uv run pytest tests/simic --ignore=tests/simic/test_data_opt.py --ignore=tests/simic/test_record_stream_fix.py --ignore=tests/simic/training/test_dual_ab.py -q
```

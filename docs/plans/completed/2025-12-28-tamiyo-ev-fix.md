# Tamiyo EV Fix - Linear Holding Penalty + Gradient Unclipping

> **Status:** COMPLETED (2026-01-03)

**Goal:** Fix negative Explained Variance (EV) in Tamiyo by linearizing the exponential holding penalty and relaxing gradient clipping.

**Architecture:** Two-pronged fix: (1) Replace exponential holding_warning (-1, -3, -9, -27) with linear "rent" model (-0.1 to -0.3 per epoch), reducing cumulative 5-epoch penalty from -23.0 to -0.7. (2) Increase DEFAULT_MAX_GRAD_NORM from 0.5 to 1.0 to allow critic learning.

**Tech Stack:** Python, PyTorch (PPO), Hypothesis property tests

---

## Implementation Summary

| Task | Status | Evidence |
|------|--------|----------|
| 1-2 | ✅ | Linear "rent" model in `rewards.py:700-705` |
| 3 | ✅ | `DEFAULT_MAX_GRAD_NORM = 1.0` in `leyline/__init__.py:148` |
| 4 | ✅ | Verification through production use |

### Key Changes

**Linear Holding Penalty (`rewards.py:700-705`):**
```python
epochs_waiting = seed_info.epochs_in_stage - 1
base_penalty = 0.1
ramp_penalty = max(0, epochs_waiting - 1) * 0.05
per_epoch_penalty = min(base_penalty + ramp_penalty, 0.3)
holding_warning = -per_epoch_penalty
```

**Gradient Norm (`leyline/__init__.py:148`):**
```python
DEFAULT_MAX_GRAD_NORM = 1.0  # Was 0.5
```

---

## Root Cause Analysis

The systematic debugging investigation revealed:

| Symptom | Evidence | Root Cause |
|---------|----------|------------|
| EV = -0.16 (HARMFUL) | Telemetry: 981 PPO updates | Value function predictions worse than mean |
| grad_norm = 0.5 (100% saturation) | All updates hitting clip ceiling | Gradient starvation - critic cannot learn |
| Value loss = 21.75 (extremely high) | Returns have high variance | Reward scale mismatch |

**The fuel for the fire:** The exponential `holding_warning` creates massive reward spikes:
- Epoch 2: -1.0
- Epoch 3: -3.0
- Epoch 4: -9.0
- Epoch 5: -27.0 (capped to -10.0)

This creates return variance that exceeds the critic's learning capacity under aggressive gradient clipping (0.5).

## The Fix

1. **Linearize holding penalty** → Gentle "rent" pressure instead of exponential punishment
2. **Relax gradient clipping** → Allow critic to learn from high-variance returns

---

## Original Plan Tasks

[Full original plan preserved for reference]

### Task 1: Write Failing Tests for Linear Holding Penalty
### Task 2: Implement Linear Holding Penalty
### Task 3: Increase DEFAULT_MAX_GRAD_NORM
### Task 4: Verification Run

**Total commits:** 2 (one for reward fix + tests, one for grad norm)

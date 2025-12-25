# Tolaria Remediation: Dead Code Removal + Gradient Clipping

**Date:** 2025-12-25
**Status:** Planned
**Priority:** P0 (Gradient Clipping) + P1 (Dead Code Deletion)
**Branch:** `fix/tolaria-remediation`

## Overview

Tolaria has a split-brain problem: `trainer.py` contains tested-but-never-executed training functions, while production uses inline implementations in `simic/training/vectorized.py`. This plan:

1. **Adds gradient clipping to production** (missing critical feature)
2. **Deletes dead code** (~1,750 lines of unused trainer functions + tests)
3. **Adds perplexity tracking** for LM tasks (nice-to-have)

### Goals

1. **Production has gradient clipping** - Critical for Transformer hosts (TinyStories)
2. **No dead code remains** - Follow "No Legacy Code Policy"
3. **Tests validate production code** - Not dead abstractions
4. **Tolaria is minimal** - Only `create_model` and `TolariaGovernor`

### Non-Goals

- Wiring the dead Tolaria trainer functions into production (wrong architecture)
- Backward compatibility with the dead API
- Per-class accuracy tracking (low priority, can add later)

## Current State (Problem)

```
┌─────────────────────────────────────────────────────────────────┐
│                         TOLARIA                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ environment.py                                               ││
│  │   └── create_model()               ← USED IN PRODUCTION     ││
│  │                                                              ││
│  │ governor.py                                                  ││
│  │   └── TolariaGovernor              ← USED IN PRODUCTION     ││
│  │                                                              ││
│  │ trainer.py                         ← DEAD CODE (433 lines)  ││
│  │   └── train_epoch_normal()         ← Never called           ││
│  │   └── train_epoch_incubator_mode() ← Never called           ││
│  │   └── train_epoch_blended()        ← Never called           ││
│  │   └── validate_and_get_metrics()   ← Never called           ││
│  │   └── validate_with_attribution()  ← Never called           ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    SIMIC (PRODUCTION)                            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ training/vectorized.py                                       ││
│  │   └── process_train_batch()        ← ACTUAL TRAINING        ││
│  │       └── NO gradient clipping     ← MISSING FEATURE        ││
│  │       └── AMP with GradScaler      ← Needs unscale_()       ││
│  │   └── process_val_batch()          ← ACTUAL VALIDATION      ││
│  │                                                              ││
│  │ attribution/counterfactual.py      ← BETTER than Tolaria    ││
│  │   └── CounterfactualEngine         ← Shapley values         ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### The Split-Brain Evidence

| Function | Tests? | Production Calls | Verdict |
|----------|--------|------------------|---------|
| `train_epoch_normal` | 15+ tests | 0 | DEAD |
| `train_epoch_incubator_mode` | 10+ tests | 0 | DEAD |
| `train_epoch_blended` | 5+ tests | 0 | DEAD |
| `validate_and_get_metrics` | 8+ tests | 0 | DEAD |
| `validate_with_attribution` | 12+ tests | 0 | DEAD (superseded by CounterfactualEngine) |

### Missing Production Feature

The dead code has `max_grad_norm` gradient clipping. Production does not:

```python
# Dead code (tolaria/trainer.py:131) - HAS clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

# Production (vectorized.py:1426-1458) - NO clipping
if env_state.scaler is not None:
    env_state.scaler.step(env_state.host_optimizer)  # No unscale_(), no clip!
```

**Expert Consensus:**
- **DRL Expert:** "Nice-to-have for CNNs, CRITICAL for Transformers"
- **PyTorch Expert:** "P0 - Do Now. Also missing `unscale_()` before clipping"

## Target State

```
┌─────────────────────────────────────────────────────────────────┐
│                         TOLARIA (MINIMAL)                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ __init__.py                                                  ││
│  │   └── exports: create_model, TolariaGovernor, GovernorReport││
│  │                                                              ││
│  │ environment.py                                               ││
│  │   └── create_model()                                        ││
│  │                                                              ││
│  │ governor.py                                                  ││
│  │   └── TolariaGovernor, GovernorReport                       ││
│  │                                                              ││
│  │ (trainer.py DELETED)                                         ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    SIMIC (WITH GRADIENT CLIPPING)                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ training/vectorized.py                                       ││
│  │   └── process_train_batch()                                  ││
│  │       └── scaler.unscale_()        ← NEW                    ││
│  │       └── clip_grad_norm_()        ← NEW                    ││
│  │       └── scaler.step()                                      ││
│  │                                                              ││
│  │ training/config.py                                           ││
│  │   └── TrainingConfig                                         ││
│  │       └── max_grad_norm: float = 1.0  ← NEW                 ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Task Dependency Chain

```
Task 1 ──► Task 2 ──► Task 3 ──► Task 4 ──► Task 5 ──► Task 6
  │           │           │           │           │
  │           │           │           │           └── Integration verification
  │           │           │           └── Delete dead tests
  │           │           └── Delete trainer.py
  │           └── Implement gradient clipping
  └── Add TrainingConfig parameters
```

**IMPORTANT:** Tasks are strictly sequential. Do NOT parallelize.

---

## Implementation Tasks

### Task 1: Add Gradient Clipping Config Parameters

**Depends on:** Nothing (first task)

Add `max_grad_norm` to `TrainingConfig` and wire through to `train_ppo_vectorized()`.

**Files:**
- MODIFY: `src/esper/simic/training/config.py`

**Changes to TrainingConfig:**

```python
@dataclass
class TrainingConfig:
    # ... existing fields ...

    # NEW: Gradient clipping for host/seed training
    # Default 1.0 is standard for supervised training (PPO policy uses 0.5)
    max_grad_norm: float = 1.0
```

**Changes to to_train_kwargs():**

```python
def to_train_kwargs(self) -> dict:
    return {
        # ... existing fields ...
        "max_grad_norm": self.max_grad_norm,  # NEW
    }
```

**Test:** `uv run pytest tests/simic/training/test_config.py -v`

---

### Task 2: Implement Gradient Clipping in process_train_batch()

**Depends on:** Task 1 (config parameter must exist)

Add proper AMP-safe gradient clipping to the production training loop.

**Files:**
- MODIFY: `src/esper/simic/training/vectorized.py`

**Critical Implementation Notes:**

The correct AMP ordering is:
1. `scaler.scale(loss).backward()` - Scale loss for numeric range
2. `scaler.unscale_(optimizer)` - **REQUIRED BEFORE CLIPPING**
3. `clip_grad_norm_(params, max_norm)` - Clip at true FP32 magnitude
4. `scaler.step(optimizer)` - Apply update (skips if inf/NaN)
5. `scaler.update()` - Update scale factor

**Current code (lines 1414-1458):**
```python
if env_state.scaler is not None:
    env_state.scaler.scale(loss).backward()
else:
    loss.backward()

# ... telemetry collection ...

if env_state.scaler is not None:
    env_state.scaler.step(env_state.host_optimizer)
    for slot_id in slots_to_step:
        # ...
        env_state.scaler.step(seed_opt)
    env_state.scaler.update()
else:
    env_state.host_optimizer.step()
    for slot_id in slots_to_step:
        env_state.seed_optimizers[slot_id].step()
```

**New code (replaces existing step logic):**

```python
# Compute grad presence once for each seed optimizer (avoid redundant checks)
seed_opts_with_grads: dict[str, tuple[torch.optim.Optimizer, bool]] = {}
for slot_id in slots_to_step:
    seed_opt = env_state.seed_optimizers[slot_id]
    has_grads = any(
        p.grad is not None for group in seed_opt.param_groups for p in group["params"]
    )
    seed_opts_with_grads[slot_id] = (seed_opt, has_grads)

# Gradient clipping (AMP-safe)
if max_grad_norm is not None and max_grad_norm > 0:
    if env_state.scaler is not None:
        # Unscale before clipping (required for correct FP32 magnitude)
        env_state.scaler.unscale_(env_state.host_optimizer)
        for slot_id, (seed_opt, has_grads) in seed_opts_with_grads.items():
            if has_grads:
                env_state.scaler.unscale_(seed_opt)

    # Clip all parameters (works for both AMP and non-AMP)
    all_params = list(model.get_host_parameters())
    for slot_id in slots_to_step:
        all_params.extend(model.get_seed_parameters(slot_id))
    torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)

# Optimizer step (reuses has_grads computation)
if env_state.scaler is not None:
    env_state.scaler.step(env_state.host_optimizer)
    for slot_id, (seed_opt, has_grads) in seed_opts_with_grads.items():
        if has_grads:
            env_state.scaler.step(seed_opt)
        else:
            seed_opt.step()  # No grads from scaled backward - step without scaler
    env_state.scaler.update()
else:
    env_state.host_optimizer.step()
    for slot_id in slots_to_step:
        env_state.seed_optimizers[slot_id].step()
```

**Key implementation notes:**
1. `has_grads` computed once, reused for both `unscale_()` and `step()`
2. `unscale_()` must be called before `clip_grad_norm_()` in AMP path
3. `scaler.step()` automatically skips if inf/NaN detected during unscaling
4. Non-AMP path uses identical clipping, just skips `unscale_()`

**Also update function signature:**

```python
def process_train_batch(
    env_state: ParallelEnvState,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    use_telemetry: bool = False,
    slots: list[str] | None = None,
    use_amp: bool = False,
    max_grad_norm: float | None = None,  # NEW
) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, dict] | None]:
```

**And update the caller in the epoch loop** to pass `max_grad_norm` from config.

**Test:**
```bash
# Unit test for clipping behavior
uv run pytest tests/simic/training/test_gradient_clipping.py -v

# Integration test - verify training still works
PYTHONPATH=src uv run python -m esper.scripts.train ppo --episodes 2 --n-envs 2
```

---

### Task 3: Delete trainer.py

**Depends on:** Task 2 (gradient clipping must be in production first)

Remove the dead trainer module entirely.

**Files:**
- DELETE: `src/esper/tolaria/trainer.py` (433 lines)
- MODIFY: `src/esper/tolaria/__init__.py`

**New __init__.py contents:**

```python
"""Tolaria - Model Training Infrastructure

This package provides:
- environment: Model factory (create_model)
- governor: Fail-safe watchdog for catastrophic failure detection

Training loops are implemented inline in simic/training/vectorized.py
for performance (CUDA streams, AMP, multi-env parallelism).
"""

from esper.tolaria.environment import create_model
from esper.tolaria.governor import GovernorReport, TolariaGovernor

__all__ = [
    "create_model",
    "TolariaGovernor",
    "GovernorReport",
]
```

**Test:** `uv run pytest tests/tolaria/ -v` (should only run governor tests)

---

### Task 4: Delete Dead Tests

**Depends on:** Task 3 (trainer.py must be deleted first)

Remove all tests that validate the dead trainer functions.

**Files to DELETE entirely:**
- `tests/tolaria/test_trainer.py` (937 lines)
- `tests/tolaria/test_lm_validation.py` (~50 lines)
- `tests/integration/test_tolaria_kasmina.py` (164 lines) - tests dead trainer+kasmina
- `tests/integration/test_tolaria_simic.py` (158 lines) - tests dead trainer+simic

**Files to MODIFY (remove dead imports):**
- `tests/integration/test_tamiyo_tolaria.py` - Remove trainer imports, or DELETE if all tests use dead code
- `tests/integration/test_tamiyo_simic.py` - Remove trainer imports, keep SignalTracker tests

**Verification:**
```bash
# Should find NO imports from tolaria.trainer
grep -r "from esper.tolaria.trainer import" tests/
grep -r "from esper.tolaria import.*train" tests/

# These should return empty
```

**Total lines deleted:** ~1,350

**Test:** `uv run pytest tests/ -v --ignore=tests/karn`

---

### Task 5: Add Perplexity to Telemetry (Nice-to-Have)

**Depends on:** Task 4 (dead code cleanup complete)

Add perplexity calculation for LM tasks to the production validation path.

**Files:**
- MODIFY: `src/esper/simic/training/vectorized.py` (in epoch summary)
- MODIFY: `src/esper/leyline/telemetry.py` (add perplexity field if needed)

**Implementation:**

```python
# In epoch summary computation (after validation)
if task_spec.task_type == "lm":
    perplexity = math.exp(val_loss) if val_loss < 20 else float("inf")
    # Add to telemetry payload
```

**Test:** `uv run pytest tests/simic/training/test_lm_perplexity.py -v`

---

### Task 6: Integration Verification

**Depends on:** Task 5 (all changes complete)

Full test suite and manual verification.

```bash
# Full test suite
uv run pytest tests/ -v

# Training smoke test
PYTHONPATH=src uv run python -m esper.scripts.train ppo --episodes 5 --n-envs 2

# Verify no dead imports remain
grep -r "from esper.tolaria.trainer" src/ tests/
# Should return NO results

# Verify tolaria module is minimal
ls src/esper/tolaria/
# Should only show: __init__.py, environment.py, governor.py
```

**Verification checklist:**
- [ ] All tests pass
- [ ] Training runs without errors
- [ ] Gradient clipping is active (check logs for gradient norms)
- [ ] No imports from `esper.tolaria.trainer` remain
- [ ] `trainer.py` is deleted
- [ ] ~1,350 lines of dead tests deleted

---

## Files Summary

### To DELETE (Total: ~1,750 lines)

| File | Lines | Reason |
|------|-------|--------|
| `src/esper/tolaria/trainer.py` | 433 | Dead training functions |
| `tests/tolaria/test_trainer.py` | 937 | Tests dead code |
| `tests/tolaria/test_lm_validation.py` | ~50 | Tests dead code |
| `tests/integration/test_tolaria_kasmina.py` | 164 | Tests dead code |
| `tests/integration/test_tolaria_simic.py` | 158 | Tests dead code (keep governor tests?) |

### To MODIFY

| File | Changes |
|------|---------|
| `src/esper/simic/training/config.py` | Add `max_grad_norm` parameter |
| `src/esper/simic/training/vectorized.py` | Add gradient clipping with proper AMP ordering |
| `src/esper/tolaria/__init__.py` | Remove trainer exports |
| `tests/integration/test_tamiyo_*.py` | Remove dead trainer imports |

### To CREATE

| File | Purpose |
|------|---------|
| `tests/simic/training/test_gradient_clipping.py` | Test new gradient clipping feature |

---

## Expert Review Findings

### DRL Expert (2025-12-25)

**Verdict:** Gradient clipping is CRITICAL for Transformer hosts.

Key points:
- CNN gradients (CIFAR-10) are naturally stable - low risk without clipping
- Transformer attention gradients can explode, especially early training
- STE architecture compounds risk: seed gradients reflect host residuals
- Recommended `max_grad_norm=1.0` for host/seed (vs PPO's 0.5)

### PyTorch Expert (2025-12-25)

**Verdict:** P0 priority. Also identified missing `unscale_()` call.

Key points:
- Current AMP flow skips `unscale_()` - can't clip at true FP32 magnitude
- Correct ordering: `backward() → unscale_() → clip_grad_norm_() → step() → update()`
- `clip_grad_norm_()` works fine inside torch.compile regions
- Avoid `.item()` on gradient norm tensor to prevent graph breaks

**Implementation Review (2025-12-25):**

| Issue | Severity | Resolution |
|-------|----------|------------|
| Unscale ordering | OK | Ordering is correct |
| Global vs per-group clipping | Minor | Using global clip for simplicity; document semantic choice |
| `step()` inf/NaN detection | OK | Works as designed after `unscale_()` |
| `has_grads` guard consistency | **Fixed** | Compute once, reuse for both unscale and step |
| Non-AMP clipping | OK | Same logic, just skip unscale |

---

## Rollback Plan

All changes are in a dedicated branch. If issues arise:

```bash
git checkout main
git branch -D fix/tolaria-remediation
```

No database migrations or external dependencies affected.

---

## Success Criteria

1. All tests pass
2. Training produces same or better results (gradient clipping improves stability)
3. No imports from `esper.tolaria.trainer` remain anywhere
4. `trainer.py` file is deleted
5. `TrainingConfig.max_grad_norm` controls host/seed gradient clipping
6. ~1,750 lines of dead code removed

---

## Architecture Diagram (Final)

```
┌───────────────────────────────────────────────────────────────────┐
│                            TOLARIA                                 │
│                    "Metabolism (Minimal)"                          │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ environment.py → create_model()                              │  │
│  │ governor.py    → TolariaGovernor (anomaly detection)        │  │
│  │                                                              │  │
│  │ (trainer.py DELETED - was dead code)                         │  │
│  └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
                                │
                                │ TolariaGovernor
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│                             SIMIC                                  │
│                    "Evolution (Production)"                        │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ training/vectorized.py                                       │  │
│  │   └── process_train_batch()                                  │  │
│  │       └── scaler.scale(loss).backward()                      │  │
│  │       └── scaler.unscale_(optimizers)      ← NEW            │  │
│  │       └── clip_grad_norm_(params, max_norm) ← NEW           │  │
│  │       └── scaler.step() / scaler.update()                    │  │
│  │                                                              │  │
│  │ training/config.py                                           │  │
│  │   └── TrainingConfig.max_grad_norm = 1.0   ← NEW            │  │
│  │                                                              │  │
│  │ attribution/counterfactual.py                                │  │
│  │   └── CounterfactualEngine (Shapley values)                  │  │
│  │   └── (Supersedes dead validate_with_attribution)            │  │
│  └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
```

---

## Notes

- The dead Tolaria trainer functions were architecturally correct (included gradient clipping) but implemented for the wrong paradigm (sequential single-env vs parallel multi-env)
- The fix is to port the critical feature (gradient clipping) to production, not to wire up dead code
- Counterfactual attribution in Simic is superior to Tolaria's simple two-pass approach (supports Shapley values)
- Perplexity is a nice-to-have for TUI/dashboard readability, not training stability

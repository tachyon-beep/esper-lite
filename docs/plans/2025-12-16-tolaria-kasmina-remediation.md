# Tolaria & Kasmina Remediation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix high/medium priority issues identified in PyTorch deep dive of Tolaria and Kasmina modules.

**Architecture:** Address serialization bugs, performance bottlenecks, and safety gaps while maintaining backward compatibility with existing checkpoints.

**Tech Stack:** PyTorch 2.x, Python 3.11+

---

## Issue Summary

| Priority | Issue | Task |
|----------|-------|------|
| HIGH | GatedBlend serialization - learned weights may not persist | Task 1 |
| HIGH | Governor rollback fails with active seeds | Task 2 |
| HIGH | Tensor allocation in blend hot path | Task 3 |
| MEDIUM | Missing gradient clipping | Task 4 |
| MEDIUM | isinstance branch hurts torch.compile | Task 5 |

---

## Task 1: Fix GatedBlend Serialization

### Decision Record: GatedBlend Checkpoint Persistence

**Date:** 2025-12-16
**Status:** RESOLVED - Root cause identified after multiple investigations
**Context:** We have flip-flopped on this issue multiple times. This section documents the full investigation to prevent future confusion.

---

### Investigation History

#### Initial Hypothesis (INCORRECT)

We initially believed the bug was:

1. `alpha_schedule` is assigned dynamically in `start_blending()` (slot.py:1253)
2. Dynamic assignment doesn't register with PyTorch's `_modules` dict
3. Therefore `load_state_dict(strict=True)` fails with orphan keys like `alpha_schedule.gate.0.weight`
4. Fix: Use `register_module()` explicitly

**This hypothesis was WRONG.**

#### PyTorch Expert Investigation (2025-12-16)

A dedicated PyTorch specialist verified the actual behavior:

**Finding 1: Dynamic assignment DOES work in PyTorch 2.x**

```python
# This DOES register the module in _modules automatically:
self.alpha_schedule = BlendCatalog.create(...)

# PyTorch's __setattr__ detects nn.Module and calls register_module internally.
# Verified with test: dynamic assignment == register_module for state_dict purposes.
```

**Finding 2: set_extra_state call order is BEFORE child enumeration**

The plan claimed: "child modules were enumerated BEFORE set_extra_state"

Reality (verified with instrumentation):
```
[0] load_state_dict START
[1] set_extra_state START, _modules=['main']
[2] After module creation in set_extra_state, _modules=['main', 'dynamic_child']
[3] Child module parameters NOW loaded successfully
```

`set_extra_state()` is called early enough that modules created within it WILL receive their weights from the checkpoint.

**Finding 3: The ACTUAL bug**

The real bug is in `set_extra_state()`:

```python
# Current code (BUGGY):
if config.get("algorithm_id") and self.state and self.state.stage == SeedStage.BLENDING:
    self.start_blending(total_steps=config.get("total_steps", 10))  # BUG!
```

`start_blending()` uses:
```python
algorithm_id = getattr(self, "_blend_algorithm_id", "sigmoid")  # Defaults to sigmoid!
```

**The `_blend_algorithm_id` is NEVER restored from the checkpoint config!**

So when loading a GatedBlend checkpoint:
1. `set_extra_state()` is called
2. `start_blending()` runs with `_blend_algorithm_id` defaulting to `"sigmoid"`
3. A SigmoidBlend is created instead of GatedBlend
4. The GatedBlend weights in the checkpoint become `unexpected_keys`
5. If `strict=False`, they're silently ignored. If `strict=True`, load fails.

---

### Approaches Considered

| Approach | Verdict | Rationale |
|----------|---------|-----------|
| **A. Use register_module()** | REJECTED | Unnecessary - PyTorch 2.x handles dynamic assignment. Also causes `KeyError: "attribute already exists"` if attribute was initialized to None in __init__. |
| **B. Override state_dict/load_state_dict** | REJECTED | Overly complex, bypasses PyTorch's standard module management, error-prone. |
| **C. Use __getstate__/__setstate__** | REJECTED | These are for pickle, not state_dict. Wrong abstraction level. |
| **D. Serialize GatedBlend separately** | REJECTED | Splits state across multiple mechanisms, complicates checkpoint handling. |
| **E. Fix _blend_algorithm_id restoration** | ACCEPTED | Simple one-line fix addressing the actual root cause. |

---

### Module Removal: del vs None

We also investigated how to properly "unregister" a module in `_on_blending_complete()`:

```python
# Option 1: del self._modules["alpha_schedule"]
# - Removes from _modules dict
# - ALSO deletes the attribute entirely (self.alpha_schedule becomes undefined)
# - Could cause AttributeError if code later checks self.alpha_schedule

# Option 2: self.alpha_schedule = None
# - Module stays in _modules dict as None
# - BUT PyTorch correctly excludes None modules from state_dict()
# - Attribute remains accessible (returns None)
```

**Decision:** Use `self.alpha_schedule = None` (current behavior). It's simpler and safer.

---

### The Fix

**Root Cause:** `_blend_algorithm_id` not restored from checkpoint before `start_blending()`.

**Fix:** Add one line to `set_extra_state()`:

```python
# Before:
if config.get("algorithm_id") and self.state and self.state.stage == SeedStage.BLENDING:
    self.start_blending(total_steps=config.get("total_steps", 10))

# After:
if config.get("algorithm_id") and self.state and self.state.stage == SeedStage.BLENDING:
    self._blend_algorithm_id = config["algorithm_id"]  # THE FIX
    self.start_blending(total_steps=config.get("total_steps", 10))
```

---

### Implementation

**Files:**
- Modify: `src/esper/kasmina/slot.py:1561-1568` (set_extra_state)
- Test: `tests/kasmina/test_slot_serialization.py` (new)

**Step 1: Write failing test for GatedBlend serialization**

```python
# tests/kasmina/test_slot_serialization.py
"""Tests for SeedSlot checkpoint serialization.

These tests verify the fix for GatedBlend checkpoint persistence.
See docs/plans/2025-12-16-tolaria-kasmina-remediation.md for full investigation.
"""

import tempfile
import torch
import torch.nn as nn
import pytest

from esper.kasmina.slot import SeedSlot, SeedState
from esper.kasmina.blending import GatedBlend, LinearBlend
from esper.leyline import SeedStage


class TestGatedBlendSerialization:
    """Test that GatedBlend learned weights survive checkpoint round-trip.

    The key test is that we do NOT set _blend_algorithm_id on the loading slot.
    It must be restored from the checkpoint config.
    """

    def test_gatedblend_algorithm_id_restored_from_checkpoint(self):
        """_blend_algorithm_id must be restored from checkpoint, not default to sigmoid."""
        # Setup: Create slot with GatedBlend
        slot = SeedSlot(slot_id="test", channels=64, device="cpu")
        slot._blend_algorithm_id = "gated"
        slot.seed = nn.Linear(64, 64)
        slot.state = SeedState(seed_id="test_seed", slot_id="test")
        slot.state.stage = SeedStage.BLENDING
        slot.start_blending(total_steps=10)

        assert isinstance(slot.alpha_schedule, GatedBlend), "Setup failed"

        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(slot.state_dict(), f.name)

            # Create fresh slot - DO NOT set _blend_algorithm_id
            # This tests that it's restored from checkpoint
            new_slot = SeedSlot(slot_id="test", channels=64, device="cpu")
            new_slot.seed = nn.Linear(64, 64)
            # Note: NOT setting new_slot._blend_algorithm_id

            state_dict = torch.load(f.name, weights_only=True)
            new_slot.load_state_dict(state_dict, strict=False)

        # The bug: without the fix, this would be SigmoidBlend
        assert isinstance(new_slot.alpha_schedule, GatedBlend), \
            f"Expected GatedBlend but got {type(new_slot.alpha_schedule).__name__}. " \
            "This means _blend_algorithm_id was not restored from checkpoint."

    def test_gatedblend_weights_persist_through_checkpoint(self):
        """GatedBlend gate network weights must survive save/load cycle."""
        slot = SeedSlot(slot_id="test", channels=64, device="cpu")
        slot._blend_algorithm_id = "gated"
        slot.seed = nn.Linear(64, 64)
        slot.state = SeedState(seed_id="test_seed", slot_id="test")
        slot.state.stage = SeedStage.BLENDING
        slot.start_blending(total_steps=10)

        # Modify weights to known values (simulates training)
        with torch.no_grad():
            for param in slot.alpha_schedule.gate.parameters():
                param.fill_(0.42)

        original_weights = {
            k: v.clone() for k, v in slot.alpha_schedule.state_dict().items()
        }

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(slot.state_dict(), f.name)

            new_slot = SeedSlot(slot_id="test", channels=64, device="cpu")
            new_slot.seed = nn.Linear(64, 64)

            state_dict = torch.load(f.name, weights_only=True)
            new_slot.load_state_dict(state_dict, strict=False)

        # Verify weights restored
        for key, original_value in original_weights.items():
            loaded_value = new_slot.alpha_schedule.state_dict()[key]
            torch.testing.assert_close(
                loaded_value, original_value,
                msg=f"GatedBlend weight {key} was not restored correctly"
            )

    def test_linear_blend_step_restored(self):
        """LinearBlend current_step should also round-trip correctly."""
        slot = SeedSlot(slot_id="test", channels=64, device="cpu")
        slot._blend_algorithm_id = "linear"
        slot.seed = nn.Linear(64, 64)
        slot.state = SeedState(seed_id="test_seed", slot_id="test")
        slot.state.stage = SeedStage.BLENDING
        slot.start_blending(total_steps=10)
        slot.alpha_schedule.step(5)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(slot.state_dict(), f.name)

            new_slot = SeedSlot(slot_id="test", channels=64, device="cpu")
            new_slot.seed = nn.Linear(64, 64)

            state_dict = torch.load(f.name, weights_only=True)
            new_slot.load_state_dict(state_dict, strict=False)

        assert new_slot.alpha_schedule._current_step == 5

    def test_non_blending_slot_loads_without_alpha_schedule(self):
        """Slot not in BLENDING stage should not create alpha_schedule on load."""
        slot = SeedSlot(slot_id="test", channels=64, device="cpu")
        slot.seed = nn.Linear(64, 64)
        slot.state = SeedState(seed_id="test_seed", slot_id="test")
        slot.state.stage = SeedStage.TRAINING  # Not BLENDING

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(slot.state_dict(), f.name)

            new_slot = SeedSlot(slot_id="test", channels=64, device="cpu")
            new_slot.seed = nn.Linear(64, 64)

            state_dict = torch.load(f.name, weights_only=True)
            new_slot.load_state_dict(state_dict, strict=False)

        assert new_slot.alpha_schedule is None
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/test_slot_serialization.py -v`
Expected: FAIL on `test_gatedblend_algorithm_id_restored_from_checkpoint` - gets SigmoidBlend instead of GatedBlend

**Step 3: Implement the fix**

Modify `src/esper/kasmina/slot.py` in `set_extra_state()` (around line 1561):

```python
def set_extra_state(self, state: dict) -> None:
    """Restore SeedState from primitive dict."""
    self.isolate_gradients = state.get("isolate_gradients", False)

    if state.get("seed_state"):
        self.state = SeedState.from_dict(state["seed_state"])

    # Alpha schedule reconstruction
    # The nn.Module weights are restored via load_state_dict() automatically
    # because PyTorch 2.x includes dynamically assigned modules in state_dict.
    # We only need to restore config and ensure the correct algorithm type.
    if state.get("alpha_schedule_config"):
        config = state["alpha_schedule_config"]
        if config.get("algorithm_id") and self.state and self.state.stage == SeedStage.BLENDING:
            # CRITICAL: Restore algorithm_id BEFORE start_blending()
            # Without this, start_blending() defaults to "sigmoid" and
            # GatedBlend weights become orphaned "unexpected_keys".
            # See: docs/plans/2025-12-16-tolaria-kasmina-remediation.md
            self._blend_algorithm_id = config["algorithm_id"]
            self.start_blending(total_steps=config.get("total_steps", 10))
            self.alpha_schedule._current_step = config.get("current_step", 0)
            # Re-restore blending_steps_done (start_blending resets it to 0)
            if self.state:
                self.state.blending_steps_done = state["seed_state"].get("blending_steps_done", 0)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/test_slot_serialization.py -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `PYTHONPATH=src uv run pytest tests/ -v --tb=short`
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/esper/kasmina/slot.py tests/kasmina/test_slot_serialization.py
git commit -m "fix(kasmina): restore _blend_algorithm_id from checkpoint in set_extra_state

GatedBlend learned weights were lost during checkpoint load because
set_extra_state() called start_blending() without first restoring
_blend_algorithm_id from the checkpoint config. This caused
start_blending() to default to 'sigmoid', creating the wrong blend
type and orphaning the GatedBlend weights.

Fix: Set self._blend_algorithm_id = config['algorithm_id'] before
calling start_blending().

See docs/plans/2025-12-16-tolaria-kasmina-remediation.md for full
investigation history (we flip-flopped on this multiple times)."
```

---

## Task 2: Fix Governor Rollback with Active Seeds

**Problem:** If `Governor.snapshot()` is called while experimental seeds are active, the snapshot contains seed parameters. Later, if those seeds are culled and rollback is triggered, `load_state_dict(strict=True)` fails because the snapshot has keys that don't exist in the current model.

**Files:**
- Modify: `src/esper/tolaria/governor.py:82-100`
- Test: `tests/tolaria/test_governor.py` (add test)

**Step 1: Write failing test**

```python
# Add to tests/tolaria/test_governor.py

def test_rollback_succeeds_after_seed_culled():
    """Governor rollback should succeed even if seeds were culled after snapshot."""
    from esper.kasmina.host import MorphogeneticModel
    from esper.tolaria.governor import TolariaGovernor
    from esper.leyline import SeedStage

    # Create model with an active seed
    model = create_test_model_with_slots()  # Helper that creates MorphogeneticModel
    slot = model.seed_slots["r0c0"]
    slot.germinate()  # Now has experimental seed

    gov = TolariaGovernor(model)  # Takes snapshot with seed

    # Cull the seed
    slot.cull("test")

    # Rollback should handle the key mismatch gracefully
    report = gov.execute_rollback()
    assert report.rollback_occurred
```

**Step 2: Implement fix - snapshot only host + fossilized**

```python
# In Governor.snapshot() (governor.py:82-100)
def snapshot(self) -> None:
    """Save Last Known Good state to CPU memory.

    Only snapshots host parameters and fossilized seeds. Experimental
    (non-fossilized) seeds are excluded because they may be culled
    before rollback, causing state_dict key mismatches.
    """
    if self.last_good_state is not None:
        del self.last_good_state
        self.last_good_state = None

    # Get model state, filtering out experimental seed keys
    full_state = self.model.state_dict()

    # If model has seed_slots, filter out non-fossilized seed parameters
    # hasattr AUTHORIZED by John on 2025-12-16 12:00:00 UTC
    # Justification: Feature detection - MorphogeneticModel has seed_slots, base models don't
    if hasattr(self.model, 'seed_slots'):
        experimental_prefixes = []
        for slot_id, slot in self.model.seed_slots.items():
            if slot.state is not None and slot.state.stage != SeedStage.FOSSILIZED:
                # This seed is experimental - exclude its keys from snapshot
                experimental_prefixes.append(f"seed_slots.{slot_id}.seed.")
                experimental_prefixes.append(f"seed_slots.{slot_id}.alpha_schedule.")

        # Filter state dict
        filtered_state = {
            k: v for k, v in full_state.items()
            if not any(k.startswith(prefix) for prefix in experimental_prefixes)
        }
    else:
        filtered_state = full_state

    with torch.no_grad():
        self.last_good_state = {
            k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
            for k, v in filtered_state.items()
        }
```

```python
# In execute_rollback(), change strict=True to strict=False:
    # Restore with strict=False since snapshot excludes experimental seeds
    # and current model may have different seed state
    self.model.load_state_dict(state_on_device, strict=False)
```

**Step 3: Run tests**

Run: `PYTHONPATH=src uv run pytest tests/tolaria/test_governor.py -v`

**Step 4: Commit**

```bash
git add src/esper/tolaria/governor.py tests/tolaria/test_governor.py
git commit -m "fix(governor): exclude experimental seeds from snapshot

Snapshots now only include host + fossilized seed parameters. This
prevents load_state_dict failures when seeds are culled between
snapshot and rollback."
```

---

## Task 3: Eliminate Tensor Allocation in Blend Hot Path

**Problem:** `get_alpha_for_blend()` creates a new tensor every forward pass.

**Files:**
- Modify: `src/esper/kasmina/blending.py:76-79, 101-104`
- Test: Existing blend tests should still pass

**Step 1: Write performance test (optional, for verification)**

```python
# tests/kasmina/test_blending_perf.py
def test_linear_blend_no_allocation_per_call():
    """LinearBlend should reuse cached alpha tensor."""
    blend = LinearBlend(total_steps=10)
    blend.step(5)

    x = torch.randn(2, 64, 32, 32)

    # Get alpha twice
    alpha1 = blend.get_alpha_for_blend(x)
    alpha2 = blend.get_alpha_for_blend(x)

    # Should be same tensor (cached)
    assert alpha1.data_ptr() == alpha2.data_ptr(), \
        "Alpha tensor should be cached, not recreated"
```

**Step 2: Implement caching in BlendAlgorithm base class**

```python
# In BlendAlgorithm (blending.py:15-61)
class BlendAlgorithm(ABC, nn.Module):
    """Base class for blending algorithms."""

    algorithm_id: str = "base"
    _current_step: int = 0

    def __init__(self):
        super().__init__()
        # Cache for alpha tensor to avoid per-forward allocation
        # Format: (device, dtype, value, tensor)
        self._alpha_cache: tuple[torch.device, torch.dtype, float, torch.Tensor] | None = None

    def _get_cached_alpha_tensor(self, value: float, x: torch.Tensor) -> torch.Tensor:
        """Get alpha tensor, using cache if possible."""
        if self._alpha_cache is not None:
            cached_device, cached_dtype, cached_value, cached_tensor = self._alpha_cache
            if cached_device == x.device and cached_dtype == x.dtype and cached_value == value:
                return cached_tensor

        # Create new tensor and cache it
        tensor = torch.tensor(value, device=x.device, dtype=x.dtype)
        self._alpha_cache = (x.device, x.dtype, value, tensor)
        return tensor
```

```python
# In LinearBlend.get_alpha_for_blend (blending.py:76-79)
def get_alpha_for_blend(self, x: torch.Tensor) -> torch.Tensor:
    """Return scalar alpha as 0-dim tensor (broadcasts to any shape)."""
    alpha = min(1.0, max(0.0, self._current_step / self.total_steps))
    return self._get_cached_alpha_tensor(alpha, x)
```

```python
# In SigmoidBlend.get_alpha_for_blend (blending.py:101-104)
def get_alpha_for_blend(self, x: torch.Tensor) -> torch.Tensor:
    """Return scalar alpha as 0-dim tensor (broadcasts to any shape)."""
    alpha = self._compute_alpha(self._current_step)
    return self._get_cached_alpha_tensor(alpha, x)
```

**Step 3: Run tests**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/ -v -k blend`

**Step 4: Commit**

```bash
git add src/esper/kasmina/blending.py
git commit -m "perf(blending): cache alpha tensor to avoid per-forward allocation

LinearBlend and SigmoidBlend now cache their alpha tensors. Cache is
invalidated only when device, dtype, or alpha value changes. This
eliminates thousands of tensor allocations per episode in vectorized RL."
```

---

## Task 4: Add Optional Gradient Clipping to Training Functions

**Problem:** No gradient clipping - risk of exploding gradients.

**Files:**
- Modify: `src/esper/tolaria/trainer.py:88-205`
- Test: `tests/tolaria/test_trainer.py`

**Step 1: Add max_grad_norm parameter to training functions**

```python
# Modify train_epoch_normal signature and body
def train_epoch_normal(
    model: nn.Module,
    trainloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    task_type: str = "classification",
    max_grad_norm: float | None = None,  # NEW: optional gradient clipping
) -> None:
    model.train()
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = compute_task_loss(outputs, labels, criterion, task_type)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
```

Apply same pattern to `train_epoch_incubator_mode` and `train_epoch_blended`.

**Step 2: Run tests**

Run: `PYTHONPATH=src uv run pytest tests/tolaria/test_trainer.py -v`

**Step 3: Commit**

```bash
git add src/esper/tolaria/trainer.py tests/tolaria/test_trainer.py
git commit -m "feat(trainer): add optional gradient clipping parameter

All training functions now accept max_grad_norm parameter. When set,
gradients are clipped before optimizer step. Default None preserves
existing behavior."
```

---

## Task 5: Eliminate isinstance Branch in blend_with_isolation

**Problem:** `isinstance(alpha, torch.Tensor)` check causes torch.compile to generate two graph variants.

**Files:**
- Modify: `src/esper/kasmina/isolation.py:54-60`
- Modify: Callers to always pass tensors

**Step 1: Simplify blend_with_isolation to always expect tensor**

```python
# In isolation.py, change blend_with_isolation:
def blend_with_isolation(
    host_features: torch.Tensor,
    seed_features: torch.Tensor,
    alpha: torch.Tensor,  # Now required to be tensor
) -> torch.Tensor:
    """Blend host and seed features with gradient-safe alpha clamping.

    Args:
        alpha: Must be a tensor (use torch.tensor(float_value) if needed).
              This requirement enables torch.compile to generate a single
              optimized graph without type-checking branches.
    """
    alpha = torch.clamp(alpha, 0.0, 1.0)
    return torch.lerp(host_features, seed_features, alpha)
```

**Step 2: Update SeedSlot.forward to ensure tensor alpha**

The current code already uses `self.alpha_schedule.get_alpha_for_blend(host_features)` which returns a tensor, or falls back to `self.state.alpha` which is a float. Update the fallback path:

```python
# In SeedSlot.forward, the blend path should ensure tensor alpha
alpha_tensor = self.alpha_schedule.get_alpha_for_blend(host_features) if self.alpha_schedule else \
               torch.tensor(self.state.alpha, device=host_features.device, dtype=host_features.dtype)
return blend_with_isolation(host_features, seed_features, alpha_tensor)
```

**Step 3: Run tests**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/ -v`

**Step 4: Commit**

```bash
git add src/esper/kasmina/isolation.py src/esper/kasmina/slot.py
git commit -m "perf(isolation): remove isinstance branch for torch.compile

blend_with_isolation now requires alpha to be a tensor. This eliminates
a runtime type check that caused torch.compile to generate multiple
graph variants. Callers updated to always provide tensor alpha."
```

---

## Verification

After all tasks complete:

```bash
# Full test suite
PYTHONPATH=src uv run pytest tests/ -v

# Verify torch.compile works (if GPU available)
PYTHONPATH=src python -c "
import torch
from esper.kasmina.slot import SeedSlot
slot = SeedSlot('test', 64)
slot.germinate()
slot.advance_stage()  # TRAINING
compiled = torch.compile(slot)
x = torch.randn(2, 64, 32, 32)
out = compiled(x)
print('torch.compile test passed')
"
```

---

## Appendix: PyTorch Module Registration Deep Dive

For future reference, here's what we learned about PyTorch 2.x module registration:

### Dynamic Assignment Works

```python
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.dynamic_child = None  # Initially None

    def add_child(self):
        self.dynamic_child = nn.Linear(10, 10)  # This DOES register!
```

PyTorch's `__setattr__` detects `nn.Module` assignments and automatically adds them to `_modules`. You do NOT need `register_module()` for this to work.

### set_extra_state Timing

`set_extra_state()` is called DURING `load_state_dict()`, BEFORE child modules are enumerated. This means you CAN create modules in `set_extra_state()` and they WILL receive their weights from the checkpoint.

### Module Removal

Setting a module attribute to `None` is sufficient:
```python
self.child = None  # Module stays in _modules as None, excluded from state_dict
```

No need for `del self._modules["child"]` which also deletes the attribute itself.

### strict=False Considerations

When using `strict=False` with `load_state_dict()`:
- Missing keys: Module parameters not in checkpoint (OK for new features)
- Unexpected keys: Checkpoint has keys not in model (potential data loss!)

Always verify unexpected_keys when using `strict=False` to catch bugs.

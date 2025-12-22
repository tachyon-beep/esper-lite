# Deep Code Review: simic/training/helpers.py

**File**: `/home/john/esper-lite/src/esper/simic/training/helpers.py`
**Lines**: 684
**Reviewer**: DRL Expert Agent
**Date**: 2025-12-17

---

## Executive Summary

`helpers.py` contains two main training pathways:
1. **Compiled Training Step** (lines 29-116): A `torch.compile`-optimized forward/backward step
2. **Heuristic Training** (lines 213-685): The baseline training loop using rule-based policy decisions

**Overall Assessment**: The code is well-structured with good performance optimizations (tensor accumulation, async gradient collection, deferred CUDA sync). However, there are several correctness issues around gradient collection timing, potential training stability concerns with optimizer recreation, and some best-practice violations that could impact training reproducibility.

### Key Findings Summary

| Severity | Count | Summary |
|----------|-------|---------|
| Critical | 2 | Gradient stats overwrite, missing gradient clipping |
| High | 3 | Optimizer recreation, seed_optimizer scope, LM accuracy metric |
| Medium | 5 | Telemetry gating, action count tracking, validation early-stop, batch limit consistency, epoch indexing |
| Low | 4 | Code organization, type hints, docstring accuracy, dead code |

---

## Critical Issues

### C1. Gradient Statistics Overwritten Each Batch (Line 190-191)

**Location**: `_train_one_epoch()`, lines 188-191

```python
# Collect gradient stats as tensors (async-safe, no .item() sync)
# Overwrites each batch; final value materialized after loop
if collect_gradients:
    grad_stats = collect_seed_gradients_async(model.get_seed_parameters())
```

**Problem**: The comment acknowledges this overwrites each batch, but the caller likely expects gradient statistics aggregated across the entire epoch, not just the final batch. This is problematic because:

1. **Statistical Bias**: The final batch may not be representative of epoch-wide gradient behavior
2. **Gradient Health Misdiagnosis**: Vanishing/exploding gradient detection based on a single batch can produce false positives/negatives
3. **Signal-to-Noise Ratio**: Early batches may have critical gradient pathology that gets masked

**RL Impact**: Gradient health metrics feed into seed lifecycle decisions (G2 gate). Using only final-batch statistics could cause:
- Premature culling if final batch has anomalous gradients
- Missed early warning signs of training instability

**Recommendation**: Implement running statistics accumulation:
```python
# In loop:
if collect_gradients:
    batch_grad_stats = collect_seed_gradients_async(model.get_seed_parameters())
    if grad_stats is None:
        grad_stats = batch_grad_stats
    else:
        # Accumulate: running max for explosion, running min for vanishing
        grad_stats = _merge_grad_stats(grad_stats, batch_grad_stats)
```

---

### C2. Missing Gradient Clipping in Training Loops (Lines 170-210, 370-393)

**Location**: Both `_train_one_epoch()` and `run_heuristic_episode()` training loops

**Problem**: Neither training loop implements gradient clipping before `optimizer.step()`. This is a critical omission for RL-guided training where:

1. **Seed gradient explosion**: New seeds can have unstable gradients early in training
2. **Host-seed interaction**: Gradient isolation doesn't prevent explosion within the seed's own parameters
3. **Catastrophic forgetting**: Large gradient updates during BLENDING can destabilize the host

**Code Path Without Clipping**:
```python
loss.backward()
# <-- No clip_grad_norm_ here!
host_optimizer.step()
if seed_optimizer:
    seed_optimizer.step()
```

**RL Impact**: PPO training stability depends on bounded gradient updates. The simic module's own PPO implementation (`ppo.py`) likely has clipping, but the heuristic baseline in `helpers.py` does not, leading to:
- Non-comparable results between heuristic and PPO baselines
- Potential training divergence in heuristic mode

**Recommendation**: Add gradient clipping with standard RL values:
```python
torch.nn.utils.clip_grad_norm_(model.get_host_parameters(), max_norm=1.0)
if seed_optimizer and model.has_active_seed:
    torch.nn.utils.clip_grad_norm_(model.get_seed_parameters(), max_norm=1.0)
```

---

## High-Priority Issues

### H1. Optimizer Recreation on Every Action (Lines 517-521, 536-539)

**Location**: `run_heuristic_episode()`, optimizer handling after GERMINATE and CULL

```python
# After GERMINATE:
seed_optimizer = torch.optim.SGD(
    model.get_seed_parameters(), lr=task_spec.seed_lr, momentum=0.9
)

# After CULL:
seed_optimizer = (
    torch.optim.SGD(model.get_seed_parameters(), lr=task_spec.seed_lr, momentum=0.9)
    if model.has_active_seed else None
)
```

**Problem**: Recreating the optimizer discards momentum buffers, which:
1. **Resets momentum**: SGD momentum (0.9) takes several batches to rebuild
2. **Training discontinuity**: Sudden loss spikes after seed lifecycle transitions
3. **Non-optimal baseline**: Heuristic results are artificially worse due to optimizer state loss

**RL Impact**: If the heuristic baseline underperforms due to implementation issues (not algorithmic limitations), the comparison with PPO is unfair, potentially overestimating PPO's value.

**Recommendation**: Cache and restore optimizer state for seeds that survive CULL operations:
```python
# Before transition:
_cached_optimizer_state = seed_optimizer.state_dict() if seed_optimizer else None

# After operation that preserves seeds:
if seed_optimizer and _cached_optimizer_state:
    # Filter state to only surviving parameters
    seed_optimizer.load_state_dict(_cached_optimizer_state, strict=False)
```

---

### H2. seed_optimizer Scope Issue (Lines 348, 518-521, 536-539)

**Location**: `run_heuristic_episode()`

```python
seed_optimizer = None  # Line 348 - initialized to None

# Much later in loop...
if factored_action.is_germinate:
    if germinate_slot is not None:
        # ...
        seed_optimizer = torch.optim.SGD(...)  # Line 518-520
```

**Problem**: `seed_optimizer` is created inside the epoch loop but used in subsequent epochs. If GERMINATE fails (e.g., no available slot) but `seed_optimizer` was already set from a previous seed, the old optimizer continues to be used for a non-existent seed's parameters.

**Scenario**:
1. Epoch 1: GERMINATE succeeds, seed_optimizer created
2. Epoch 5: Seed fossilizes
3. Epoch 6: GERMINATE attempted but fails (no slot)
4. Epoch 7: `seed_optimizer.step()` called on stale parameter references

**RL Impact**: Silent training corruption - gradients applied to wrong parameters or detached tensors.

**Recommendation**: Explicitly reset `seed_optimizer = None` when seeds are fossilized/pruned:
```python
elif factored_action.is_fossilize:
    # ... fossilize logic ...
    if gate_result.passed:
        seed_optimizer = None  # Seed is now part of host

elif factored_action.is_cull:
    model.cull_seed(slot=target_slot)
    # Update optimizer for remaining seeds (or None)
    seed_optimizer = (...)
```

---

### H3. LM Accuracy Metric Returns Zero (Lines 183-185)

**Location**: `_train_one_epoch()`, lines 183-185

```python
else:  # LM task
    correct_batch = torch.tensor(0, device=outputs.device)
    batch_total = targets.numel()
```

**Problem**: For language modeling tasks, `correct_batch` is always zero, meaning:
- `train_acc` will always be 0.0 for LM tasks
- Telemetry and reward signals based on accuracy are meaningless

**Context**: The validation loop in `run_heuristic_episode()` uses `compute_task_loss_with_metrics()` which does compute token-level accuracy for LM. But `_train_one_epoch()` does not.

**RL Impact**: The `train_acc` value is passed to `signal_tracker.update()` and influences plateau detection. For LM tasks, plateau detection will never trigger based on training accuracy.

**Recommendation**: Use `compute_task_loss_with_metrics()` consistently:
```python
loss, correct_batch, batch_total = compute_task_loss_with_metrics(
    outputs, targets, criterion, task_type
)
```

---

## Medium-Priority Issues

### M1. Telemetry Gating Logic Complexity (Lines 314-329)

**Location**: `run_heuristic_episode()`, slot telemetry configuration

```python
slot.fast_mode = not ops_telemetry_enabled
slot.telemetry_lifecycle_only = telemetry_lifecycle_only and not ops_telemetry_enabled
slot.on_telemetry = (
    telemetry_callback
    if ops_telemetry_enabled or telemetry_lifecycle_only
    else None
)
```

**Problem**: The boolean logic is complex and potentially contradictory:
- `telemetry_lifecycle_only and not ops_telemetry_enabled` - only True if lifecycle_only=True AND ops disabled
- But `on_telemetry` is set if `ops_telemetry_enabled OR telemetry_lifecycle_only`

This creates a state where `telemetry_lifecycle_only=True` still sets `on_telemetry` even when ops are disabled, but the flag itself is False due to the AND condition.

**Recommendation**: Simplify with explicit states or a telemetry mode enum.

---

### M2. Action Count Tracking Uses Enum Names (Line 353)

**Location**: `run_heuristic_episode()`, line 353

```python
action_counts = {op.name: 0 for op in LifecycleOp}
```

**Problem**: Action counts are tracked by string name, but incremented using `factored_action.op.name` (line 480). If `LifecycleOp` enum values change, the string keys remain the same but the mapping breaks silently.

**Better Pattern**:
```python
action_counts = {op: 0 for op in LifecycleOp}  # Use enum as key
# At reporting time:
action_counts_serializable = {op.name: count for op, count in action_counts.items()}
```

---

### M3. Validation Early Stop Not Applied Consistently (Lines 406-408)

**Location**: `run_heuristic_episode()`, validation loop

```python
for inputs, targets in testloader:
    if max_batches and batch_count >= max_batches:
        break
```

**Problem**: `max_batches` is an optional parameter with a default of 50 in `train_heuristic()` (line 575), but validation should typically run on the full test set for accurate metrics. Using `max_batches` for both train and validation produces:
- Noisy accuracy estimates
- Poor seed lifecycle decisions based on incomplete validation

**Recommendation**: Add a separate `max_val_batches` parameter or default to full validation.

---

### M4. Inconsistent Batch Limiting (Lines 371-373 vs 406-408)

**Location**: Training loop uses `batch_count` zero-indexed check, incremented after check

```python
for inputs, targets in trainloader:
    if max_batches and batch_count >= max_batches:
        break
    batch_count += 1
```

**Problem**: `batch_count` starts at 0, so `batch_count >= max_batches` triggers at exactly `max_batches` iterations. This is correct but the pattern differs from typical `range(max_batches)` semantics.

More importantly, `batch_count` is used in loss normalization (line 395):
```python
train_loss = running_loss.item() / max(1, batch_count)
```

If the loop processes exactly `max_batches` batches, `batch_count == max_batches` is correct. But if the dataloader has fewer batches than `max_batches`, the division is still correct. No bug, but the logic could be clearer.

---

### M5. Epoch Indexing Starts at 1 (Line 358)

**Location**: `run_heuristic_episode()`, line 358

```python
for epoch in range(1, max_epochs + 1):
```

**Problem**: 1-indexed epochs are used throughout, but PBRS calculations in rewards.py use `epochs_in_stage` which is 0-indexed internally. This mismatch can cause off-by-one errors in:
- Terminal bonus calculation (line 660 in rewards.py checks `epoch == max_epochs`)
- PBRS epoch progress bonus

**Note**: The current implementation appears consistent, but the mixed indexing is error-prone for future changes.

---

## Low-Priority Suggestions

### L1. `_train_one_epoch()` Not Used by Heuristic Path

**Location**: Lines 123-210

**Observation**: `_train_one_epoch()` is a unified training loop, but `run_heuristic_episode()` implements its own training loop (lines 364-396) instead of calling `_train_one_epoch()`. This duplication means:
- Bug fixes must be applied twice
- Gradient collection behavior differs between paths
- The `collect_gradients` parameter is unused in heuristic mode

**Recommendation**: Refactor `run_heuristic_episode()` to use `_train_one_epoch()` or document why the duplication is intentional.

---

### L2. Incomplete Type Hints

**Location**: Multiple function signatures

```python
def run_heuristic_episode(
    policy,  # No type hint
    trainloader,  # No type hint
    testloader,  # No type hint
    ...
)
```

**Recommendation**: Add proper type hints for better IDE support and static analysis:
```python
from esper.tamiyo import HeuristicTamiyo

def run_heuristic_episode(
    policy: HeuristicTamiyo,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    ...
)
```

---

### L3. Docstring Parameter Mismatch

**Location**: `run_heuristic_episode()`, lines 277-289

```python
def run_heuristic_episode(
    ...
    slots: list[str] | None = None,
    telemetry_config: TelemetryConfig | None = None,
    telemetry_lifecycle_only: bool = False,
) -> tuple[float, dict[str, int], list[float]]:
    """Run a single training episode with heuristic policy.

    Args:
        policy: HeuristicTamiyo instance
        ...
        task_spec: Task specification

    Returns:
        (final_accuracy, action_counts, episode_rewards)
    """
```

**Problem**: Docstring Args section is incomplete - missing `slots`, `telemetry_config`, and `telemetry_lifecycle_only`.

---

### L4. `__all__` Export Incomplete (Lines 681-684)

**Location**: Module exports

```python
__all__ = [
    "run_heuristic_episode",
    "train_heuristic",
]
```

**Problem**: The module exports only 2 functions, but `compiled_train_step`, `_train_one_epoch`, and `_convert_flat_to_factored` are potentially useful to other modules. Either:
- Mark them as truly private (`_` prefix is already there for some)
- Export them if intended for external use

---

## Summary

The `helpers.py` file provides essential training loop infrastructure for the heuristic baseline. While the code demonstrates good performance awareness (tensor accumulation, async gradient collection), the critical issues around gradient statistics and missing gradient clipping should be addressed before using heuristic results as a valid baseline comparison for PPO.

### Priority Remediation Order

1. **C2**: Add gradient clipping (training stability)
2. **C1**: Fix gradient stats aggregation (correct telemetry)
3. **H1/H2**: Fix optimizer lifecycle management (training correctness)
4. **H3**: Fix LM accuracy metric (task compatibility)

### Positive Patterns Observed

- Proper CUDA sync deferral with tensor accumulation
- Good use of `non_blocking=True` for async transfers
- Clean separation between training step compilation and control flow
- Comprehensive telemetry integration with Nissa hub

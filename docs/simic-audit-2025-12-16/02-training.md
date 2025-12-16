# Simic Training Module Audit Report

**File:** `/home/john/esper-lite/src/esper/simic/training.py`
**Date:** 2025-12-16
**Auditor:** Claude (PyTorch Engineering Specialist)
**Lines:** 657

---

## Executive Summary

The `training.py` module provides compiled training steps and heuristic episode training for the Esper seed lifecycle system. Overall code quality is good with proper tensor accumulation patterns for CUDA efficiency. However, several issues warrant attention:

- **P1:** Global mutable state for torch.compile flag creates process-wide side effects
- **P2:** `_train_one_epoch` helper is defined but NOT USED by `run_heuristic_episode`
- **P2:** Potential device mismatch in LM path tensor creation
- **P3:** Missing guard for dynamic shapes in compiled training step

---

## 1. torch.compile Analysis

### 1.1 Compilation Strategy

**Location:** Lines 27-88

```python
USE_COMPILED_TRAIN_STEP = True  # Global mutable flag

try:
    _compiled_train_step = torch.compile(_train_step_impl, mode="reduce-overhead")
except Exception:
    _compiled_train_step = _train_step_impl
    USE_COMPILED_TRAIN_STEP = False
```

**Severity: P1 - High**

**Issues:**

1. **Global mutable state** - The `USE_COMPILED_TRAIN_STEP` flag is module-level state that:
   - Gets mutated if compilation fails during import
   - Can be mutated by external code (tests, debugging)
   - Affects ALL callers in the process, not just the current training run
   - Creates non-deterministic behavior in multi-worker/DDP scenarios

2. **Bare except clause** - Catches all exceptions including `KeyboardInterrupt`, `SystemExit`

3. **Compilation mode may not match use case** - `mode="reduce-overhead"` enables CUDA graph capture, which requires:
   - Static input shapes (no dynamic batching)
   - Warm-up runs before measurement
   - No graph breaks in the compiled region

**Known Bug:** This is tracked as `JANK-001-P2` in `/home/john/esper-lite/docs/bugs/JANK-001-P2-simic-global-compile-flag.md`

**Recommendation:**
```python
# Move to per-instance configuration
@dataclass
class CompileConfig:
    enabled: bool = True
    mode: str = "default"  # Safer than reduce-overhead for varying shapes

def get_train_step(config: CompileConfig | None = None):
    if config is None or not config.enabled:
        return _train_step_impl
    return torch.compile(_train_step_impl, mode=config.mode)
```

### 1.2 Graph Break Risk Assessment

**Location:** Lines 35-62

```python
def _train_step_impl(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(inputs)
    if outputs.dim() == 3:  # Dynamic control flow - potential graph break
        vocab = outputs.size(-1)
        loss = criterion(outputs.view(-1, vocab), targets.view(-1))
    else:
        loss = criterion(outputs, targets)
    return loss, outputs
```

**Severity: P3 - Low**

**Issues:**

1. The `if outputs.dim() == 3` branch introduces data-dependent control flow
2. With `mode="reduce-overhead"`, the first execution path determines the CUDA graph
3. If both classification and LM tasks use the same compiled function, the second task type will trigger recompilation or graph break

**PyTorch 2.9 Recommendation:**
```python
# Use error_on_graph_break() for diagnosis
with torch._dynamo.error_on_graph_break():
    compiled_fn(inputs)

# Or use separate compiled functions per task type
_compiled_lm_step = torch.compile(_lm_step_impl, mode="default")
_compiled_cls_step = torch.compile(_cls_step_impl, mode="default")
```

---

## 2. Device Placement Analysis

### 2.1 Proper Device Placement Patterns

**Location:** Lines 143-145

```python
for inputs, targets in trainloader:
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
```

**Status: GOOD** - Proper non-blocking transfers with explicit device argument.

### 2.2 Device Mismatch Risk in LM Path

**Location:** Lines 156-158 in `_train_one_epoch`

```python
else:  # LM task
    correct_batch = torch.tensor(0, device=outputs.device)
    batch_total = targets.numel()
```

**Severity: P2 - Medium**

**Issue:** The tensor is created using `outputs.device` but accumulated into `running_correct` which uses the `device` parameter:

```python
running_correct = torch.zeros(1, device=device, dtype=torch.long)
# ...
running_correct.add_(correct_batch)  # Potential cross-device add
```

If `outputs.device` differs from `device` (e.g., model partially on different GPU), this could cause:
- Silent data transfer (performance hit)
- Error in strict device checking mode

**Recommendation:**
```python
correct_batch = torch.tensor(0, device=device, dtype=torch.long)
```

### 2.3 Tensor Accumulation Pattern

**Location:** Lines 136-182

```python
# Pre-allocate accumulators on device to avoid .item() sync per batch
running_loss = torch.zeros(1, device=device)
running_correct = torch.zeros(1, device=device, dtype=torch.long)
# ...
running_loss.add_(loss.detach())
running_correct.add_(correct_batch)
# ...
# Single sync at epoch end
epoch_loss = running_loss.item()
epoch_correct = running_correct.item()
```

**Status: EXCELLENT** - This is the correct pattern for avoiding per-batch CUDA synchronization. The single `.item()` call at epoch end is optimal.

---

## 3. Gradient Flow Analysis

### 3.1 Gradient Collection Path

**Location:** Lines 161-168

```python
if collect_gradients:
    grad_stats = collect_seed_gradients_async(model.get_seed_parameters())

host_optimizer.step()
if seed_optimizer:
    seed_optimizer.step()
```

**Status: GOOD** - Gradients are collected AFTER `backward()` and BEFORE `step()`, which is correct timing.

### 3.2 Gradient Clearing Strategy

**Location:** Lines 146-148

```python
host_optimizer.zero_grad(set_to_none=True)
if seed_optimizer:
    seed_optimizer.zero_grad(set_to_none=True)
```

**Status: EXCELLENT** - Using `set_to_none=True` is the PyTorch 2.x best practice:
- Reduces memory footprint (no zero tensor allocation)
- Faster than zeroing memory
- Compatible with compiled training

### 3.3 Async Gradient Stats Materialization

**Location:** Lines 179-181

```python
# Now safe to materialize gradient tensors (after implicit sync above)
if grad_stats is not None and not grad_stats.get('_empty', False):
    grad_stats = materialize_grad_stats(grad_stats)
```

**Status: GOOD** - The materialization happens after `.item()` calls which force CUDA sync.

---

## 4. Memory Management Analysis

### 4.1 Tensor Detachment

**Location:** Line 171

```python
running_loss.add_(loss.detach())
```

**Status: EXCELLENT** - Critical for preventing gradient graph retention. Without `.detach()`, the loss tensor would keep the entire computation graph in memory.

### 4.2 In-Place Operations

**Location:** Lines 171-172

```python
running_loss.add_(loss.detach())
running_correct.add_(correct_batch)
```

**Status: GOOD** - In-place `.add_()` is correct here because:
- These tensors are not part of the autograd graph
- Reduces temporary allocations

### 4.3 Memory Leak Risk: `grad_stats` Overwrite

**Location:** Lines 163-164

```python
if collect_gradients:
    grad_stats = collect_seed_gradients_async(model.get_seed_parameters())
```

**Severity: P3 - Low**

**Issue:** Each batch overwrites the previous `grad_stats` dict. If `collect_seed_gradients_async` allocates new tensors, the old tensors become garbage but may not be freed immediately if CUDA caching allocator holds them.

**Impact:** Minor memory pressure in long epochs. The comment at line 161-162 acknowledges this is intentional ("Overwrites each batch; final value materialized after loop").

---

## 5. Integration Risk Analysis

### 5.1 Dead Code: `_train_one_epoch` Not Used

**Severity: P2 - Medium**

**Finding:** The `_train_one_epoch` helper (lines 96-183) is:
- Defined and tested (`tests/simic/test_training_helper.py`)
- Exported (implicitly via module)
- **NOT used by `run_heuristic_episode`** (lines 236-541)

`run_heuristic_episode` duplicates the training loop inline (lines 336-369) with subtle differences:
- Different metric computation
- Different batch limiting logic (`max_batches` parameter)
- Manual `compute_task_loss_with_metrics` instead of `compiled_train_step`

**Evidence:**
```python
# _train_one_epoch at line 150:
loss, outputs = compiled_train_step(model, inputs, targets, criterion)

# run_heuristic_episode at line 355:
loss, correct_batch, batch_total = compute_task_loss_with_metrics(outputs, targets, criterion, task_type)
```

**Risk:**
1. Code duplication means bugs must be fixed in two places
2. Performance differences (heuristic path doesn't use torch.compile)
3. Subtle behavioral divergence over time

**Archived Plan Reference:** `/home/john/esper-lite/docs/plans/archive/2025-12-09-train-helper-integration.md` describes an integration plan that was never completed.

### 5.2 Model Interface Assumptions

**Location:** Lines 164, 329, 596-604 (cross-reference to kasmina/host.py)

```python
# training.py line 164
grad_stats = collect_seed_gradients_async(model.get_seed_parameters())

# training.py line 329
host_params = sum(p.numel() for p in model.get_host_parameters() if p.requires_grad)
```

**Dependencies:**
- `model.get_seed_parameters()` - Generator yielding seed params
- `model.get_host_parameters()` - Generator yielding host params
- `model.seed_slots[slot_id]` - Dict-like access to SeedSlot objects
- `model.has_active_seed_in_slot(slot_id)` - Boolean check
- `model.germinate_seed()`, `model.cull_seed()` - Lifecycle operations

**Status: OK** - These are defined in `/home/john/esper-lite/src/esper/kasmina/host.py` and form a stable interface.

### 5.3 Telemetry Callback Wiring

**Location:** Lines 289-303

```python
def telemetry_callback(event):
    event.data.setdefault("env_id", 0)
    event.data.setdefault("device", device)
    hub.emit(event)

for slot_id in enabled_slots:
    slot = model.seed_slots[slot_id]
    slot.fast_mode = not ops_telemetry_enabled
    slot.telemetry_lifecycle_only = telemetry_lifecycle_only and not ops_telemetry_enabled
    slot.on_telemetry = (
        telemetry_callback
        if ops_telemetry_enabled or telemetry_lifecycle_only
        else None
    )
```

**Status: GOOD** - Proper conditional wiring of telemetry based on config level.

### 5.4 `__all__` Export List

**Location:** Lines 653-656

```python
__all__ = [
    "run_heuristic_episode",
    "train_heuristic",
]
```

**Severity: P3 - Low**

**Issue:** Missing exports:
- `compiled_train_step` - Used externally? Unclear.
- `_train_one_epoch` - Intended for extraction but currently dead code
- `USE_COMPILED_TRAIN_STEP` - Accessed by tests

**Recommendation:** Either add to `__all__` or document as private via `_` prefix.

---

## 6. Code Quality Analysis

### 6.1 Type Annotations

**Status: GOOD** - Comprehensive type hints throughout:

```python
def run_heuristic_episode(
    policy,
    trainloader,
    testloader,
    max_epochs: int = 75,
    max_batches: int | None = None,
    base_seed: int = 42,
    device: str = "cuda:0",
    task_spec=None,
    slots: list[str] | None = None,
    telemetry_config: TelemetryConfig | None = None,
    telemetry_lifecycle_only: bool = False,
) -> tuple[float, dict[str, int], list[float]]:
```

**Minor Issue:** `policy` and `trainloader`/`testloader` lack type hints. Consider:
```python
from esper.tamiyo import HeuristicTamiyo
from torch.utils.data import DataLoader

def run_heuristic_episode(
    policy: HeuristicTamiyo,
    trainloader: DataLoader,
    testloader: DataLoader,
    ...
```

### 6.2 Error Handling

**Location:** Lines 278-281

```python
if not slots:
    raise ValueError("slots parameter is required and cannot be empty")
if len(slots) != len(set(slots)):
    raise ValueError(f"slots contains duplicates: {slots}")
```

**Status: EXCELLENT** - Fail-fast validation with clear error messages.

**Location:** Lines 406-408

```python
if state.seed_id in seed_ids:
    raise RuntimeError(f"Duplicate seed_id '{state.seed_id}' across slots in one env")
```

**Status: GOOD** - Runtime invariant checking.

### 6.3 Documentation

**Status: GOOD** - Docstrings present on public functions with Args/Returns sections.

**Minor Issue:** The module docstring (lines 1-4) is minimal:
```python
"""Training loops for PPO.

This module contains the main training functions extracted from ppo.py.
"""
```

Consider expanding to describe:
- Relationship to vectorized.py (which does NOT use this module)
- The compiled vs non-compiled path
- The dead code situation with `_train_one_epoch`

### 6.4 Import Organization

**Status: GOOD** - Imports are grouped (stdlib, torch, local) and use absolute imports.

**Lazy Imports:** Local imports inside functions are used appropriately:
```python
def run_heuristic_episode(...):
    from esper.leyline import SeedStage
    from esper.tolaria import create_model
    from esper.tamiyo import SignalTracker
```

This prevents circular imports and reduces initial load time.

---

## 7. Summary of Findings

| ID | Severity | Category | Issue | Location |
|----|----------|----------|-------|----------|
| T1 | P1 | torch.compile | Global mutable `USE_COMPILED_TRAIN_STEP` flag | L27-72 |
| T2 | P3 | torch.compile | Dynamic control flow in compiled function | L57-62 |
| D1 | P2 | Device | LM path tensor created on wrong device | L157 |
| I1 | P2 | Integration | `_train_one_epoch` helper unused by heuristic path | L96-183 vs L336-369 |
| I2 | P3 | Integration | Incomplete `__all__` exports | L653-656 |
| Q1 | P3 | Quality | Missing type hints on function parameters | L236-248 |

---

## 8. Recommendations

### Immediate (P1-P2)

1. **Fix global compile flag** - Move to per-call or per-config control
2. **Fix LM device mismatch** - Use explicit `device` parameter
3. **Complete helper integration** - Either use `_train_one_epoch` in heuristic path or document why it's not used

### Near-term (P3)

4. **Add type hints** to `policy`, `trainloader`, `testloader` parameters
5. **Expand module docstring** to clarify architecture
6. **Consider separate compiled functions** per task type for CUDA graph efficiency

### Testing Recommendations

1. Add test for cross-device behavior in LM path
2. Add integration test verifying heuristic path matches PPO path behavior
3. Add test for compile flag isolation (per JANK-001)

---

## Appendix: Cross-References

- **Known Bug:** `JANK-001-P2` - Global compile flag issue
- **Archived Plan:** `2025-12-09-train-helper-integration.md` - Incomplete integration
- **Dependencies:**
  - `esper.kasmina.host.SeedHostModel` - Model interface
  - `esper.simic.gradient_collector` - Async gradient collection
  - `esper.simic.rewards` - Reward computation
  - `esper.tamiyo.SignalTracker` - Training signal tracking
  - `esper.utils.loss` - Loss computation utilities

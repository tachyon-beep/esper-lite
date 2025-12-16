# Simic Audit: TamiyoRolloutBuffer

**File:** `/home/john/esper-lite/src/esper/simic/tamiyo_buffer.py`
**Date:** 2025-12-16
**Auditor:** Claude (PyTorch Engineering Specialist)

---

## Executive Summary

`TamiyoRolloutBuffer` is a well-designed, pre-allocated rollout buffer for factored recurrent PPO. The implementation correctly addresses the P0 GAE interleaving bug through per-environment storage and demonstrates good PyTorch practices. However, there are several issues ranging from minor code quality concerns to potential integration risks that warrant attention.

**Overall Assessment:** GOOD with minor improvements recommended

| Category | Rating | Critical Issues |
|----------|--------|-----------------|
| torch.compile Compatibility | Good | 0 |
| Device Placement | Good | 0 |
| Gradient Flow | Good | 0 |
| Memory Efficiency | Good | 0 |
| Integration Risks | Moderate | 1 |
| Code Quality | Good | 0 |

---

## 1. torch.compile Compatibility

### 1.1 @torch.compiler.disable on GAE Computation

**Severity:** INFO (Correct Design)

```python
@torch.compiler.disable  # Python loops cause graph breaks; runs once per rollout
def compute_advantages_and_returns(
    self,
    gamma: float = DEFAULT_GAMMA,
    gae_lambda: float = 0.95,
) -> None:
```

**Analysis:** This decorator is correctly applied. The GAE computation contains:
- Python `for` loops over environments and timesteps
- Dynamic slicing based on `step_counts`
- Conditional branching for truncation handling

These patterns would cause graph breaks regardless. The decorator prevents wasted compilation attempts and makes the intent explicit. Since GAE runs once per rollout (not in the hot path), the performance impact is negligible.

**Recommendation:** No change needed. The comment accurately describes the rationale.

### 1.2 TamiyoRolloutStep NamedTuple

**Severity:** INFO (Correct Design)

```python
class TamiyoRolloutStep(NamedTuple):
    """Single transition for factored recurrent actor-critic.

    Uses flat NamedTuple (not nested) for torch.compile compatibility.
    """
```

**Analysis:** The flat structure is correct for compile compatibility. Nested dataclasses or complex objects in collections would cause guard failures. However, `TamiyoRolloutStep` is not used anywhere in the codebase - the buffer uses direct tensor storage instead.

**Recommendation:** Consider removing `TamiyoRolloutStep` if it's unused (dead code), or document its intended future use.

### 1.3 get_batched_sequences Device Transfer

**Severity:** LOW

```python
def get_batched_sequences(
    self,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    # ...
    nb = device.type == "cuda"
    return {
        "states": self.states.to(device, non_blocking=nb),
        # ... 20+ tensor transfers
    }
```

**Analysis:** The method correctly uses `non_blocking=True` for CUDA transfers. However, the dictionary comprehension pattern with 20+ `.to()` calls may not be optimal for torch.compile:
- Each `.to()` is a separate op in the graph
- The dict construction creates Python control flow

**Recommendation:** For PPO update frequency (once per rollout), this is acceptable. If ever called in a tight loop, consider batching transfers or using pinned memory.

---

## 2. Device Placement

### 2.1 Consistent Device Handling

**Severity:** INFO (Correct Implementation)

```python
device: torch.device = field(default_factory=lambda: torch.device("cpu"))

def __post_init__(self):
    device = self.device
    # All tensors allocated on self.device
    self.states = torch.zeros(n, m, self.state_dim, device=device)
```

**Analysis:** Device placement is consistent throughout:
- All pre-allocated tensors use `self.device`
- `get_batched_sequences` handles cross-device transfer correctly
- `compute_advantages_and_returns` creates intermediate tensors on `self.device`

**Recommendation:** No change needed.

### 2.2 Hidden State Device Handling in add()

**Severity:** LOW

```python
def add(self, ..., hidden_h: torch.Tensor, hidden_c: torch.Tensor, ...):
    self.hidden_h[env_id, step_idx] = hidden_h.detach().squeeze(1)
```

**Analysis:** The `add()` method assumes `hidden_h` and `hidden_c` are on the same device as the buffer. If they're on a different device, this will trigger an implicit device transfer. While this works, it's not explicit.

**Recommendation:** Consider adding an assertion or explicit `.to(self.device)` for robustness:

```python
# Suggested (defensive):
self.hidden_h[env_id, step_idx] = hidden_h.detach().squeeze(1).to(self.device)
```

---

## 3. Gradient Flow

### 3.1 Proper Gradient Detachment

**Severity:** INFO (Correct Implementation)

```python
def add(self, ...):
    self.states[env_id, step_idx] = state.detach()
    # ...
    self.slot_masks[env_id, step_idx] = slot_mask.detach().bool()
    # ...
    self.hidden_h[env_id, step_idx] = hidden_h.detach().squeeze(1)
```

**Analysis:** All tensors stored in the buffer are correctly `.detach()`ed. This prevents:
- Gradient graph retention (memory leak)
- Accidental backprop through old transitions
- Staleness issues with LSTM hidden states

**Recommendation:** No change needed. This is correct PPO practice.

### 3.2 Advantages Not Requiring Gradients

**Severity:** INFO (Correct Implementation)

```python
advantages = torch.zeros(num_steps, device=self.device)
# GAE computation is pure arithmetic, no grad needed
self.advantages[env_id, :num_steps] = advantages
```

**Analysis:** Advantages are computed without gradients, which is correct. PPO uses advantages as fixed targets during the policy update - they should not be part of the computation graph.

---

## 4. Memory Efficiency

### 4.1 Pre-allocation Strategy

**Severity:** INFO (Good Design)

```python
def __post_init__(self):
    """Allocate all tensors upfront."""
    self.step_counts = [0] * self.num_envs
    # ~20 tensor allocations
```

**Analysis:** Pre-allocation is the correct approach for fixed-length episodes:
- Avoids GC pressure from intermediate allocations
- Enables torch.compile to reason about tensor shapes
- Predictable memory footprint (~100KB for 4 envs x 25 steps)

**Recommendation:** No change needed.

### 4.2 reset() Does Not Zero Tensors

**Severity:** INFO (Intentional Optimization)

```python
def reset(self) -> None:
    """Reset buffer for new episode collection."""
    self.step_counts = [0] * self.num_envs
    self._current_episode_start = {}
    self.episode_boundaries = {}
    # Tensors don't need zeroing - step_counts controls valid range
```

**Analysis:** This is a performance optimization. Since `step_counts` controls the valid range, stale data in unused indices is never accessed. This saves N * M * tensor_size zero operations per reset.

**Recommendation:** The comment correctly documents the rationale. No change needed.

### 4.3 Tensor Memory Layout

**Severity:** LOW

```python
# Shape: [num_envs, max_steps, ...]
self.states = torch.zeros(n, m, self.state_dim, device=device)
self.hidden_h = torch.zeros(n, m, self.lstm_layers, self.lstm_hidden_dim, device=device)
```

**Analysis:** The `[num_envs, max_steps, ...]` layout is optimal for per-environment iteration in `compute_advantages_and_returns`. However, for batched network evaluation in `get_batched_sequences`, this may not be the most cache-friendly layout.

**Recommendation:** Current layout is fine for the access patterns. If profiling shows memory bandwidth issues, consider `[max_steps, num_envs, ...]` for network evaluation.

---

## 5. Integration Risks

### 5.1 Action Mask Dimension Mismatch Risk

**Severity:** MEDIUM

```python
# Buffer allocation uses constants from leyline
num_slots: int = NUM_SLOTS        # 3
num_blueprints: int = NUM_BLUEPRINTS  # 5
num_blends: int = NUM_BLENDS      # 3
num_ops: int = NUM_OPS            # 4

# Allocates:
self.slot_masks = torch.zeros(n, m, self.num_slots, dtype=torch.bool, device=device)
self.blueprint_masks = torch.zeros(n, m, self.num_blueprints, dtype=torch.bool, device=device)
```

**Analysis:** The buffer pre-allocates action masks based on `leyline.factored_actions` constants. If these constants change (e.g., adding a new blueprint), the buffer will:
- Silently accept mismatched masks during `add()` (broadcasting or truncation)
- Potentially cause shape mismatches during PPO update

**Current usage in tests:**
```python
# test_tamiyo_buffer.py
slot_mask=torch.ones(3, dtype=torch.bool),
blueprint_mask=torch.ones(5, dtype=torch.bool),
```

Tests hardcode dimensions rather than using the leyline constants, which creates a maintenance burden.

**Recommendation:** Add shape validation in `add()`:

```python
def add(self, ..., slot_mask: torch.Tensor, ...):
    if slot_mask.shape != (self.num_slots,):
        raise ValueError(
            f"slot_mask shape {slot_mask.shape} != expected ({self.num_slots},)"
        )
```

### 5.2 LSTM Hidden State Shape Assumption

**Severity:** MEDIUM

```python
# add() assumes hidden_h has shape [num_layers, batch=1, hidden_dim]
self.hidden_h[env_id, step_idx] = hidden_h.detach().squeeze(1)
```

**Analysis:** The `squeeze(1)` assumes `batch=1` in the LSTM output. This is true for single-environment inference but:
- Would fail silently if batch size > 1 (squeeze would no-op)
- Creates a tight coupling between buffer and inference pattern

The test verifies this works:
```python
# test_tamiyo_buffer.py
hidden_h = torch.randn(1, 1, 128)  # [layers, batch, hidden]
# ...
stored_h = buffer.hidden_h[0, 0]
assert stored_h.shape == (1, 128)  # [layers, hidden]
```

**Recommendation:** Make the squeeze explicit with dimension check:

```python
# Explicit assertion
assert hidden_h.size(1) == 1, f"Expected batch=1, got {hidden_h.size(1)}"
self.hidden_h[env_id, step_idx] = hidden_h.detach().squeeze(1)
```

### 5.3 Episode Boundary Tracking Unused

**Severity:** LOW

```python
# Episode boundary tracking
_current_episode_start: dict[int, int] = field(default_factory=dict, init=False)
episode_boundaries: dict[int, list[tuple[int, int]]] = field(
    default_factory=dict, init=False
)

def start_episode(self, env_id: int) -> None:
    """Mark start of a new episode for an environment."""
    self._current_episode_start[env_id] = self.step_counts[env_id]

def end_episode(self, env_id: int) -> None:
    """Mark end of episode, recording boundary."""
    # ... records (start, end) tuple
```

**Analysis:** Episode boundaries are tracked but never used in:
- `compute_advantages_and_returns` - processes all steps, not per-episode
- `get_batched_sequences` - returns all data
- `normalize_advantages` - normalizes globally

The only callers in `vectorized.py` do call `start_episode`/`end_episode`, but the recorded data is never consumed.

**Recommendation:** Either:
1. Remove dead code if boundaries are not needed
2. Document intended future use (e.g., per-episode advantage normalization)

### 5.4 Truncation Bootstrap Value Handling

**Severity:** INFO (Correct Implementation)

```python
# In compute_advantages_and_returns:
if truncated[t]:
    next_value = bootstrap_values[t]
    # Truncation is NOT a true terminal - the episode was cut off
    # by time limit. We MUST use next_non_terminal=1.0
    next_non_terminal = 1.0
```

**Analysis:** This correctly handles truncation vs. true termination for GAE. The bootstrap value is used for truncated episodes (time limit), while `next_non_terminal=0.0` would reset the value estimate at true terminals.

The comment accurately explains the subtlety. This matches best practices from CleanRL and SB3.

---

## 6. Code Quality

### 6.1 Type Annotations

**Severity:** INFO (Good)

```python
def add(
    self,
    env_id: int,
    state: torch.Tensor,
    slot_action: int,
    # ... comprehensive type hints
) -> None:
```

**Analysis:** Type annotations are complete and accurate throughout the module.

### 6.2 Documentation Quality

**Severity:** INFO (Excellent)

The module has excellent documentation:
- Module-level docstring explains design rationale
- Class docstrings explain pre-allocation strategy
- Method docstrings describe behavior and parameters
- Inline comments explain non-obvious decisions (truncation handling, memory optimization)

### 6.3 TamiyoRolloutStep Unused

**Severity:** LOW

```python
class TamiyoRolloutStep(NamedTuple):
    """Single transition for factored recurrent actor-critic."""
    # 18 fields defined
```

**Analysis:** This type is defined but never instantiated or used in the codebase. The buffer uses direct tensor storage instead of step objects.

**Recommendation:** Remove if unused, or document intended API if it's meant for external consumers.

### 6.4 Magic Numbers

**Severity:** LOW

```python
gae_lambda: float = 0.95,  # In compute_advantages_and_returns
```

**Analysis:** The default `gae_lambda=0.95` differs from `DEFAULT_GAE_LAMBDA=0.97` in leyline. While the caller can override this, having different defaults creates confusion.

**Recommendation:** Use the leyline constant:

```python
from esper.leyline import DEFAULT_GAMMA, DEFAULT_GAE_LAMBDA

def compute_advantages_and_returns(
    self,
    gamma: float = DEFAULT_GAMMA,
    gae_lambda: float = DEFAULT_GAE_LAMBDA,  # Use leyline constant
) -> None:
```

---

## 7. Test Coverage Analysis

The test file (`tests/simic/test_tamiyo_buffer.py`) covers:

| Test Case | Coverage |
|-----------|----------|
| Per-env storage isolation (P0 bug) | YES |
| Per-head log probs storage | YES |
| LSTM hidden state storage | YES |
| Empty buffer handling | YES |
| Buffer overflow | YES |

**Missing test coverage:**

1. **Truncation vs. done handling** - GAE computation differs for truncated episodes
2. **Bootstrap value usage** - Critical for correct value estimates
3. **Advantage normalization** - `normalize_advantages()` not directly tested
4. **get_batched_sequences output** - Shape and device correctness
5. **Episode boundary API** - `start_episode`/`end_episode` not tested independently

**Recommendation:** Add tests for truncation handling and advantage normalization.

---

## 8. Summary of Recommendations

### High Priority (Should Fix)

1. **Add shape validation in `add()` for action masks** - Prevents silent dimension mismatches when action space changes

### Medium Priority (Should Consider)

2. **Add explicit batch dimension check for LSTM hidden states** - Documents assumption, fails fast on misuse
3. **Use leyline constants for GAE lambda default** - Consistency with rest of codebase

### Low Priority (Nice to Have)

4. **Remove or document `TamiyoRolloutStep`** - Dead code or missing documentation
5. **Remove or document episode boundary tracking** - Currently unused infrastructure
6. **Add test coverage for truncation/bootstrap** - Critical path not directly tested

---

## 9. Appendix: Integration Points

### Files that import TamiyoRolloutBuffer

```
src/esper/simic/ppo.py          # PPOAgent.buffer
src/esper/simic/tamiyo_buffer.py  # Definition
```

### Key integration contracts

| Contract | Location | Notes |
|----------|----------|-------|
| Action space dimensions | `leyline.factored_actions` | NUM_SLOTS, NUM_BLUEPRINTS, NUM_BLENDS, NUM_OPS |
| LSTM hidden dim | `leyline.DEFAULT_LSTM_HIDDEN_DIM` | 128 |
| Episode length | `leyline.DEFAULT_EPISODE_LENGTH` | 25 |
| Gamma | `leyline.DEFAULT_GAMMA` | 0.995 |
| GAE lambda | `leyline.DEFAULT_GAE_LAMBDA` | 0.97 (but buffer defaults to 0.95) |

### PPO Update Integration

```python
# ppo.py:388-394
self.buffer.compute_advantages_and_returns(
    gamma=self.gamma, gae_lambda=self.gae_lambda
)
self.buffer.normalize_advantages()
data = self.buffer.get_batched_sequences(device=self.device)
```

### Vectorized Training Integration

```python
# vectorized.py:1369, 2151, 2177, 2218
agent.buffer.start_episode(env_id=env_idx)
agent.buffer.add(env_id=env_idx, ...)
agent.buffer.end_episode(env_id=env_idx)
agent.buffer.reset()  # On rollback
```

---

*End of Audit Report*

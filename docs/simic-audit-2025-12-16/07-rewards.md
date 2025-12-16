# Simic Rewards Module Audit Report

**File:** `/home/john/esper-lite/src/esper/simic/rewards.py`
**Date:** 2025-12-16
**Auditor:** Claude (PyTorch Engineering Specialist)

---

## Executive Summary

The `rewards.py` module is a **pure-Python reward computation module** with no PyTorch dependencies. This is an intentional design choice that keeps reward calculations on CPU, avoiding GPU synchronization overhead in the hot path. The module is well-designed with extensive documentation, comprehensive test coverage, and mathematically sound PBRS (Potential-Based Reward Shaping) implementation.

**Overall Assessment: HEALTHY**

The module exhibits no PyTorch-specific issues because it deliberately avoids PyTorch. The reward computations use standard Python `math` operations, which is appropriate for this domain. The code quality is high with thorough anti-gaming mechanisms and extensive property-based testing.

---

## 1. PyTorch/torch.compile Analysis

### 1.1 No PyTorch Dependencies (Intentional Design)

| Finding | Severity | Status |
|---------|----------|--------|
| Module uses only Python `math` stdlib | INFO | Correct Design |

**Analysis:**

The module contains exactly zero PyTorch imports or operations. This is deliberate and correct:

```python
import logging
import math  # <-- Only stdlib math
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple
```

**Rationale:**
- Reward computation happens per-timestep in RL training loops
- Keeping rewards on CPU avoids GPU<->CPU synchronization points
- Python `math` operations are sufficient for scalar arithmetic
- Computed rewards are batched and converted to tensors in `vectorized.py`

### 1.2 Integration with torch.compile

The reward module integrates correctly with the compiled training pipeline:

- **vectorized.py** imports `compute_reward`, `SeedInfo` and uses them outside the compiled graph
- Reward values are later batched into tensors for PPO updates
- No risk of graph breaks since rewards never enter the compiled region

**Verdict: No issues. Design is correct for torch.compile compatibility.**

---

## 2. Device Placement Analysis

### 2.1 CPU-Only by Design

| Finding | Severity | Status |
|---------|----------|--------|
| All computations are CPU scalars | INFO | Correct |
| No device placement code needed | INFO | Correct |

**Analysis:**

Since the module uses only Python floats and `math` operations, there are no device placement concerns. The integration layer (`vectorized.py`) handles tensor conversion:

```python
# In vectorized.py (line ~1964)
seed_info = SeedInfo.from_seed_state(seed_state, seed_params_for_slot)
reward = compute_reward(...)  # Returns Python float
# Later batched into torch.tensor for PPO
```

---

## 3. Gradient Flow Analysis

### 3.1 No Gradient Concerns

| Finding | Severity | Status |
|---------|----------|--------|
| Module produces scalar rewards (no gradients) | INFO | Correct |

**Analysis:**

Reward computation is inherently non-differentiable in policy gradient methods. The reward signal is used to weight log-probabilities, not backpropagated through. The module correctly returns Python floats, which are later used in:

```python
# PPO advantage estimation
advantages = rewards + gamma * values_next - values  # In tensor form elsewhere
```

**Verdict: No gradient concerns. Rewards are correctly non-differentiable scalars.**

---

## 4. Memory Management Analysis

### 4.1 Efficient Memory Patterns

| Finding | Severity | Status |
|---------|----------|--------|
| Uses `dataclass(slots=True)` for configs | INFO | Good Practice |
| Uses `NamedTuple` for `SeedInfo` | INFO | Good Practice |
| Module-level default config singleton | INFO | Good Practice |

**Analysis:**

Memory efficiency is well-considered:

```python
@dataclass(slots=True)
class ContributionRewardConfig:
    """Uses __slots__ for reduced memory footprint."""
    ...

class SeedInfo(NamedTuple):
    """Immutable, memory-efficient tuple subclass."""
    ...

# Singleton pattern avoids repeated allocations
_DEFAULT_CONTRIBUTION_CONFIG = ContributionRewardConfig()
```

### 4.2 RewardComponentsTelemetry Efficiency

```python
@dataclass(slots=True)
class RewardComponentsTelemetry:
    """Slots-based telemetry for hot path."""

    def to_dict(self) -> dict:
        """Uses explicit dict construction instead of asdict() for 3-5x performance."""
        return {
            "base_acc_delta": self.base_acc_delta,
            # ... explicit fields
        }
```

**Verdict: Memory patterns are efficient and appropriate.**

---

## 5. Integration Risk Analysis

### 5.1 Contract Adherence with Leyline

| Finding | Severity | Status |
|---------|----------|--------|
| Imports constants from leyline | LOW | Good |
| Stage values hardcoded match SeedStage enum | MEDIUM | Minor Risk |

**Analysis:**

The module imports key constants from leyline:

```python
from esper.leyline import SeedStage, MIN_CULL_AGE, MIN_PROBATION_EPOCHS, DEFAULT_GAMMA
from esper.leyline import DEFAULT_MIN_FOSSILIZE_CONTRIBUTION
```

However, stage values are also locally cached:

```python
STAGE_GERMINATED = SeedStage.GERMINATED.value  # 2
STAGE_TRAINING = SeedStage.TRAINING.value      # 3
STAGE_BLENDING = SeedStage.BLENDING.value      # 4
STAGE_FOSSILIZED = SeedStage.FOSSILIZED.value  # 7
STAGE_PROBATIONARY = SeedStage.PROBATIONARY.value  # 6
```

**Risk:** If `SeedStage` enum values change in leyline, these cached values would auto-update. However, the `STAGE_POTENTIALS` dict uses integer keys directly:

```python
STAGE_POTENTIALS = {
    0: 0.0,   # UNKNOWN
    1: 0.0,   # DORMANT
    2: 1.0,   # GERMINATED
    3: 2.0,   # TRAINING
    4: 3.5,   # BLENDING
    6: 5.5,   # PROBATIONARY (5 skipped)
    7: 6.0,   # FOSSILIZED
}
```

**Recommendation:** Consider using enum values as keys for `STAGE_POTENTIALS` for type safety:

```python
STAGE_POTENTIALS = {
    SeedStage.UNKNOWN.value: 0.0,
    SeedStage.DORMANT.value: 0.0,
    # etc.
}
```

**Severity: LOW** - The current approach works and has extensive test coverage.

### 5.2 Vectorized Training Integration

| Finding | Severity | Status |
|---------|----------|--------|
| `compute_reward` is primary entry point | INFO | Correct |
| `SeedInfo.from_seed_state` bridges kasmina->simic | INFO | Correct |
| Telemetry components properly tracked | INFO | Correct |

The integration in `vectorized.py` (lines 1964-2000) correctly:
1. Creates `SeedInfo` from kasmina's `SeedState`
2. Calls `compute_reward` with all required parameters
3. Optionally captures `RewardComponentsTelemetry` for debugging

### 5.3 Multi-Slot Support

| Finding | Severity | Status |
|---------|----------|--------|
| Rewards computed per-slot in vectorized.py | INFO | Correct |
| No internal slot awareness needed | INFO | Correct |

The reward module is slot-agnostic by design. `vectorized.py` handles multi-slot iteration.

---

## 6. Code Quality Analysis

### 6.1 Documentation Quality

| Finding | Severity | Status |
|---------|----------|--------|
| Module docstring explains usage | INFO | Excellent |
| PBRS design rationale documented | INFO | Excellent |
| Function docstrings comprehensive | INFO | Excellent |
| DRL Expert review comments present | INFO | Excellent |

**Highlights:**

The PBRS section (lines 62-106) provides exceptional documentation:
- Mathematical foundation (Ng et al., 1999)
- Key properties maintained
- Value rationale for each stage
- Tuning history and validation references

### 6.2 Anti-Gaming Mechanisms

| Finding | Severity | Status |
|---------|----------|--------|
| Ransomware seed detection | INFO | Robust |
| Attribution discount (sigmoid) | INFO | Well-tuned |
| Ratio penalty for entanglement | INFO | Effective |
| Legitimacy discount for rapid fossilize | INFO | Proper |
| Anti-penalty-stacking logic | INFO | Critical fix |

**Key anti-gaming features:**

1. **Attribution Discount** (lines 439-440):
```python
if total_imp < 0:
    attribution_discount = 1.0 / (1.0 + math.exp(-10 * total_imp))
```
Sigmoid with -10 coefficient zeros rewards for ransomware seeds.

2. **Ratio Penalty** (lines 444-456):
```python
if seed_contribution > 1.0 and attribution_discount >= 0.5:
    if total_imp > 0.1:
        ratio = seed_contribution / total_imp
        if ratio > 5.0:
            ratio_penalty = -min(0.3, 0.1 * (ratio - 5) / 5)
```
Detects structural entanglement (high counterfactual, low improvement).

3. **Anti-Stacking** (lines 547-550, 559):
```python
# Only penalize when attribution is positive (legitimate seed being farmed)
if seed_info.epochs_in_stage >= 2 and bounded_attribution > 0:
```
Prevents unlearnable reward landscapes from penalty stacking.

### 6.3 Type Safety

| Finding | Severity | Status |
|---------|----------|--------|
| Type hints comprehensive | INFO | Good |
| Optional types correctly used | INFO | Good |
| Enum usage for actions | INFO | Good |

### 6.4 Test Coverage

| Finding | Severity | Status |
|---------|----------|--------|
| Unit tests in `test_simic_rewards.py` | INFO | Comprehensive |
| Property-based tests for bounds | INFO | Excellent |
| Property-based tests for PBRS | INFO | Excellent |
| Anti-gaming tests | INFO | Thorough |
| Semantic invariant tests | INFO | Critical |

Test files identified:
- `tests/test_simic_rewards.py` (1541 lines) - Extensive unit tests
- `tests/properties/test_reward_properties.py` - Mathematical invariants
- `tests/properties/test_pbrs_telescoping.py` - PBRS guarantee verification
- `tests/simic/properties/test_reward_semantics.py` - Semantic invariants
- `tests/simic/properties/test_reward_antigaming.py` - Gaming prevention
- `tests/simic/properties/test_reward_invariants.py` - Additional invariants

---

## 7. Potential Issues and Recommendations

### 7.1 Minor Issues

| Issue | Severity | Recommendation |
|-------|----------|----------------|
| Hardcoded stage integers in `STAGE_POTENTIALS` | LOW | Consider using enum values as keys |
| `_logger.warning` in hot path (line 936-938) | LOW | Consider rate-limiting or debug-only |
| No validation of `host_params > 0` before division | LOW | Add defensive check |

### 7.2 Defensive Division Check

Line 588-591:
```python
if host_params > 0 and total_params > 0:
    growth_ratio = total_params / host_params
```

The `host_params > 0` check is present, which is good. However, line 1181 in `compute_loss_reward` has:
```python
if host_params > 0 and total_params > 0:
    ...
    growth_ratio = total_params / host_params
```

Both are correctly guarded.

### 7.3 Warning in PBRS Calculation

Line 932-938:
```python
if seed_info.previous_epochs_in_stage == 0 and seed_info.previous_stage != 0:
    _logger.warning(
        "PBRS telescoping risk: transition from stage %d with previous_epochs_in_stage=0. "
        ...
    )
```

**Concern:** This warning could fire frequently during normal training, polluting logs.

**Recommendation:** Consider adding a rate limiter or making this debug-level only:
```python
_logger.debug(...)  # Or use a counter to warn only once
```

**Severity: LOW** - This is a correctness warning that helps catch integration bugs.

---

## 8. Performance Characteristics

### 8.1 Hot Path Efficiency

| Operation | Cost | Notes |
|-----------|------|-------|
| `compute_contribution_reward` | O(1) | ~50 arithmetic ops |
| `SeedInfo.from_seed_state` | O(1) | Field copying |
| `RewardComponentsTelemetry.to_dict` | O(1) | Explicit dict, no reflection |

The reward computation is lightweight and appropriate for per-timestep invocation.

### 8.2 Memory Allocation

| Pattern | Assessment |
|---------|------------|
| Config singleton | Avoids repeated allocs |
| `slots=True` on dataclasses | Reduced memory |
| NamedTuple for SeedInfo | Immutable, efficient |
| No intermediate lists/dicts in hot path | Good |

---

## 9. PBRS Correctness Verification

### 9.1 Telescoping Property

The PBRS implementation correctly follows Ng et al. (1999):

```python
def compute_pbrs_bonus(
    potential_prev: float,
    potential_next: float,
    gamma: float = DEFAULT_GAMMA,
) -> float:
    """F(s, s') = gamma * potential(s') - potential(s)"""
    return gamma * potential_next - potential_prev
```

### 9.2 Stage Potential Monotonicity

| Stage | Potential | Increment |
|-------|-----------|-----------|
| UNKNOWN | 0.0 | - |
| DORMANT | 0.0 | 0.0 |
| GERMINATED | 1.0 | 1.0 |
| TRAINING | 2.0 | 1.0 |
| BLENDING | 3.5 | 1.5 (largest) |
| PROBATIONARY | 5.5 | 2.0 |
| FOSSILIZED | 6.0 | 0.5 (smallest) |

This design:
- Incentivizes lifecycle progression (monotonic)
- Emphasizes BLENDING (value creation phase)
- De-emphasizes FOSSILIZED (anti-farming)

### 9.3 Gamma Consistency

```python
# From leyline/__init__.py
DEFAULT_GAMMA = 0.995  # CRITICAL: PPO gamma MUST equal PBRS gamma

# Used in rewards.py
from esper.leyline import DEFAULT_GAMMA  # Single source of truth
```

**Verdict: PBRS implementation is mathematically correct.**

---

## 10. Summary of Findings

### Critical Issues: None

### High Severity Issues: None

### Medium Severity Issues: None

### Low Severity Issues

| # | Issue | Location | Recommendation |
|---|-------|----------|----------------|
| 1 | Hardcoded integers in STAGE_POTENTIALS | Line 108-117 | Use enum values as keys |
| 2 | Warning in hot path | Line 932-938 | Rate-limit or debug-level |

### Positive Findings

1. **Correct separation of concerns** - No PyTorch in reward computation
2. **Excellent documentation** - PBRS rationale, DRL Expert comments
3. **Comprehensive anti-gaming** - Multiple mechanisms prevent reward hacking
4. **Strong test coverage** - Property-based tests verify mathematical invariants
5. **Memory efficient** - Slots, NamedTuple, singleton patterns
6. **Type-safe** - Comprehensive type hints
7. **torch.compile compatible** - No graph break risks

---

## 11. Conclusion

The `rewards.py` module is a well-engineered, CPU-only reward computation system that correctly integrates with the PyTorch-based training pipeline. It demonstrates excellent software engineering practices including:

- Clear separation between reward logic and tensor operations
- Mathematically rigorous PBRS implementation
- Sophisticated anti-gaming mechanisms
- Comprehensive test coverage

**No changes required.** The two low-severity issues identified are cosmetic and do not affect correctness or performance.

---

*Report generated by Claude (PyTorch Engineering Specialist) for Esper-Lite Simic audit.*

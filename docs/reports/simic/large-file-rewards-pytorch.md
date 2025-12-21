# PyTorch Code Review: simic/rewards/rewards.py

**File:** `/home/john/esper-lite/src/esper/simic/rewards/rewards.py`
**Lines:** 1377
**Reviewer:** PyTorch Engineering Specialist
**Date:** 2025-12-17

---

## Executive Summary

The `rewards.py` module implements reward computation for the Simic (RL-based evolution) subsystem of Esper. This is a **pure Python reward engineering module** with no direct PyTorch or NumPy operations. All tensor-related computation happens in the calling code (vectorized training loops).

**Overall Assessment:** The code is well-structured with sophisticated reward engineering that addresses real failure modes (ransomware patterns, fossilization farming, dependency gaming). The mathematical foundations are sound, and the PBRS implementation follows the theory correctly.

### Key Findings

| Severity | Count | Summary |
|----------|-------|---------|
| Critical | 0 | No correctness bugs found |
| High | 2 | Potential numerical stability edge cases |
| Medium | 4 | Best practices and maintainability issues |
| Low | 5 | Minor suggestions and style improvements |

---

## Critical Issues

**None identified.** The reward computation logic is mathematically correct and handles edge cases appropriately.

---

## High-Priority Issues

### H1: Division-by-Zero Risk in Attribution Ratio Calculation

**Location:** Lines 470-485

```python
if seed_contribution > 1.0 and attribution_discount >= 0.5:
    if total_imp > config.improvement_safe_threshold:
        # Safe zone: actual improvement exists
        ratio = seed_contribution / total_imp
```

**Issue:** While the code guards against `total_imp <= improvement_safe_threshold` (default 0.1), there is no explicit guard against `total_imp == 0.0`. The threshold check uses `>` not `>=`, meaning `total_imp = 0.0` falls through to the else branch at line 482, but if the threshold were ever configured to 0.0, we'd have a division by zero.

**Risk Level:** Low in practice (threshold defaults to 0.1), but the defensive pattern is incomplete.

**Recommendation:** Add explicit zero check or use a safe division pattern:
```python
if total_imp > max(config.improvement_safe_threshold, 1e-8):
    ratio = seed_contribution / total_imp
```

### H2: Potential Exponential Overflow in Holding Warning

**Location:** Lines 594-599

```python
# Exponential: epoch 2 -> -1.0, epoch 3 -> -3.0, epoch 4 -> -9.0
# Formula: -1.0 * (3 ** (epochs_waiting - 1))
epochs_waiting = seed_info.epochs_in_stage - 1
holding_warning = -1.0 * (3 ** (epochs_waiting - 1))
# Cap at -10.0 (clip boundary) to avoid extreme penalties
holding_warning = max(holding_warning, -10.0)
```

**Issue:** For very long episodes (edge case), `epochs_waiting` could grow large. While `max(..., -10.0)` caps the value, the intermediate calculation `3 ** (epochs_waiting - 1)` could overflow for extreme values:
- `epochs_waiting = 40`: `3^39` exceeds float64 range
- `epochs_waiting = 20`: `3^19 = 1.16e9` (manageable but large)

**Risk Level:** Low (episodes are typically 25 epochs), but the exponential growth is unbounded before clamping.

**Recommendation:** Compute the capped result directly:
```python
# Safe exponential with early capping
if epochs_waiting > 5:  # 3^4 = 81, already past -10 threshold
    holding_warning = -10.0
else:
    holding_warning = -1.0 * (3 ** max(0, epochs_waiting - 1))
    holding_warning = max(holding_warning, -10.0)
```

---

## Medium-Priority Issues

### M1: Sigmoid Numerical Stability Edge Case

**Location:** Lines 465-466

```python
if total_imp < 0:
    attribution_discount = 1.0 / (1.0 + math.exp(-config.attribution_sigmoid_steepness * total_imp))
```

**Issue:** With default `attribution_sigmoid_steepness = 10.0` and extreme negative `total_imp` (e.g., -100), the exponent becomes `10 * 100 = 1000`, and `math.exp(1000)` overflows to `inf`. The result is `1.0 / inf = 0.0`, which is mathematically correct for the discount behavior, but relies on IEEE 754 inf handling.

**Current Behavior:** Works correctly due to Python/IEEE 754 semantics (`1.0 / inf = 0.0`).

**Recommendation:** Document this reliance on IEEE 754 behavior or add explicit clamping:
```python
# Clamp exponent to avoid overflow (result would be ~0 anyway beyond this)
exponent = min(-config.attribution_sigmoid_steepness * total_imp, 700)
attribution_discount = 1.0 / (1.0 + math.exp(exponent))
```

### M2: Magic Numbers in Ransomware Detection

**Location:** Lines 1017-1024

```python
# Base penalty + scaled by damage done (capped at 1.0 extra)
base_penalty = -0.5
damage_scale = min(abs(total_delta) * 0.2, 1.0)

# Extra penalty for ransomware signature
ransomware_signature = (
    seed_contribution is not None
    and seed_contribution > 0.1
    and total_delta < -0.2
)
ransomware_penalty = -0.3 if ransomware_signature else 0.0
```

**Issue:** Magic numbers `-0.5`, `0.2`, `1.0`, `0.1`, `-0.2`, `-0.3` are not configurable and not documented. These values interact with other reward components but are hardcoded, making tuning difficult.

**Recommendation:** Move to `ContributionRewardConfig` dataclass:
```python
@dataclass(slots=True)
class ContributionRewardConfig:
    # ... existing fields ...

    # Ransomware detection
    ransomware_base_penalty: float = -0.5
    ransomware_damage_scale: float = 0.2
    ransomware_max_damage: float = 1.0
    ransomware_contribution_threshold: float = 0.1
    ransomware_degradation_threshold: float = -0.2
    ransomware_signature_penalty: float = -0.3
```

### M3: Inconsistent None Handling in compute_loss_reward

**Location:** Lines 1272-1320

```python
def compute_loss_reward(
    action: int,  # <- Not LifecycleOp, inconsistent with other functions
    ...
```

**Issue:** The `action` parameter type is `int` while all other reward functions use `LifecycleOp`. Additionally, the function never uses the `action` parameter - it's dead code.

**Evidence:** The function signature accepts `action: int` but the body never references it.

**Recommendation:** Either:
1. Remove the unused parameter
2. Add action-specific shaping as in `compute_contribution_reward`

### M4: Defensive Check with Warning but No Return

**Location:** Lines 962-967

```python
if seed_info.epochs_in_stage == 0:
    # Just transitioned - use actual previous epoch count for correct telescoping
    if seed_info.previous_epochs_in_stage == 0 and seed_info.previous_stage != 0:
        _logger.warning(
            "PBRS telescoping risk: transition from stage %d with previous_epochs_in_stage=0. "
            ...
        )
```

**Issue:** The warning is logged but computation continues with potentially incorrect PBRS bonus. This could silently corrupt reward signals.

**Recommendation:** Consider returning a safe default or raising an exception for this invariant violation:
```python
if seed_info.previous_epochs_in_stage == 0 and seed_info.previous_stage != 0:
    _logger.warning(...)
    # Use stage potential only, skip epoch progress bonus to avoid corruption
    phi_prev = STAGE_POTENTIALS.get(seed_info.previous_stage, 0.0)
    # previous_epochs_in_stage contribution is zero (conservative)
```

---

## Low-Priority Issues

### L1: Singleton Default Config Anti-Pattern

**Location:** Lines 352-353

```python
# Default config singleton (avoid repeated allocations)
_DEFAULT_CONTRIBUTION_CONFIG = ContributionRewardConfig()
```

**Issue:** Module-level mutable singleton. While `ContributionRewardConfig` is a dataclass with `slots=True`, it's not frozen. If any code modifies the default, all callers would see corrupted defaults.

**Recommendation:** Use `frozen=True` in the dataclass definition:
```python
@dataclass(frozen=True, slots=True)
class ContributionRewardConfig:
    ...
```

### L2: Redundant Stage Constant Definitions

**Location:** Lines 339-344

```python
STAGE_GERMINATED = SeedStage.GERMINATED.value
STAGE_TRAINING = SeedStage.TRAINING.value
STAGE_BLENDING = SeedStage.BLENDING.value
STAGE_FOSSILIZED = SeedStage.FOSSILIZED.value
STAGE_HOLDING = SeedStage.HOLDING.value
```

**Issue:** These constants duplicate values already available via `SeedStage` enum. They create maintenance burden if stage values change.

**Recommendation:** Use `SeedStage` directly or create a typed alias:
```python
# Direct usage (preferred)
if seed_info.stage == SeedStage.BLENDING.value:
    ...
```

### L3: TODO Comment for Unwired Telemetry

**Location:** Lines 1076-1077

```python
# TODO: [UNWIRED TELEMETRY] - Call _check_reward_hacking() and _check_ransomware_signature()
# from compute_contribution_reward() when attribution is computed.
```

**Issue:** Telemetry functions are defined but not called. This represents dead code or incomplete implementation.

**Recommendation:** Either wire up the telemetry or remove the dead code per the project's "No Legacy Code Policy".

### L4: Unclear achievable_range Property Guard

**Location:** Lines 1316-1317

```python
achievable_range = config.achievable_range or 1.0
normalized = max(0.0, min(improvement / achievable_range, 1.0))
```

**Issue:** The `or 1.0` guard protects against zero division but `achievable_range` is a computed property that can't be zero unless `baseline_loss == target_loss`. The guard is defensive but the condition seems impossible given the config.

**Recommendation:** Add a comment explaining the guard or remove if truly unreachable.

### L5: Import Inside Function for Telemetry

**Location:** Lines 1098, 1144

```python
def _check_reward_hacking(...):
    from esper.leyline import TelemetryEvent, TelemetryEventType
```

**Issue:** Deferred imports inside functions that are never called (dead code). If these were called frequently, the import would happen on every call.

**Recommendation:** Since this is dead code (per L3), either remove it or move imports to module level if enabling.

---

## Positive Observations

### P1: Excellent PBRS Implementation

The PBRS (Potential-Based Reward Shaping) implementation follows Ng et al. (1999) correctly:
- Single source of truth for potentials (`STAGE_POTENTIALS`)
- Gamma consistency enforced via `DEFAULT_GAMMA` from leyline
- Telescoping property preserved in `_contribution_pbrs_bonus`
- Comprehensive documentation in header comments (lines 62-106)

### P2: Sophisticated Anti-Gaming Mechanisms

The reward function addresses real failure modes:
- **Ransomware detection** (lines 1007-1024): Penalizes seeds with high contribution but negative total improvement
- **Attribution discount** (lines 458-466): Sigmoid discount for regressing seeds
- **Ratio penalty** (lines 470-485): Catches contribution >> improvement gaming
- **Legitimacy discount** (lines 1026-1028): Prevents rapid fossilization farming

### P3: Clean Separation from PyTorch

The module correctly separates reward computation (pure Python) from tensor operations (vectorized training). This:
- Avoids device placement issues
- Enables easy testing without GPU
- Allows reward functions to be called per-environment in vectorized training

### P4: Comprehensive Test Coverage

The test suite (`tests/simic/properties/`) includes:
- Property-based tests for bounds and invariants
- Anti-gaming property tests
- PBRS telescoping verification
- Edge case coverage for all reward paths

---

## torch.compile Compatibility

**N/A** - This module contains no PyTorch operations. Reward computation is done in pure Python and the results are stored in Python floats before being converted to tensors in the rollout buffer.

**Observation:** The calling code in `vectorized.py` correctly handles the Python-to-tensor boundary:
```python
reward = compute_reward(...)  # Returns Python float
rewards[env_idx] = reward     # Stored in pre-allocated tensor
```

---

## Memory Efficiency

**N/A** - No tensor allocations in this module. All reward computation uses scalar Python floats.

**Positive:** The `RewardComponentsTelemetry` dataclass uses `slots=True` for memory efficiency when telemetry is enabled:
```python
@dataclass(slots=True)
class RewardComponentsTelemetry:
    ...
```

---

## Numerical Stability Summary

| Component | Risk | Mitigation |
|-----------|------|------------|
| Sigmoid attribution discount | Low | IEEE 754 inf handling works correctly |
| Exponential holding warning | Low | Capped at -10.0, but intermediate can overflow |
| Division for ratio penalty | Very Low | Guarded by threshold > 0.1 |
| Log rent calculation | None | `math.log(1.0 + x)` for x >= 0 is safe |

---

## Recommendations Summary

### Must Fix (High Priority)
1. **H1:** Add explicit zero guard in ratio calculation
2. **H2:** Cap exponential before computation, not after

### Should Fix (Medium Priority)
3. **M2:** Move magic numbers to config
4. **M3:** Fix inconsistent `action` parameter in `compute_loss_reward`

### Consider (Low Priority)
5. **L1:** Make `ContributionRewardConfig` frozen
6. **L3:** Wire up or remove dead telemetry functions

---

## Appendix: Code Quality Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| Lines of Code | 1377 | Large but justified (see P2-2 comment in code) |
| Cyclomatic Complexity | High | Expected for reward engineering |
| Test Coverage | Excellent | Property-based + unit tests |
| Documentation | Good | Header comments, design rationale documented |
| Type Hints | Complete | All public functions have type annotations |

---

*Report generated by PyTorch Engineering Specialist review.*

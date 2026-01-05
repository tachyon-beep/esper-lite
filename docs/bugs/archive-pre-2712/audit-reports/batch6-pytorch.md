# Batch 6 Code Review: Simic Rewards (PyTorch Engineering Focus)

**Reviewer:** PyTorch Engineering Specialist
**Date:** 2025-12-27
**Files Reviewed:**
1. `/home/john/esper-lite/src/esper/simic/rewards/__init__.py`
2. `/home/john/esper-lite/src/esper/simic/rewards/rewards.py`
3. `/home/john/esper-lite/src/esper/simic/rewards/reward_telemetry.py`

---

## Executive Summary

The reward computation module is **well-designed for pure Python reward engineering** - no PyTorch tensors are involved directly in these files, which is architecturally correct. Rewards are computed as Python floats and are converted to tensors at the training loop level (in `vectorized.py`). This separation is good design.

However, several issues were found:
- **P1**: Test suite has stale component sum tests (missing `synergy_bonus`)
- **P2**: Unused `boost_received` parameter in `_compute_synergy_bonus`
- **P2**: Potential numerical instability in PBRS bonus calculation
- **P3**: Multiple test files have inconsistent component sum calculations
- **P3**: Telemetry `shaped_reward_ratio` includes terms that don't sum correctly

---

## File-by-File Analysis

### 1. `__init__.py` (Module Interface)

**Purpose:** Clean public API for reward computation subpackage.

**Assessment:** Well-organized with clear categorization of exports (config classes, reward functions, PBRS utilities, constants). Exports internal helpers (`_contribution_pbrs_bonus`, etc.) explicitly for testing, which is a good practice.

**No issues found.**

---

### 2. `rewards.py` (Core Reward Computation)

**Purpose:** Implements dense reward shaping for the seed lifecycle controller. Supports multiple reward modes (SHAPED, SPARSE, MINIMAL, SIMPLIFIED) for experimentation.

**Architectural Observations:**

1. **No PyTorch Dependencies** - Deliberate and correct. Rewards are pure Python `float` values. Tensor conversion happens in the training loop where device placement is handled centrally.

2. **Numerical Stability** - Uses `math.log`, `math.exp`, `math.tanh`, `math.sqrt` from Python stdlib. These are numerically stable for the value ranges involved.

3. **PBRS Implementation** - Follows Ng et al. (1999) correctly. The telescoping property is maintained by careful tracking of `previous_stage` and `previous_epochs_in_stage`.

**Detailed Findings:**

#### P2-1: Unused Parameter in `_compute_synergy_bonus`

```python
def _compute_synergy_bonus(
    interaction_sum: float,
    boost_received: float,  # <-- NEVER USED
    synergy_weight: float = 0.1,
) -> float:
```

**Location:** Lines 1210-1233
**Impact:** The `boost_received` parameter is accepted but never used in the computation. This suggests either:
- Incomplete implementation (boost_received should factor into the bonus)
- API pollution from planned future functionality

**Recommendation:** Either use `boost_received` (perhaps as a multiplier or threshold) or remove the parameter.

---

#### P2-2: Potential Numerical Instability in PBRS Bonus

```python
def _contribution_pbrs_bonus(
    seed_info: SeedInfo,
    config: ContributionRewardConfig,
) -> float:
    phi_current = STAGE_POTENTIALS.get(seed_info.stage, 0.0)
    phi_current += min(
        seed_info.epochs_in_stage * config.epoch_progress_bonus,
        config.max_progress_bonus,
    )
    # ... similar for phi_prev
    return config.pbrs_weight * (config.gamma * phi_current - phi_prev)
```

**Location:** Lines 1165-1207
**Issue:** When `seed_info.epochs_in_stage == 0` (just transitioned) AND `seed_info.previous_epochs_in_stage == 0` (incorrect SeedInfo construction), the PBRS bonus will be computed incorrectly. The code logs a warning but continues with potentially wrong values.

**Impact:** If SeedInfo is constructed incorrectly upstream, PBRS telescoping property breaks, invalidating policy invariance guarantees.

**Recommendation:** Consider raising an exception instead of warning, or add a validation layer for SeedInfo construction.

---

#### P3-1: Magic Number in `_compute_synergy_bonus`

```python
raw_bonus = math.tanh(interaction_sum * 0.5)  # Why 0.5?
```

**Location:** Line 1232
**Impact:** The `0.5` scaling factor is undocumented. The docstring doesn't explain the choice.

**Recommendation:** Document the rationale or extract to a constant:
```python
SYNERGY_TANH_SCALE = 0.5  # Controls saturation rate: tanh(2.0) = 0.96
```

---

#### P3-2: Magic Number in `compute_scaffold_hindsight_credit`

```python
raw_credit = math.tanh(boost_given * beneficiary_improvement * 0.1)  # Why 0.1?
```

**Location:** Line 1268
**Impact:** The `0.1` factor is documented in the docstring ("The 0.1 scaling factor controls tanh saturation") - good practice.

**No action needed** - this is well-documented, unlike P3-1.

---

#### P3-3: Exp Overflow Guard is Correct but Could be Cleaner

```python
exp_arg = min(-config.attribution_sigmoid_steepness * total_imp, 700.0)
attribution_discount = 1.0 / (1.0 + math.exp(exp_arg))
```

**Location:** Lines 536-537
**Assessment:** Correct - `exp(709)` is roughly the float64 limit. The guard at 700 is conservative.

**Recommendation:** Consider documenting why 700:
```python
# Guard: math.exp(709.78) overflows float64; 700 is conservative
```

---

#### P4-1: Inconsistent Type Annotation

```python
def compute_loss_reward(
    action: int,  # <-- Should be LifecycleOp
    ...
) -> float:
```

**Location:** Line 1571
**Impact:** `action` is typed as `int` but `LifecycleOp` is used throughout. This doesn't break anything (IntEnum IS an int) but is inconsistent with other functions.

---

### 3. `reward_telemetry.py` (Telemetry Dataclass)

**Purpose:** Captures per-component reward breakdown for debugging and reward hacking detection.

**Architectural Observations:**

1. **slots=True** - Correct for hot-path dataclass; reduces memory overhead.
2. **to_dict()** - Uses explicit dict construction instead of `dataclasses.asdict()` for performance - good.

**Detailed Findings:**

#### P2-3: `shaped_reward_ratio` May Be Misleading

```python
shaped = (
    # Bonuses
    self.stage_bonus
    + self.pbrs_bonus
    + self.synergy_bonus
    + self.action_shaping
    + self.terminal_bonus
    + self.fossilize_terminal_bonus
    + self.hindsight_credit
    # Penalties (these shape behavior, so include in total)
    + self.compute_rent
    + self.alpha_shock
    + self.blending_warning
    + self.holding_warning
    + self.ratio_penalty
)
```

**Location:** Lines 84-99
**Issues:**
1. `stage_bonus` is defined in the dataclass but is NEVER SET by `compute_contribution_reward`. It's always 0.0.
2. `hindsight_credit` is defined but never set by the main reward function.
3. `fossilize_terminal_bonus` is set but is already INCLUDED in `terminal_bonus` (see rewards.py line 798).

**Impact:** The `shaped_reward_ratio` metric is computing a ratio that includes double-counted values and unset fields, making it unreliable for reward hacking detection.

---

## Cross-Cutting Integration Risks

### Risk 1: Component Sum Test Suite is Stale (P1)

**Files Affected:**
- `/home/john/esper-lite/tests/simic/properties/test_reward_invariants.py` (lines 76-85)
- `/home/john/esper-lite/tests/simic/test_reward_telemetry.py` (lines 84-92)

**Issue:** Both tests calculate component sums but **neither includes `synergy_bonus`**:

```python
# test_reward_invariants.py
component_sum = (
    components.bounded_attribution
    + components.blending_warning
    + components.holding_warning
    + components.pbrs_bonus
    + components.compute_rent
    + components.alpha_shock
    + components.action_shaping
    + components.terminal_bonus
)
# MISSING: synergy_bonus
```

```python
# test_reward_telemetry.py
computed_sum = (
    components.bounded_attribution
    + components.compute_rent
    + components.alpha_shock
    + components.pbrs_bonus
    + components.action_shaping
    + components.terminal_bonus
)
# MISSING: synergy_bonus, blending_warning, holding_warning
```

**Impact:** These tests will PASS but are not verifying the invariant correctly. If `synergy_bonus` is ever non-zero, the tests won't catch the discrepancy because they don't include it in the sum.

**Evidence:** The scaffolding rewards test file (`tests/simic/rewards/test_scaffolding_rewards.py`) shows `synergy_bonus` IS populated:
```python
assert components.synergy_bonus > 0, (
    f"Synergy bonus should be positive, got {components.synergy_bonus}"
)
```

---

### Risk 2: `base_acc_delta` is Dead Code (P3)

**Field in `RewardComponentsTelemetry`:**
```python
base_acc_delta: float = 0.0  # Comment says "legacy shaped reward"
```

**Impact:** This field is never set by any reward function. It's vestigial from an older reward design. Consider removing to reduce confusion.

---

### Risk 3: Telemetry Import Creates Circular Dependency Risk (P4)

```python
# rewards.py line 44
from .reward_telemetry import RewardComponentsTelemetry
```

**Assessment:** Currently safe because `reward_telemetry.py` has no imports from `rewards.py`. However, if someone adds a validation method to `RewardComponentsTelemetry` that imports from `rewards.py`, it will create a circular import.

**Recommendation:** Consider keeping telemetry dataclasses truly standalone (no validation that requires reward logic).

---

## Severity-Tagged Findings Summary

| ID | Severity | File | Line(s) | Description |
|----|----------|------|---------|-------------|
| P1-1 | P1 | test_reward_invariants.py | 76-85 | Component sum test missing `synergy_bonus` - test passes incorrectly |
| P1-2 | P1 | test_reward_telemetry.py | 84-92 | Component sum test missing multiple components |
| P2-1 | P2 | rewards.py | 1210-1212 | Unused `boost_received` parameter in `_compute_synergy_bonus` |
| P2-2 | P2 | rewards.py | 1186-1193 | PBRS bonus logs warning but continues with wrong values |
| P2-3 | P2 | reward_telemetry.py | 84-99 | `shaped_reward_ratio` includes unset/double-counted fields |
| P3-1 | P3 | rewards.py | 1232 | Undocumented magic number 0.5 in synergy bonus |
| P3-2 | P3 | reward_telemetry.py | 21 | `base_acc_delta` is dead code (never set) |
| P3-3 | P3 | reward_telemetry.py | 37 | `stage_bonus` is dead code (never set) |
| P3-4 | P3 | reward_telemetry.py | 43 | `hindsight_credit` is dead code (never set in main path) |
| P4-1 | P4 | rewards.py | 1571 | `action` typed as `int` instead of `LifecycleOp` |

---

## PyTorch-Specific Assessment

### Device Placement
**Status: N/A** - This module correctly operates in pure Python. Device placement for reward tensors is handled in `vectorized.py` where rewards are converted:
```python
reward = compute_reward(**reward_args)  # Returns float
# Later converted to tensor in rollout buffer
```

### Gradient Concerns
**Status: N/A** - Rewards are Python floats, inherently detached. No gradient tracking issues possible.

### torch.compile Compatibility
**Status: N/A** - No torch operations to compile.

### Memory Efficiency
**Status: Good** - Uses `slots=True` for telemetry dataclass. Reward computation is allocation-minimal (no intermediate objects created).

---

## Recommendations

### Must Fix (P1)
1. Update `test_reward_invariants.py` line 76-85 to include `synergy_bonus` in component sum
2. Update `test_reward_telemetry.py` line 84-92 to include all components (`synergy_bonus`, `blending_warning`, `holding_warning`)

### Should Fix (P2)
1. Either use or remove `boost_received` parameter from `_compute_synergy_bonus`
2. Review `shaped_reward_ratio` calculation for correctness (currently unreliable)
3. Consider raising exception instead of warning for PBRS construction errors

### Consider Fixing (P3/P4)
1. Remove dead fields from `RewardComponentsTelemetry`: `base_acc_delta`, `stage_bonus`, `hindsight_credit` (if not used)
2. Document magic number 0.5 in synergy bonus
3. Fix type annotation in `compute_loss_reward`

---

## Positive Observations

1. **Excellent PBRS Documentation** - The 50-line comment block (lines 70-114) explaining PBRS rationale, tuning history, and validation is exemplary.

2. **Anti-Gaming Design** - The reward function has multiple layers of defense against reward hacking:
   - Attribution discount for negative trajectories
   - Ratio penalty for contribution >> improvement
   - Ransomware signature detection
   - Legitimacy discount for rapid fossilization

3. **Config-Driven Design** - All magic numbers are in `ContributionRewardConfig` with sensible defaults and ablation flags.

4. **Property-Based Testing** - Extensive property tests exist for reward bounds, monotonicity, and anti-gaming properties.

5. **Telemetry Integration** - `RewardComponentsTelemetry` enables debugging without modifying reward logic.

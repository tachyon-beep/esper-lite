# Batch 6 Code Review: Simic Rewards Module

**Reviewer**: Senior Code Reviewer (Python/RL specialization)
**Date**: 2025-12-27
**Files Reviewed**:
1. `/home/john/esper-lite/src/esper/simic/rewards/__init__.py`
2. `/home/john/esper-lite/src/esper/simic/rewards/rewards.py`
3. `/home/john/esper-lite/src/esper/simic/rewards/reward_telemetry.py`

---

## Executive Summary

The Simic rewards module is a well-engineered reward computation system for reinforcement learning-based seed lifecycle control. The code demonstrates sophisticated domain knowledge of PBRS (Potential-Based Reward Shaping), anti-gaming mechanisms, and RL credit assignment. Test coverage is extensive with property-based tests for mathematical invariants.

**Overall Assessment**: High quality with minor issues requiring attention.

| Severity | Count |
|----------|-------|
| P0 (Critical) | 0 |
| P1 (Correctness) | 2 |
| P2 (Performance/Resource) | 1 |
| P3 (Code Quality) | 4 |
| P4 (Style/Minor) | 3 |

---

## File-by-File Analysis

### 1. `/home/john/esper-lite/src/esper/simic/rewards/__init__.py`

**Purpose**: Package entry point exporting reward functions, configs, and constants.

**Observations**:
- Clean re-export pattern with explicit `__all__`
- Exports internal helpers (`_contribution_pbrs_bonus`, etc.) for testing - appropriate
- Docstring accurately describes module purpose

**Concerns**: None identified.

---

### 2. `/home/john/esper-lite/src/esper/simic/rewards/rewards.py`

**Purpose**: Core reward computation with PBRS, counterfactual validation, and anti-gaming mechanisms.

**Strengths**:
- Excellent documentation of PBRS theory and tuning history (lines 70-114)
- Unified `STAGE_POTENTIALS` dictionary prevents telescoping breakage
- Comprehensive anti-gaming: ransomware detection, ratio penalty, attribution discount
- Runtime gamma validation prevents misconfiguration (lines 498-502)
- Multiple reward modes (SHAPED, SPARSE, MINIMAL, SIMPLIFIED) for experimentation

**P1 Findings**:

#### P1-1: Component Sum Missing `synergy_bonus` in Some Tests

**Location**: `/home/john/esper-lite/tests/simic/properties/test_reward_invariants.py`, lines 76-85

The property test `test_components_sum_to_total` does not include `synergy_bonus` in its sum:

```python
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
```

The `synergy_bonus` is computed and added to reward (lines 714-721 in rewards.py) but missing from test verification. This could hide bugs where synergy_bonus causes component sum mismatch.

**Recommendation**: Add `+ components.synergy_bonus` to the component sum in the property test.

**Cross-reference**: Same issue in `/home/john/esper-lite/tests/simic/test_rewards.py` line 411-418 in `TestContributionRewardComponents.test_components_sum_to_total` and `/home/john/esper-lite/tests/simic/test_reward_telemetry.py` line 84-92.

---

#### P1-2: `_compute_synergy_bonus` Ignores `boost_received` Parameter

**Location**: `/home/john/esper-lite/src/esper/simic/rewards/rewards.py`, lines 1210-1233

```python
def _compute_synergy_bonus(
    interaction_sum: float,
    boost_received: float,  # <-- NEVER USED
    synergy_weight: float = 0.1,
) -> float:
    """..."""
    if interaction_sum <= 0:
        return 0.0
    raw_bonus = math.tanh(interaction_sum * 0.5)
    return raw_bonus * synergy_weight
```

The `boost_received` parameter is accepted but never used. Either:
- It's dead code from a removed feature, or
- The implementation is incomplete

The docstring mentions `boost_received` as "Maximum single interaction" but the implementation ignores it.

**Recommendation**: Either remove the parameter or incorporate it into the bonus calculation as intended. Check `SeedInfo.boost_received` usage in `/home/john/esper-lite/tests/simic/rewards/test_scaffolding_rewards.py` to determine original intent.

---

**P2 Findings**:

#### P2-1: Repeated Import of `TelemetryEvent` and `TelemetryEventType` Inside Functions

**Location**: `/home/john/esper-lite/src/esper/simic/rewards/rewards.py`, lines 1390-1391 and 1436-1437

```python
def _check_reward_hacking(...) -> bool:
    from esper.leyline import TelemetryEvent, TelemetryEventType  # Inside function
    ...

def _check_ransomware_signature(...) -> bool:
    from esper.leyline import TelemetryEvent, TelemetryEventType  # Same import again
    ...
```

These functions are called on every timestep when slot_id/seed_id are provided. While Python caches imports, the overhead is non-zero in a hot path.

**Recommendation**: Move imports to module level (near line 35) since `esper.leyline` is already imported there.

---

**P3 Findings**:

#### P3-1: Magic Numbers in Sigmoid Discount Clamping

**Location**: `/home/john/esper-lite/src/esper/simic/rewards/rewards.py`, line 536

```python
exp_arg = min(-config.attribution_sigmoid_steepness * total_imp, 700.0)
```

The value `700.0` is the float64 exp() overflow protection threshold, but it appears without explanation at point of use.

**Recommendation**: Extract to a named constant with comment, e.g.:
```python
_EXP_OVERFLOW_GUARD = 700.0  # exp(709) is float64 limit; guard at 700
```

---

#### P3-2: Inconsistent Handling of None Values in SeedInfo.from_seed_state

**Location**: `/home/john/esper-lite/src/esper/simic/rewards/rewards.py`, lines 362-387

The method directly accesses `seed_state.metrics` and its attributes without None checks on individual metric fields:

```python
if metrics:
    improvement = metrics.current_val_accuracy - metrics.accuracy_at_stage_start
    total_improvement = metrics.total_improvement
    # ...
```

However, `metrics.total_improvement` could potentially be None if `SeedMetrics` has optional fields. The code assumes all fields are populated when `metrics` exists.

**Recommendation**: Verify `SeedMetrics` contract guarantees these fields are always populated, or add explicit defaults:
```python
total_improvement = metrics.total_improvement if metrics.total_improvement is not None else 0.0
```

---

#### P3-3: Type Annotation Mismatch in `compute_loss_reward`

**Location**: `/home/john/esper-lite/src/esper/simic/rewards/rewards.py`, line 1570

```python
def compute_loss_reward(
    action: int,  # Should be LifecycleOp, not int
    ...
```

Other reward functions correctly use `action: LifecycleOp`, but `compute_loss_reward` uses `int`. While this works due to IntEnum, it reduces type safety and IDE support.

**Recommendation**: Change to `action: LifecycleOp` for consistency.

---

#### P3-4: Redundant or-clause in Terminal Bonus

**Location**: `/home/john/esper-lite/src/esper/simic/rewards/rewards.py`, line 1615

```python
achievable_range = config.achievable_range or 1.0
```

The `achievable_range` property is computed as `self.baseline_loss - self.target_loss`. With defaults `baseline_loss=2.3` and `target_loss=0.3`, this is `2.0`. The `or 1.0` fallback is only hit if `achievable_range` is `0` (would require `baseline_loss == target_loss`).

**Recommendation**: If this is intentional (defensive against misconfiguration), add a comment. Otherwise, consider raising ValueError for invalid config instead of silent fallback.

---

**P4 Findings**:

#### P4-1: Unused Import in SeedInfo.from_seed_state

**Location**: `/home/john/esper-lite/src/esper/simic/rewards/rewards.py`, line 33

```python
from typing import Any, NamedTuple, cast
```

`Any` is used in `from_seed_state(seed_state: Any, ...)` instead of proper type annotation. Consider importing the actual type from kasmina for better type checking:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from esper.kasmina.slot import SeedState
```

---

#### P4-2: Empty Comment Blocks

**Location**: `/home/john/esper-lite/src/esper/simic/rewards/rewards.py`, lines 132-144

```python
# =============================================================================
# Reward Configuration
# =============================================================================


# =============================================================================
# Loss-Primary Reward Configuration (Phase 2)
# =============================================================================


# =============================================================================
# Contribution-Primary Reward Configuration (uses counterfactual validation)
# =============================================================================
```

Three consecutive section headers with no content between them before `ContributionRewardConfig`. Clean up or consolidate.

---

#### P4-3: Docstring Parameter Order Mismatch

**Location**: `/home/john/esper-lite/src/esper/simic/rewards/rewards.py`, `compute_contribution_reward` docstring

The docstring lists parameters in a different order than the function signature. While not strictly wrong, it makes cross-referencing harder.

---

### 3. `/home/john/esper-lite/src/esper/simic/rewards/reward_telemetry.py`

**Purpose**: Telemetry dataclass for reward component breakdown.

**Strengths**:
- `slots=True` for memory efficiency
- Explicit `to_dict()` instead of `asdict()` for performance (per comment)
- `shaped_reward_ratio` property is useful for hacking detection

**P3 Findings**:

#### P3-5: `shaped_reward_ratio` Includes Some Terms Twice?

**Location**: `/home/john/esper-lite/src/esper/simic/rewards/reward_telemetry.py`, lines 84-99

The comment says "PyTorch Expert Review 2025-12-26: Added all shaping terms" but includes `ratio_penalty` which is NOT a shaping term - it's part of the attribution computation (applied before weight multiplication per line 616 in rewards.py).

Including `ratio_penalty` in shaped_reward_ratio calculation may give misleading results when ratio_penalty is non-zero.

**Recommendation**: Review whether `ratio_penalty` belongs in the shaping terms sum or should be excluded since it modifies the primary attribution signal rather than being separate shaping.

---

## Cross-Cutting Concerns

### Test Coverage

**Excellent coverage** via property-based tests:
- `test_reward_properties.py`: Bounds, monotonicity, PBRS
- `test_reward_invariants.py`: Finiteness, boundedness, composition
- `test_reward_antigaming.py`: Ransomware, fossilization farming, ratio penalty
- `test_reward_semantics.py`: Fossilized behavior, PRUNE inversion, terminal bonus timing

**Gap identified**: The composition tests are missing `synergy_bonus` (P1-1 above).

### Integration with Leyline Contracts

- Correctly imports `DEFAULT_GAMMA`, `DEFAULT_MIN_FOSSILIZE_CONTRIBUTION`, `MIN_HOLDING_EPOCHS`, `MIN_PRUNE_AGE` from leyline
- `STAGE_POTENTIALS` aligns with `SeedStage` enum values (0-10, skipping 5)
- `TelemetryEventType.REWARD_HACKING_SUSPECTED` correctly used for anomaly detection

### PBRS Correctness

The PBRS implementation correctly implements Ng et al. (1999):
- `F(s, s') = gamma * phi(s') - phi(s)`
- Gamma validated at runtime (line 498-502)
- Telescoping property tested in `tests/simic/properties/test_pbrs_properties.py`
- Unified `STAGE_POTENTIALS` prevents inconsistency across reward functions

### Anti-Gaming Mechanisms

Well-designed multi-layer defense:
1. **Attribution discount**: Sigmoid on `total_improvement` zeros rewards for regressing seeds
2. **Ratio penalty**: Catches `contribution >> improvement` (dependency gaming)
3. **Ransomware signature detection**: Emits telemetry for `contribution > 1.0 AND total_improvement < -0.2`
4. **Legitimacy discount**: Fossilize bonus reduced for short HOLDING periods
5. **Penalty anti-stacking**: When `attribution_discount < 0.5`, skips `ratio_penalty` and `holding_warning`

---

## Severity-Tagged Findings Summary

| ID | Severity | Location | Description |
|----|----------|----------|-------------|
| P1-1 | P1 | tests/simic/properties/test_reward_invariants.py:76 | Component sum missing `synergy_bonus` |
| P1-2 | P1 | rewards.py:1210 | `boost_received` parameter unused in `_compute_synergy_bonus` |
| P2-1 | P2 | rewards.py:1390,1436 | Repeated import inside hot-path functions |
| P3-1 | P3 | rewards.py:536 | Magic number 700.0 for exp overflow |
| P3-2 | P3 | rewards.py:370 | Potential None access in SeedInfo.from_seed_state |
| P3-3 | P3 | rewards.py:1570 | Type `int` instead of `LifecycleOp` |
| P3-4 | P3 | rewards.py:1615 | Silent fallback for achievable_range |
| P3-5 | P3 | reward_telemetry.py:98 | `ratio_penalty` in shaping terms questionable |
| P4-1 | P4 | rewards.py:33 | `Any` type instead of proper import |
| P4-2 | P4 | rewards.py:132-144 | Empty section headers |
| P4-3 | P4 | rewards.py:docstring | Parameter order mismatch |

---

## Recommendations

### Must Fix (P1)

1. **P1-1**: Update the three composition tests to include `synergy_bonus` in component sum validation. Files:
   - `/home/john/esper-lite/tests/simic/properties/test_reward_invariants.py`
   - `/home/john/esper-lite/tests/simic/test_rewards.py`
   - `/home/john/esper-lite/tests/simic/test_reward_telemetry.py`

2. **P1-2**: Decide on `boost_received` - either remove from signature or implement its contribution to synergy bonus.

### Should Fix (P2-P3)

3. **P2-1**: Move telemetry imports to module level.
4. **P3-1**: Extract exp overflow guard to named constant.
5. **P3-3**: Change `action: int` to `action: LifecycleOp`.

### Nice to Have (P4)

6. Clean up empty section headers.
7. Align docstring parameter order with function signature.

---

## Positive Observations

1. **Documentation Excellence**: The PBRS rationale comment block (lines 70-114) is exemplary - explains theory, values, tuning history, and validation approach.

2. **Defensive Runtime Checks**: The gamma mismatch ValueError (line 498-502) prevents silent misconfiguration that would invalidate PBRS guarantees.

3. **Multi-Mode Architecture**: The `RewardMode` enum with dispatcher pattern enables clean A/B testing of reward functions.

4. **Anti-Gaming Depth**: The layered defense (attribution discount -> ratio penalty -> ransomware telemetry) shows sophisticated understanding of RL gaming patterns.

5. **Test Quality**: Property-based tests with Hypothesis cover the mathematical invariants that reward functions must maintain.

---

*Report generated by Senior Code Reviewer specializing in Python/RL*

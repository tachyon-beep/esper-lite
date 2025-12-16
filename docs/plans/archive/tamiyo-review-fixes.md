# Tamiyo Code Review Fixes - Implementation Plans

**Date:** 2025-12-16
**Source:** External code review + DRL/PyTorch specialist validation

---

## P1-A: Stabilisation Latch Counts Loss Regression as "Stable"

### Problem

**File:** `src/esper/tamiyo/tracker.py:110-116`

```python
relative_improvement = loss_delta / self._prev_loss
is_stable_epoch = (
    relative_improvement < self.stabilization_threshold and
    val_loss < self._prev_loss * 1.5
)
```

When loss gets worse (`val_loss > prev_loss`), `loss_delta` becomes negative, making `relative_improvement` negative. Negative values pass the `< 0.03` threshold check, so epochs where training regresses count toward stabilization.

### Impact

- Host can "stabilize" during a period of loss regression
- Germination becomes enabled prematurely
- Incorrect credit assignment in RL training

### Proposed Fix

**Option A (Strict - no regression allowed):**
```python
is_stable_epoch = (
    loss_delta >= 0 and  # Must not be regressing
    relative_improvement < self.stabilization_threshold and
    val_loss < self._prev_loss * 1.5
)
```

**Option B (Tolerant - allow tiny regression):**
```python
is_stable_epoch = (
    relative_improvement > -0.005 and  # Allow 0.5% regression (noise)
    relative_improvement < self.stabilization_threshold and
    val_loss < self._prev_loss * 1.5
)
```

### Recommendation

**Option A** - strict no-regression. Rationale:
- "Stabilized" should mean learning has plateaued, not oscillating
- The 1.5x divergence check handles catastrophic regression
- Small negative deltas during true stabilization are unlikely

### Tests to Add

1. `test_stabilization_not_triggered_during_regression` - loss increases each epoch, should never stabilize
2. `test_stabilization_requires_non_negative_improvement` - verify boundary behavior

### Risk Assessment

**LOW** - Single condition change, well-tested module, clear semantics.

---

## P1-B: Blueprint `getattr` Crashes on Non-CNN Topologies (BUG-011)

### Problem

**File:** `src/esper/tamiyo/heuristic.py:148-153`

```python
blueprint_id = self._get_next_blueprint()
germinate_action = getattr(Action, f"GERMINATE_{blueprint_id.upper()}")
```

`blueprint_rotation` defaults to CNN blueprints (`conv_light`, `conv_heavy`, etc.). If topology is `transformer`, the Action enum won't have `GERMINATE_CONV_LIGHT`, causing `AttributeError`.

### Impact

- Training crashes before first checkpoint on non-CNN topologies
- Existing BUG-011 document describes this issue

### Proposed Fix

**Option A (Validation at init):**
```python
def __init__(self, topology: str = "cnn", config: HeuristicConfig | None = None):
    self.topology = topology
    self.config = config or HeuristicConfig()

    # Build action enum for this topology
    self._action_enum = build_action_enum(topology)

    # Validate blueprint_rotation against available actions
    available_blueprints = {
        name[len("GERMINATE_"):].lower()
        for name in dir(self._action_enum)
        if name.startswith("GERMINATE_")
    }
    invalid = set(self.config.blueprint_rotation) - available_blueprints
    if invalid:
        raise ValueError(
            f"blueprint_rotation contains blueprints not available for "
            f"topology '{topology}': {invalid}. "
            f"Available: {sorted(available_blueprints)}"
        )
```

**Option B (Fallback with warning):**
Skip invalid blueprints in rotation with a warning log. Less strict but avoids crash.

### Recommendation

**Option A** - fail fast with clear error. Rationale:
- Config error should be caught at init, not during training
- Clear error message tells user exactly what to fix
- Consistent with "no silent failures" principle

### Tests to Add

1. `test_heuristic_rejects_invalid_blueprints_for_topology`
2. `test_heuristic_accepts_valid_blueprints_for_topology`
3. `test_transformer_topology_with_default_config_fails` (documents the issue)

### Risk Assessment

**LOW** - Validation at init, fails fast, no runtime behavior change for valid configs.

---

## P2-A: Plateau Threshold Scale Documentation

### Problem

**File:** `src/esper/tamiyo/tracker.py:51`

```python
plateau_threshold: float = 0.5  # Min improvement to not count as plateau
```

The threshold `0.5` assumes accuracies are on 0-100 scale (0.5 percentage points). If metrics ever use 0-1 scale, this becomes 50% per epoch (broken).

**Also affects:** `src/esper/leyline/signals.py` - `TrainingMetrics` schema has no scale documentation.

### Impact

- Potential for silent misconfiguration if metric scales change
- Contract ambiguity between modules

### Proposed Fix

1. **Add scale documentation to `TrainingMetrics`:**
```python
@dataclass
class TrainingMetrics:
    """Training metrics from the host model.

    Scale conventions:
    - train_accuracy, val_accuracy: 0-100 (percentage points)
    - train_loss, val_loss: raw loss values (typically 0-10 range)
    """
    epoch: int = 0
    train_loss: float = float('inf')
    val_loss: float = float('inf')
    train_accuracy: float = 0.0  # 0-100 scale
    val_accuracy: float = 0.0    # 0-100 scale
```

2. **Rename threshold for clarity:**
```python
plateau_threshold_pct_points: float = 0.5  # Min improvement in percentage points
```

3. **Add validation in `SignalTracker.update()`:**
```python
if val_accuracy > 100 or val_accuracy < 0:
    logger.warning(f"val_accuracy {val_accuracy} outside expected 0-100 range")
```

### Recommendation

Implement all three changes. The rename is the most important for self-documentation.

### Risk Assessment

**MEDIUM** - Touches contract definition, but no behavioral change. Requires updating references to `plateau_threshold`.

---

## P2-B: Ransomware Detection Not Implemented

### Problem

**File:** `src/esper/tamiyo/heuristic.py:220-230`

Seeds with high counterfactual contribution (model needs them) but negative total improvement (model is worse overall) can get fossilized. These "ransomware" seeds created dependencies without adding value.

### Impact

- Heuristic baseline fossilizes harmful seeds
- RL agents learn to avoid via reward shaping, but heuristic doesn't benefit
- Existing TODO documents this as future functionality

### Proposed Fix

```python
# In _decide_seed_management, PROBATIONARY branch:
if stage == SeedStage.PROBATIONARY:
    contribution = seed.metrics.counterfactual_contribution
    total_improvement = seed.metrics.total_improvement

    # Ransomware detection: high counterfactual + negative total = bad seed
    if contribution is not None and total_improvement is not None:
        is_ransomware = (
            contribution > self.config.ransomware_contribution_threshold and
            total_improvement < self.config.ransomware_improvement_threshold
        )
        if is_ransomware:
            return TamiyoDecision(
                action=Action.CULL,
                target_seed_id=seed.seed_id,
                reason=f"Ransomware pattern: high counterfactual ({contribution:.3f}) "
                       f"but negative improvement ({total_improvement:.3f})",
                confidence=0.9,
            )

    # ... existing fossilization logic
```

**Config additions:**
```python
@dataclass
class HeuristicConfig:
    # ... existing fields ...
    ransomware_contribution_threshold: float = 0.1  # Counterfactual > this
    ransomware_improvement_threshold: float = 0.0   # Total improvement < this
```

### Tests to Enable

- `test_ransomware_pattern_detected` (currently skipped)

### Risk Assessment

**MEDIUM** - New decision logic, but conservative (only culls clear bad actors). Should be tested thoroughly.

---

## Implementation Order

1. **P1-A** (Stabilisation latch) - Highest impact, simplest fix
2. **P1-B** (Blueprint validation) - Fixes crash on non-CNN
3. **P2-A** (Scale documentation) - Contract clarity
4. **P2-B** (Ransomware detection) - New feature, lowest priority

---

## Questions for Review

1. **P1-A:** Strict (Option A) or tolerant (Option B) for stabilisation?
2. **P1-B:** Fail fast (Option A) or fallback with warning (Option B)?
3. **P2-A:** Is renaming `plateau_threshold` to `plateau_threshold_pct_points` worth the churn?
4. **P2-B:** Should ransomware detection block this branch, or defer to a future milestone?

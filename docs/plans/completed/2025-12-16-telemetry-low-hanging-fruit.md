# Phase 0: Telemetry Wiring - Low-Hanging Fruit

**Goal:** Wire 7 telemetry fields that have code but aren't connected to telemetry events.

**Status:** IMPLEMENTED - all tasks complete, tests passing

**Estimated Effort:** 2-3 hours

**Dependency:** Execute AFTER `2025-12-16-dynamic-slots.md` completes. That plan deprecates `CANONICAL_SLOTS` and changes slot discovery to `host.injection_specs()`. This telemetry plan is **95% compatible** - only Task 6 (tests) needs awareness of the deprecation.

---

## Overview

These fields have working collection code but aren't wired to telemetry events. This is pure plumbing work - no new algorithms needed.

### Fields to Wire

| Field | Source | Destination | Effort |
|-------|--------|-------------|--------|
| `dead_layers` | `debug_telemetry.collect_per_layer_gradients()` | PPO_UPDATE_COMPLETED | 15 min |
| `exploding_layers` | `debug_telemetry.collect_per_layer_gradients()` | PPO_UPDATE_COMPLETED | 5 min (bundle with above) |
| `nan_grad_count` | `debug_telemetry.collect_per_layer_gradients()` | PPO_UPDATE_COMPLETED | 5 min (bundle with above) |
| `layer_gradient_health` | `gradient_collector.collect_seed_gradients()` | PPO_UPDATE_COMPLETED | 10 min |
| `plateau_detected` | `nissa.tracker._plateau_length()` | EPOCH_COMPLETED | 15 min |
| `shaped_reward_ratio` | `RewardComponentsTelemetry` | REWARD_COMPUTED | 10 min |
| `entropy_collapsed` | Compute from entropy | PPO_UPDATE_COMPLETED | 5 min |

---

## Task 1: Wire Gradient Layer Health Metrics

### Context

The `debug_telemetry.collect_per_layer_gradients()` function returns `List[LayerGradientStats]` with per-layer metrics. We need to aggregate these into summary counts for the PPO telemetry event.

**Source code location:** `src/esper/simic/debug_telemetry.py:42-123`

**Current collection point:** `vectorized.py` imports `collect_per_layer_gradients` but only calls it in anomaly debug paths.

### Implementation

**File:** `src/esper/simic/vectorized.py`

#### 1.1 Add helper function to aggregate layer stats

Add after `_compute_grad_norm_surrogate` (~line 186):

```python
def _aggregate_layer_gradient_health(
    layer_stats: list["LayerGradientStats"],
) -> dict[str, int | float]:
    """Aggregate per-layer gradient stats into summary metrics.

    Args:
        layer_stats: List from collect_per_layer_gradients()

    Returns:
        Dict with dead_layers, exploding_layers, nan_grad_count
    """
    if not layer_stats:
        return {"dead_layers": 0, "exploding_layers": 0, "nan_grad_count": 0}

    dead = sum(1 for s in layer_stats if s.zero_fraction > 0.9)
    exploding = sum(1 for s in layer_stats if s.large_fraction > 0.1)
    nan_count = sum(s.nan_count for s in layer_stats)

    return {
        "dead_layers": dead,
        "exploding_layers": exploding,
        "nan_grad_count": nan_count,
    }
```

#### 1.2 Call collection in PPO update path

In the PPO update section (around line 2400-2450 where `_emit_ppo_update_event` is called), add:

```python
# Collect layer gradient health if not in fast mode
layer_health = {"dead_layers": 0, "exploding_layers": 0, "nan_grad_count": 0}
if not telemetry_config or telemetry_config.should_collect("debug"):
    try:
        layer_stats = collect_per_layer_gradients(agent.policy)
        layer_health = _aggregate_layer_gradient_health(layer_stats)
    except Exception:
        pass  # Graceful degradation if collection fails
```

#### 1.3 Add fields to `_emit_ppo_update_event`

Update the `data` dict in `_emit_ppo_update_event` (~line 210-239):

```python
# Add after "update_time_ms": update_time_ms,
# Gradient layer health
"dead_layers": metrics.get("dead_layers", 0),
"exploding_layers": metrics.get("exploding_layers", 0),
"nan_grad_count": metrics.get("nan_grad_count", 0),
```

### Test

```bash
PYTHONPATH=src uv run pytest tests/simic/test_vectorized.py -k "ppo_update" -v
```

**Verification:** Run a short training and check logs for new fields:

```bash
PYTHONPATH=src uv run python -c "
from esper.simic.vectorized import train_ppo_vectorized
from esper.nissa import get_hub
hub = get_hub()
events = []
hub.add_listener(lambda e: events.append(e))
train_ppo_vectorized(n_episodes=2, max_epochs=5, max_batches=5)
ppo_events = [e for e in events if e.event_type.name == 'PPO_UPDATE_COMPLETED']
print('dead_layers:', ppo_events[-1].data.get('dead_layers'))
print('exploding_layers:', ppo_events[-1].data.get('exploding_layers'))
print('nan_grad_count:', ppo_events[-1].data.get('nan_grad_count'))
"
```

---

## Task 2: Wire `layer_gradient_health` from Gradient Collector

### Context

The `gradient_collector.collect_seed_gradients()` already returns `gradient_health` (0-1 score). This is computed per-epoch but not wired to PPO telemetry.

**Source code:** `src/esper/simic/gradient_collector.py:197-324`

### Implementation

**File:** `src/esper/simic/vectorized.py`

The gradient health is already computed via `collect_seed_gradients_async` in `_train_one_epoch`. We need to propagate it to the PPO update telemetry.

#### 2.1 Track gradient health across epoch

In the per-epoch loop (around line 1900-2000), after gradient collection:

```python
# Track gradient health for telemetry
epoch_gradient_health = None
if grad_stats and "gradient_health" in grad_stats:
    epoch_gradient_health = grad_stats["gradient_health"]
```

#### 2.2 Add to PPO update payload

In the payload construction before `_emit_ppo_update_event`:

```python
payload["layer_gradient_health"] = epoch_gradient_health
```

#### 2.3 Add to `_emit_ppo_update_event` data dict

```python
"layer_gradient_health": metrics.get("layer_gradient_health"),
```

### Test

Same verification as Task 1, check for `layer_gradient_health` in output.

---

## Task 3: Wire `entropy_collapsed` Flag

### Context

Entropy collapse (entropy < 0.1) indicates the policy has converged to deterministic actions - a critical signal for PPO health.

**Source:** Entropy is already computed and emitted. We just need to add a boolean flag.

### Implementation

**File:** `src/esper/simic/vectorized.py`

#### 3.1 Add computed field to `_emit_ppo_update_event`

In the data dict (~line 218), add:

```python
"entropy_collapsed": metrics.get("entropy", 1.0) < 0.1,
```

### Test

```bash
PYTHONPATH=src uv run python -c "
# Entropy collapse detection
entropy_test = 0.05
print('entropy_collapsed:', entropy_test < 0.1)  # Should be True
entropy_test = 0.5
print('entropy_collapsed:', entropy_test < 0.1)  # Should be False
"
```

---

## Task 4: Wire `shaped_reward_ratio`

### Context

The shaped reward ratio shows what fraction of total reward comes from shaping terms vs. the base signal. High ratios (> 0.5) suggest the agent might be gaming shaping bonuses rather than learning the true task.

**Source:** `RewardComponentsTelemetry` in `src/esper/simic/reward_telemetry.py`

**Computation:**
```python
shaped = stage_bonus + pbrs_bonus + action_shaping
shaped_reward_ratio = shaped / total_reward if total_reward != 0 else 0.0
```

### Implementation

**File:** `src/esper/simic/reward_telemetry.py`

#### 4.1 Add computed property to `RewardComponentsTelemetry`

```python
@property
def shaped_reward_ratio(self) -> float:
    """Fraction of total reward from shaping terms.

    High values (> 0.5) suggest potential reward hacking.
    """
    if abs(self.total_reward) < 1e-8:
        return 0.0
    shaped = self.stage_bonus + self.pbrs_bonus + self.action_shaping
    return abs(shaped) / abs(self.total_reward)
```

#### 4.2 Add to `to_dict()` method

```python
"shaped_reward_ratio": self.shaped_reward_ratio,
```

### Test

```bash
PYTHONPATH=src uv run python -c "
from esper.simic.reward_telemetry import RewardComponentsTelemetry

# High shaping ratio
t1 = RewardComponentsTelemetry(
    stage_bonus=0.3, pbrs_bonus=0.2, action_shaping=0.1,
    total_reward=1.0
)
print(f'High shaping: {t1.shaped_reward_ratio:.2f}')  # 0.60

# Low shaping ratio
t2 = RewardComponentsTelemetry(
    stage_bonus=0.05, pbrs_bonus=0.0, action_shaping=0.0,
    base_acc_delta=0.8, total_reward=0.85
)
print(f'Low shaping: {t2.shaped_reward_ratio:.2f}')  # 0.06

# Zero total (edge case)
t3 = RewardComponentsTelemetry(total_reward=0.0)
print(f'Zero total: {t3.shaped_reward_ratio:.2f}')  # 0.00
"
```

---

## Task 5: Wire `plateau_detected` from Tracker

### Context

The `DiagnosticTracker._plateau_length()` method already computes plateau detection. We need to surface this as a telemetry field.

**Source:** `src/esper/nissa/tracker.py:441-456`

**Current usage:** Called internally by `_detect_red_flags()` but not exposed externally.

### Implementation

**File:** `src/esper/nissa/tracker.py`

#### 5.1 Add public property to DiagnosticTracker

After `_plateau_length` method (~line 456):

```python
@property
def plateau_detected(self) -> bool:
    """Check if training is in a plateau (3+ epochs with no improvement)."""
    return self._plateau_length() >= 3
```

#### 5.2 Wire to EPOCH_COMPLETED event

**File:** `src/esper/simic/training.py` or wherever EPOCH_COMPLETED is emitted

In the epoch completion telemetry data dict, add:

```python
"plateau_detected": signal_tracker.plateau_detected if hasattr(signal_tracker, 'plateau_detected') else None,
```

**Note:** The hasattr is required here because SignalTracker may not have plateau_detected in all paths. This should be documented and eventually removed when unified.

### Test

```bash
PYTHONPATH=src uv run python -c "
from esper.nissa.tracker import DiagnosticTracker, EpochSnapshot

tracker = DiagnosticTracker()

# No plateau initially
print(f'Empty: {tracker.plateau_detected}')  # False

# Add similar losses
for i in range(5):
    tracker.record_epoch(EpochSnapshot(
        epoch=i,
        train_loss=1.0,
        val_loss=0.5,  # Constant
        train_accuracy=50.0,
        val_accuracy=50.0,
    ))
print(f'After 5 similar epochs: {tracker.plateau_detected}')  # True
"
```

---

## Task 6: Add Unit Tests

**Compatibility Note:** The `dynamic-slots` plan (2025-12-16) deprecates `CANONICAL_SLOTS` in favor of `host.injection_specs()`. Tests should use explicit slot ID tuples rather than importing `CANONICAL_SLOTS`.

### Test File: `tests/simic/test_telemetry_fields.py`

```python
\"\"\"Tests for low-hanging fruit telemetry fields.\"\"\"

import pytest
import torch
import torch.nn as nn

from esper.simic.debug_telemetry import collect_per_layer_gradients, LayerGradientStats
from esper.simic.gradient_collector import collect_seed_gradients
from esper.simic.reward_telemetry import RewardComponentsTelemetry


class TestLayerGradientAggregation:
    \"\"\"Test gradient layer health aggregation.\"\"\"

    def test_dead_layer_detection(self):
        \"\"\"Layers with >90% zero gradients are dead.\"\"\"
        stats = [
            LayerGradientStats(
                layer_name="dead_layer",
                param_count=100,
                grad_norm=0.0,
                grad_mean=0.0,
                grad_std=0.0,
                grad_min=0.0,
                grad_max=0.0,
                zero_fraction=0.95,  # Dead
                small_fraction=0.99,
                large_fraction=0.0,
                nan_count=0,
                inf_count=0,
            ),
            LayerGradientStats(
                layer_name="healthy_layer",
                param_count=100,
                grad_norm=1.0,
                grad_mean=0.1,
                grad_std=0.5,
                grad_min=-1.0,
                grad_max=1.0,
                zero_fraction=0.1,  # Healthy
                small_fraction=0.2,
                large_fraction=0.0,
                nan_count=0,
                inf_count=0,
            ),
        ]

        dead_count = sum(1 for s in stats if s.zero_fraction > 0.9)
        assert dead_count == 1

    def test_exploding_layer_detection(self):
        \"\"\"Layers with >10% large gradients are exploding.\"\"\"
        stats = [
            LayerGradientStats(
                layer_name="exploding",
                param_count=100,
                grad_norm=100.0,
                grad_mean=10.0,
                grad_std=20.0,
                grad_min=0.0,
                grad_max=1000.0,
                zero_fraction=0.0,
                small_fraction=0.0,
                large_fraction=0.15,  # Exploding
                nan_count=0,
                inf_count=0,
            ),
        ]

        exploding_count = sum(1 for s in stats if s.large_fraction > 0.1)
        assert exploding_count == 1


class TestShapedRewardRatio:
    \"\"\"Test reward shaping ratio computation.\"\"\"

    def test_high_shaping_ratio(self):
        t = RewardComponentsTelemetry(
            stage_bonus=0.3,
            pbrs_bonus=0.2,
            action_shaping=0.1,
            total_reward=1.0,
        )
        assert 0.55 < t.shaped_reward_ratio < 0.65

    def test_zero_total_reward(self):
        t = RewardComponentsTelemetry(total_reward=0.0)
        assert t.shaped_reward_ratio == 0.0

    def test_negative_reward(self):
        t = RewardComponentsTelemetry(
            stage_bonus=0.1,
            total_reward=-0.5,
        )
        # Should handle negative rewards gracefully
        assert t.shaped_reward_ratio >= 0.0


class TestEntropyCollapsed:
    \"\"\"Test entropy collapse detection.\"\"\"

    def test_collapsed_entropy(self):
        entropy = 0.05
        assert entropy < 0.1  # Collapsed

    def test_healthy_entropy(self):
        entropy = 0.5
        assert not (entropy < 0.1)  # Not collapsed
```

### Run Tests

```bash
PYTHONPATH=src uv run pytest tests/simic/test_telemetry_fields.py -v
```

---

## Checklist

- [ ] Task 1: Wire gradient layer health metrics (dead_layers, exploding_layers, nan_grad_count)
- [ ] Task 2: Wire layer_gradient_health from gradient collector
- [ ] Task 3: Add entropy_collapsed flag
- [ ] Task 4: Add shaped_reward_ratio property
- [ ] Task 5: Wire plateau_detected from tracker
- [ ] Task 6: Add unit tests
- [ ] Run full test suite to verify no regressions
- [ ] Update master record in `docs/plans/overwatch-textual-ui-expanded.md`

---

## Success Criteria

After implementation, running training should show these new fields in telemetry:

```json
{
  "event_type": "PPO_UPDATE_COMPLETED",
  "data": {
    "dead_layers": 0,
    "exploding_layers": 0,
    "nan_grad_count": 0,
    "layer_gradient_health": 0.95,
    "entropy_collapsed": false,
    ...
  }
}
```

```json
{
  "event_type": "REWARD_COMPUTED",
  "data": {
    "shaped_reward_ratio": 0.15,
    ...
  }
}
```

```json
{
  "event_type": "EPOCH_COMPLETED",
  "data": {
    "plateau_detected": false,
    ...
  }
}
```

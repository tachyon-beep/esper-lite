# Telemetry Record: [TELE-610] Episode Statistics

> **Status:** `[x] Planned` `[x] In Progress` `[ ] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-610` |
| **Name** | Episode Statistics |
| **Category** | `environment` |
| **Priority** | `P1-important` |
| **Type** | Grouped record (EpisodeStats dataclass) |

## 2. Purpose

### What question does this answer?

> "What are the aggregate episode-level training characteristics? Are episodes completing successfully, timing out, or terminating early? How efficient is the policy at taking key actions (germinate, prune, fossilize)?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

### When is this information needed?

- [x] Real-time (every batch/epoch)
- [ ] Periodic (every N episodes)
- [ ] On-demand (when investigating issues)
- [x] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

This is a **grouped record** containing multiple related fields in the `EpisodeStats` dataclass.

| Field | Type | Units | Range | Default |
|-------|------|-------|-------|---------|
| `total_episodes` | `int` | count | `[0, inf)` | `0` |
| `length_mean` | `float` | steps | `[0.0, max_epochs]` | `0.0` |
| `length_std` | `float` | steps | `[0.0, inf)` | `0.0` |
| `length_min` | `int` | steps | `[0, max_epochs]` | `0` |
| `length_max` | `int` | steps | `[0, max_epochs]` | `0` |
| `timeout_rate` | `float` | ratio | `[0.0, 1.0]` | `0.0` |
| `success_rate` | `float` | ratio | `[0.0, 1.0]` | `0.0` |
| `early_termination_rate` | `float` | ratio | `[0.0, 1.0]` | `0.0` |
| `steps_per_germinate` | `float` | steps/action | `[0.0, inf)` | `0.0` |
| `steps_per_prune` | `float` | steps/action | `[0.0, inf)` | `0.0` |
| `steps_per_fossilize` | `float` | steps/action | `[0.0, inf)` | `0.0` |
| `completion_trend` | `str` | enum | `"improving"`, `"stable"`, `"declining"` | `"stable"` |

### Semantic Meaning

**Episode Length Statistics:**
- `length_mean`: Average number of steps per episode. Low values may indicate early termination; high values near max_epochs suggest timeouts.
- `length_std`: Variance in episode length. High variance suggests inconsistent policy behavior.
- `length_min/max`: Range of episode lengths for quick anomaly detection.

**Outcome Tracking:**
- `timeout_rate`: Fraction of episodes hitting max_steps without terminal state. High values suggest insufficient policy progress.
- `success_rate`: Fraction of episodes achieving the goal state. Primary success metric.
- `early_termination_rate`: Fraction of episodes terminated early (failure/reset). May indicate training instability.

**Action Efficiency:**
- `steps_per_germinate`: Average steps between GERMINATE actions. Low values = aggressive seed spawning; high values = conservative.
- `steps_per_prune`: Average steps between PRUNE actions. Indicates lifecycle management frequency.
- `steps_per_fossilize`: Average steps between FOSSILIZE actions. Low values suggest fast seed maturation.

**Trend Detection:**
- `completion_trend`: Direction of episode completion rates over recent window. Derived from rolling success rate comparison.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `success_rate > 0.7` | Policy achieving goals reliably |
| **Warning** | `0.5 < success_rate <= 0.7` OR `timeout_rate > 0.1` | Moderate success, some timeouts |
| **Critical** | `success_rate < 0.3` OR `timeout_rate > 0.2` OR `early_termination_rate > 0.2` | Policy failing frequently |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | **NOT WIRED** - Data sources exist but are not aggregated |
| **Potential Sources** | `EPISODE_OUTCOME` events, `BATCH_EPOCH_COMPLETED` payload |
| **File** | `/home/john/esper-lite/src/esper/simic/training/vectorized.py` |
| **Function/Method** | Episode end handling in main training loop |
| **Line(s)** | ~3347-3364 (EPISODE_OUTCOME emission) |

```python
# EPISODE_OUTCOME is emitted but does not contain length/outcome data:
env_state.telemetry_cb(TelemetryEvent(
    event_type=TelemetryEventType.EPISODE_OUTCOME,
    epoch=episodes_completed + env_idx,
    data=EpisodeOutcomePayload(
        env_id=env_idx,
        episode_idx=episode_outcome.episode_idx,
        final_accuracy=episode_outcome.final_accuracy,
        param_ratio=episode_outcome.param_ratio,
        # ... no length, timeout, success fields
    ),
))
```

### Transport

| Stage | Mechanism | File | Status |
|-------|-----------|------|--------|
| **1. Emission** | EPISODE_OUTCOME events | `simic/training/vectorized.py` | Partial - missing length/outcome data |
| **2. Collection** | `_handle_episode_outcome()` | `karn/sanctum/aggregator.py` | Exists but doesn't populate EpisodeStats |
| **3. Aggregation** | **STUB** - `EpisodeStats(total_episodes=...)` | `karn/sanctum/aggregator.py:539` | Only total_episodes wired |
| **4. Delivery** | `snapshot.episode_stats` | `karn/sanctum/schema.py:1383` | Field exists, mostly defaults |

```
[vectorized.py] --EPISODE_OUTCOME--> [Aggregator] --STUB--> [EpisodeStats] --> [Widget]
                                          |
                                          v
                              (only total_episodes populated)
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `EpisodeStats` |
| **Field** | `episode_stats` |
| **Path from SanctumSnapshot** | `snapshot.episode_stats` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 227-256 (dataclass), 1383 (snapshot field) |

```python
@dataclass
class EpisodeStats:
    """Episode-level aggregate metrics."""
    # Episode length statistics
    length_mean: float = 0.0
    length_std: float = 0.0
    length_min: int = 0
    length_max: int = 0

    # Outcome tracking (over recent N episodes)
    total_episodes: int = 0
    timeout_count: int = 0
    success_count: int = 0
    early_termination_count: int = 0

    # Derived rates
    timeout_rate: float = 0.0
    success_rate: float = 0.0
    early_termination_rate: float = 0.0

    # Steps per action type
    steps_per_germinate: float = 0.0
    steps_per_prune: float = 0.0
    steps_per_fossilize: float = 0.0

    # Completion trend
    completion_trend: str = "stable"
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EpisodeMetricsPanel | `/home/john/esper-lite/src/esper/karn/sanctum/widgets/tamiyo_brain/episode_metrics_panel.py` | Full display of all fields in training mode |

**Widget Rendering (training mode):**
```
Length       mu:123  sigma:45   [12-500]
Outcomes     X10%  v75%  x15%
Steps/Act    germ:45  prune:120  foss:200
Trend        improving ^
```

---

## 5. Wiring Verification

### Checklist

- [ ] **Emitter exists** - EPISODE_OUTCOME emitted but lacks length/outcome fields
- [ ] **Transport works** - Handler exists but doesn't aggregate length/outcome
- [x] **Schema field exists** - `EpisodeStats` fully defined with all fields
- [x] **Default is correct** - All fields default to 0/0.0/"stable"
- [x] **Consumer reads it** - EpisodeMetricsPanel reads all fields
- [ ] **Display is correct** - Widget works but shows all zeros (stub data)
- [ ] **Thresholds applied** - Widget has rate styling but no data to style

### Wiring Gap Analysis

**What's Wired:**
1. `total_episodes` - Populated from `self._current_episode` in aggregator

**What's NOT Wired:**
1. `length_mean/std/min/max` - Episode length not tracked
2. `timeout_count/rate` - No timeout detection in telemetry
3. `success_count/rate` - No success detection in telemetry
4. `early_termination_count/rate` - No early termination tracking
5. `steps_per_germinate/prune/fossilize` - Action frequency not tracked
6. `completion_trend` - Trend detection not implemented

**Root Cause:**
The `EPISODE_OUTCOME` payload (`EpisodeOutcomePayload`) focuses on Pareto analysis metrics (accuracy, param_ratio, stability) rather than episode structure metrics (length, outcome type). The aggregator explicitly marks this as a stub:

```python
# aggregator.py:537-539
# Stub observation and episode stats (telemetry not yet wired)
observation_stats = ObservationStats()
episode_stats = EpisodeStats(total_episodes=self._current_episode)
```

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | — | No tests for episode length emission | `[ ]` |
| Unit (aggregator) | — | No tests for EpisodeStats population | `[ ]` |
| Integration (end-to-end) | — | No tests | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification | `[ ]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI
3. Observe EPISODE HEALTH panel
4. **Expected (current stub state):**
   - Length: mu:0 sigma:0 [0-0]
   - Outcomes: all 0%
   - Steps/Act: all 0
   - Trend: stable
5. Only `total_episodes` counter will update (shown in border title)

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Episode completion detection | event | Need to detect when episodes end and why |
| Step counting per episode | metric | Need to track steps from episode start to end |
| Action counting per type | metric | Need to count GERMINATE/PRUNE/FOSSILIZE actions |
| Success/failure classification | logic | Need to determine episode outcome type |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EpisodeMetricsPanel | widget | Displays all fields (currently shows defaults) |
| Training diagnostics | analysis | Would use rates for policy health assessment |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Initial telemetry record - documented as partially wired stub |

---

## 8. Notes

### Implementation Gap

The EpisodeStats dataclass and widget are fully implemented, but the data pipeline is not wired. The aggregator comment explicitly acknowledges this:

```python
# Stub observation and episode stats (telemetry not yet wired)
```

### Required Work to Complete Wiring

To fully wire this telemetry:

1. **Extend EPISODE_OUTCOME payload** or create new event with:
   - `episode_length: int` - Steps in the episode
   - `outcome_type: str` - "success", "timeout", "early_termination"

2. **Track action counts** during episode:
   - Count GERMINATE, PRUNE, FOSSILIZE actions per episode
   - Emit in episode end event or accumulate in aggregator

3. **Aggregator implementation** needed:
   - Maintain rolling window of episode lengths for mean/std/min/max
   - Track outcome counts and compute rates
   - Compute steps-per-action ratios
   - Implement trend detection from rolling success rate

### Design Consideration

The existing `EpisodeOutcomePayload` is designed for Pareto analysis (multi-objective optimization) rather than episode diagnostics. Consider whether to:
- Extend the existing payload (adds fields, may bloat Pareto-focused consumers)
- Create a new `EPISODE_DIAGNOSTICS` event type (cleaner separation of concerns)

### Related Records

- `TELE-010` current_episode - Total episode count (wired)
- `TELE-600-603` ObservationStats - Similar stub pattern (partially wired)
- `TELE-500-513` SeedLifecycleStats - Fully wired lifecycle metrics

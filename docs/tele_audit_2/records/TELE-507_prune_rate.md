# Telemetry Record: [TELE-507] Prune Rate

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-507` |
| **Name** | Prune Rate |
| **Category** | `seed` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "How frequently are seeds being pruned relative to episode progression? Is the system culling failed seeds at a healthy rate?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

### When is this information needed?

- [x] Real-time (every batch/epoch)
- [ ] Periodic (every N episodes)
- [ ] On-demand (when investigating issues)
- [ ] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `float` |
| **Units** | prunes per episode |
| **Range** | `[0.0, +inf)` - typically small values like 0.0-2.0 |
| **Precision** | 1 decimal place for display |
| **Default** | `0.0` |

### Semantic Meaning

> Prune rate measures how many seed modules are being culled per episode of training.
> Computed as:
>
> `prune_rate = cumulative_prune_count / current_episode`
>
> High prune rate indicates aggressive culling (possibly too strict thresholds).
> Low prune rate indicates minimal culling (possibly allowing underperforming seeds to persist).
> A healthy system balances pruning failed seeds without excessive churn.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `0.1 < prune_rate < 1.0` | Normal culling of failed seeds |
| **Warning** | `prune_rate >= 1.0` or `prune_rate == 0.0` for extended periods | Either too aggressive pruning or no culling at all |
| **Critical** | `prune_rate > 2.0` | Excessive churn - seeds being pruned faster than they can prove themselves |

**Note:** No explicit thresholds are coded in the UI. The prune rate display uses trend-based coloring (rising=green, falling=red, stable=dim) via `prune_trend`.

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Slot prune operation in Kasmina |
| **File** | `/home/john/esper-lite/src/esper/kasmina/slot.py` |
| **Function/Method** | `Slot.prune()` |
| **Line(s)** | ~1512-1594 |

```python
def prune(self, reason: str = "", *, initiator: str = "policy") -> bool:
    """Prune the current seed immediately (PRUNE_INSTANT)."""
    # ... validation and state updates ...
    self._emit_telemetry(
        TelemetryEventType.SEED_PRUNED,
        data=SeedPrunedPayload(
            slot_id=self.slot_id,
            env_id=-1,  # Sentinel - will be replaced by emit_with_env_context
            reason=reason,
            blueprint_id=blueprint_id,
            improvement=improvement,
            auto_pruned=is_auto_prune,
            epochs_total=epochs_total,
            counterfactual=counterfactual,
            initiator=initiator,
        )
    )
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `SEED_PRUNED` event with `SeedPrunedPayload` | `kasmina/slot.py` |
| **2. Collection** | Event captured by telemetry system | `leyline/telemetry.py` |
| **3. Aggregation** | `SanctumAggregator` increments `_cumulative_pruned` on SEED_PRUNED | `karn/sanctum/aggregator.py` |
| **4. Delivery** | `prune_rate` computed in `_build_snapshot()` as `_cumulative_pruned / current_ep` | `karn/sanctum/aggregator.py` |

```
[Slot.prune()] --SEED_PRUNED--> [Aggregator._cumulative_pruned++]
                                        |
                              [_build_snapshot()]
                                        |
                              [prune_rate = _cumulative_pruned / current_episode]
                                        |
                              [SeedLifecycleStats.prune_rate]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SeedLifecycleStats` |
| **Field** | `prune_rate` |
| **Path from SanctumSnapshot** | `snapshot.seed_lifecycle.prune_rate` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 187 |

```python
@dataclass
class SeedLifecycleStats:
    """Seed lifecycle aggregate metrics for TamiyoBrain display."""
    # ...
    prune_rate: float = 0.0  # Prunes per episode
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| SlotsPanel | `/home/john/esper-lite/src/esper/karn/sanctum/widgets/tamiyo_brain/slots_panel.py` | Displayed as `Prune{arrow}{rate:.1f}/ep` with trend indicator |

```python
# Line 122-123 in slots_panel.py
result.append(f"Prune{p_arrow}", style=p_style)
result.append(f"{lifecycle.prune_rate:.1f}/ep", style="dim")
```

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** - `Slot.prune()` emits `SEED_PRUNED` event
- [x] **Transport works** - Aggregator handles `SEED_PRUNED` and increments `_cumulative_pruned`
- [x] **Schema field exists** - `SeedLifecycleStats.prune_rate: float = 0.0`
- [x] **Default is correct** - 0.0 appropriate before any seeds are pruned
- [x] **Consumer reads it** - SlotsPanel accesses `snapshot.seed_lifecycle.prune_rate`
- [x] **Display is correct** - Value renders as `{rate:.1f}/ep` format
- [x] **Thresholds applied** - Trend-based coloring via `prune_trend` (rising=green, falling=red)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/kasmina/test_kasmina_telemetry.py` | Tests SEED_PRUNED emission | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_backend.py` | `test_seed_pruned_*` variants | `[x]` |
| Integration (end-to-end) | — | `prune_rate` computation not tested | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe SlotsPanel "Prune" row in the lifecycle section
4. Verify prune rate updates after seeds are pruned
5. Check trend arrow direction matches rate changes over time

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `SEED_PRUNED` events | event | Prune rate only increases when seeds are actually pruned |
| `current_episode` | counter | Denominator for rate calculation |
| `TELE-506` prune_count | telemetry | Numerator - cumulative count of pruned seeds |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `TELE-508` prune_trend | telemetry | Uses prune_rate history to compute trend |
| `TELE-504` blend_success_rate | telemetry | Conceptually related - blend success = fossilized / (fossilized + pruned) |
| SlotsPanel display | widget | Uses rate for lifecycle section display |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Telemetry Audit | Initial creation during TELE-507 audit |

---

## 8. Notes

> **Computation Details:** The prune rate is computed fresh in `_build_snapshot()` rather than being incrementally updated. This ensures consistency but means the rate is a cumulative average, not a moving window.
>
> **Rate History:** The aggregator maintains `_prune_rate_history` (deque of 20 values) for trend detection, updated at batch end in `handle_batch_summary()`.
>
> **Trend Coloring Logic:** The widget uses `prune_trend` (rising/stable/falling) rather than absolute thresholds. Rising prune rate shows green arrow (more aggressive culling), falling shows red arrow. This is contextual - rising prunes might be good (cleaning up failures) or bad (too strict thresholds) depending on training phase.
>
> **Related Metrics:** Works in concert with `germination_rate` and `fossilize_rate` to give a complete picture of seed lifecycle health. High germination + high prune + low fossilize indicates seeds are being spawned but not succeeding.

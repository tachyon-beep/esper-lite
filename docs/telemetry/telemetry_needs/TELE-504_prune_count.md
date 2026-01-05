# Telemetry Record: [TELE-504] Prune Count

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-504` |
| **Name** | Prune Count |
| **Category** | `seed` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "How many seeds have been pruned (removed due to poor performance) across the entire training run?"

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

| Property | Value |
|----------|-------|
| **Type** | `int` |
| **Units** | count (cumulative) |
| **Range** | `[0, inf)` non-negative integers |
| **Precision** | integer |
| **Default** | `0` |

### Semantic Meaning

> Prune count is the cumulative number of seeds that have been pruned (removed) during the training run. A seed is pruned when it fails to meet performance thresholds during training or blending stages. Pruning is the negative terminal state for seeds, contrasted with fossilization (successful permanent integration).
>
> High prune counts relative to fossilize counts indicate the policy is struggling to develop viable seed modules, which may signal:
> - Poor blueprint selection
> - Overly aggressive pruning thresholds
> - Fundamental task difficulty
> - Policy instability during blending

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `prune_count <= fossilize_count` | More seeds succeed than fail |
| **Warning** | `prune_count > fossilize_count` | More pruning than fossilization (displayed red) |
| **Critical** | N/A | No critical threshold defined |

**Threshold Source:** `slots_panel.py` line 99 - prune styled red when `prune_count > fossilize_count`

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Slot prune operation when seed fails performance thresholds |
| **File** | `/home/john/esper-lite/src/esper/kasmina/slot.py` |
| **Function/Method** | `Slot.prune()` |
| **Line(s)** | ~1512-1594 |

```python
def prune(self, reason: str = "", *, initiator: str = "policy") -> bool:
    """Prune the current seed immediately (PRUNE_INSTANT)."""
    # ... validation and state capture ...

    self._emit_telemetry(
        TelemetryEventType.SEED_PRUNED,
        data=SeedPrunedPayload(
            slot_id=self.slot_id,
            env_id=-1,  # Sentinel - replaced by emit_with_env_context
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
| **1. Emission** | `Slot._emit_telemetry(SEED_PRUNED)` | `kasmina/slot.py` |
| **2. Collection** | `TelemetryEvent` with `SeedPrunedPayload` | `leyline/telemetry.py` |
| **3. Aggregation** | `SanctumAggregator._cumulative_pruned += 1` | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Written to `SeedLifecycleStats.prune_count` | `karn/sanctum/schema.py` |

```
[Slot.prune()] --SEED_PRUNED--> [TelemetryEmitter] --event--> [SanctumAggregator] --> [SeedLifecycleStats.prune_count]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SeedLifecycleStats` |
| **Field** | `prune_count` |
| **Path from SanctumSnapshot** | `snapshot.seed_lifecycle.prune_count` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 178 |

```python
@dataclass
class SeedLifecycleStats:
    """Seed lifecycle aggregate metrics for TamiyoBrain display."""
    # Cumulative counts (entire run)
    germination_count: int = 0
    prune_count: int = 0  # <-- This field
    fossilize_count: int = 0
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| SlotsPanel | `/home/john/esper-lite/src/esper/karn/sanctum/widgets/tamiyo_brain/slots_panel.py` | Displayed in lifecycle section with conditional red styling |

```python
# slots_panel.py lines 98-100
result.append("  Prune:", style="dim")
prune_style = "red" if lifecycle.prune_count > lifecycle.fossilize_count else "dim"
result.append(f"{lifecycle.prune_count}", style=prune_style)
```

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** - `Slot.prune()` emits `SEED_PRUNED` event with `SeedPrunedPayload`
- [x] **Transport works** - Event flows through telemetry system to aggregator
- [x] **Schema field exists** - `SeedLifecycleStats.prune_count: int = 0`
- [x] **Default is correct** - 0 appropriate before any seeds are pruned
- [x] **Consumer reads it** - SlotsPanel accesses `snapshot.seed_lifecycle.prune_count`
- [x] **Display is correct** - Value renders as integer with conditional styling
- [x] **Thresholds applied** - Red styling when prune_count > fossilize_count

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/kasmina/test_kasmina_telemetry.py` | Various SEED_PRUNED tests | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_backend.py` | `test_seed_pruned_resets_slot` | `[x]` |
| Integration (prune_count) | — | No specific prune_count test | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification | `[ ]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 50`
2. Open Sanctum TUI (auto-opens)
3. Observe SlotsPanel lifecycle section
4. Verify "Prune:" count increments when seeds are pruned
5. Observe red styling when prune_count exceeds fossilize_count

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `SEED_PRUNED` event | event | Aggregator increments counter on this event |
| Seed lifecycle | system | Seeds must exist and be pruned for count to increment |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `prune_rate` | telemetry | Computed as `prune_count / current_episode` |
| `blend_success_rate` | telemetry | Computed as `fossilize_count / (fossilize_count + prune_count)` |
| `prune_trend` | telemetry | Trend direction based on rate history |
| SlotsPanel display | widget | Conditional styling based on prune vs fossilize ratio |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Claude | Initial creation during telemetry audit |

---

## 8. Notes

> **Aggregation Details:** The aggregator maintains `_cumulative_pruned` which is incremented in the `SEED_PRUNED` event handler (line 1145 in aggregator.py). This is then written to `SeedLifecycleStats.prune_count` during snapshot generation.
>
> **Related Metrics:** This metric is part of the seed lifecycle tracking system along with:
> - `germination_count` (TELE-502)
> - `fossilize_count` (TELE-503)
> - `prune_rate` (TELE-505)
>
> **Test Gap:** While there are unit tests for SEED_PRUNED event handling, there is no explicit test verifying that `prune_count` is correctly aggregated and exposed in the snapshot. The aggregation logic is covered implicitly through the `test_seed_pruned_resets_slot` test which verifies the pruned_count per-env counter.

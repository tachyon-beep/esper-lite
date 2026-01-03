# Telemetry Record: [TELE-513] Average Lifespan Epochs

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-513` |
| **Name** | Average Lifespan Epochs |
| **Category** | `seed` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "How long do seeds typically survive before reaching a terminal state (fossilized or pruned)?"

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
| **Type** | `float` |
| **Units** | epochs |
| **Range** | `[0.0, inf)` - non-negative |
| **Precision** | 0 decimal places for display (integer epochs) |
| **Default** | `0.0` |

### Semantic Meaning

> Average lifespan represents the mean number of epochs a seed survives from germination until it reaches a terminal state (either FOSSILIZED or PRUNED). Computed as:
>
> avg_lifespan = sum(epochs_total for terminated seeds) / count(terminated seeds)
>
> This metric captures seed "vitality" - short lifespans may indicate overly aggressive pruning or difficult-to-blend seeds, while long lifespans suggest seeds are contributing value over extended periods.
>
> The lifespan is tracked per-seed via `SeedMetrics.epochs_total`, which increments on each `record_accuracy()` call during the seed lifecycle.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `value >= 10` | Seeds living long enough to provide meaningful contribution |
| **Warning** | `5 <= value < 10` | Seeds may be dying prematurely |
| **Critical** | `value < 5` | Seeds dying before they can contribute (investigate pruning policy) |

**Note:** Thresholds are heuristic. Optimal lifespan depends on task complexity and training dynamics.

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Seed terminal state transitions (FOSSILIZED or PRUNED) |
| **File** | `/home/john/esper-lite/src/esper/kasmina/slot.py` |
| **Function/Method** | `Slot._try_advance_stage()` (fossilize) and `Slot.prune()` (prune) |
| **Line(s)** | ~1448-1502 (fossilize), ~1534-1594 (prune) |

```python
# From Slot._try_advance_stage() for FOSSILIZED:
epochs_total = metrics.epochs_total
...
self._emit_telemetry(
    TelemetryEventType.SEED_FOSSILIZED,
    data=SeedFossilizedPayload(
        ...
        epochs_total=epochs_total,
    )
)

# From Slot.prune() for PRUNED:
epochs_total = self.state.metrics.epochs_total
...
self._emit_telemetry(
    TelemetryEventType.SEED_PRUNED,
    data=SeedPrunedPayload(
        ...
        epochs_total=epochs_total,
    )
)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `Slot._emit_telemetry()` emits SEED_FOSSILIZED/SEED_PRUNED events | `kasmina/slot.py` |
| **2. Collection** | Event with `SeedFossilizedPayload` or `SeedPrunedPayload` | `leyline/telemetry.py` |
| **3. Aggregation** | `SanctumAggregator._handle_seed_fossilized()` / `_handle_seed_pruned()` appends to `_seed_lifespan_history` | `karn/sanctum/aggregator.py` |
| **4. Delivery** | `build_snapshot()` computes mean and writes to `snapshot.seed_lifecycle.avg_lifespan_epochs` | `karn/sanctum/aggregator.py` |

```
[Slot.prune()/fossilize] --emit_telemetry()--> [TelemetryEvent]
    --event--> [SanctumAggregator._handle_seed_*()]
    --append epochs_total--> [_seed_lifespan_history: deque[int]]
    --build_snapshot()--> [mean()]--> [SeedLifecycleStats.avg_lifespan_epochs]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SeedLifecycleStats` |
| **Field** | `avg_lifespan_epochs` |
| **Path from SanctumSnapshot** | `snapshot.seed_lifecycle.avg_lifespan_epochs` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 192 |

### Aggregation Logic

Located in `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py`:

```python
# Line 222: History buffer initialization
_seed_lifespan_history: deque[int] = field(default_factory=lambda: deque(maxlen=100))

# Lines 1082-1083: Append on fossilize
if fossilized_payload.epochs_total > 0:
    self._seed_lifespan_history.append(fossilized_payload.epochs_total)

# Lines 1121-1122: Append on prune
if pruned_payload.epochs_total > 0:
    self._seed_lifespan_history.append(pruned_payload.epochs_total)

# Lines 514-518: Compute average in build_snapshot()
avg_lifespan = (
    sum(self._seed_lifespan_history) / len(self._seed_lifespan_history)
    if self._seed_lifespan_history
    else 0.0
)
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| SlotsPanel | `/home/john/esper-lite/src/esper/karn/sanctum/widgets/tamiyo_brain/slots_panel.py` | Line 133: Displayed as "Lifespan: u{X} eps" in cyan |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** - `Slot.prune()` and `Slot._try_advance_stage()` emit events with `epochs_total`
- [x] **Transport works** - `SeedFossilizedPayload` and `SeedPrunedPayload` carry `epochs_total` field
- [x] **Schema field exists** - `SeedLifecycleStats.avg_lifespan_epochs: float = 0.0`
- [x] **Default is correct** - 0.0 appropriate before any seed terminates
- [x] **Consumer reads it** - `SlotsPanel` accesses `lifecycle.avg_lifespan_epochs`
- [x] **Display is correct** - Value renders as integer epochs with "u" prefix (mu symbol)
- [ ] **Thresholds applied** - No color coding based on lifespan value (always cyan)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/kasmina/test_slot.py` | (epochs_total incremented) | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_backend.py` | `test_seed_fossilized_increments_counters` | `[x]` |
| Integration (end-to-end) | `tests/integration/test_transformer_integration.py` | `test_transformer_with_seed_lifecycle` | `[x]` |
| Visual (TUI snapshot) | - | Manual verification | `[ ]` |

**Note:** No dedicated tests for `avg_lifespan_epochs` computation or `_seed_lifespan_history` buffer.

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 50`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe SlotsPanel "Lifespan" row
4. Wait for seeds to reach terminal states (FOSSILIZED or PRUNED)
5. Verify lifespan value updates as seeds terminate
6. Check that value reflects average (should stabilize after multiple terminations)

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `SeedMetrics.epochs_total` | computation | Incremented in `record_accuracy()` each epoch |
| SEED_FOSSILIZED event | event | Contributes lifespan data on successful blend |
| SEED_PRUNED event | event | Contributes lifespan data on prune |
| `_seed_lifespan_history` buffer | aggregator state | Rolling window of last 100 terminated seed lifespans |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| SlotsPanel display | widget | Shows lifecycle quality metric |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Initial creation during telemetry audit |

---

## 8. Notes

> **Design Decision:** The lifespan history uses a rolling window (`deque(maxlen=100)`) rather than cumulative statistics. This means the average reflects recent training behavior, not the entire run history. This is intentional for monitoring training dynamics as they evolve.
>
> **Edge Cases:**
> - Seeds that terminate with `epochs_total == 0` are excluded from the history (guard at lines 1082, 1121)
> - Before any seed terminates, `avg_lifespan_epochs` is 0.0
> - The average is recomputed on every `build_snapshot()` call from the history buffer
>
> **Display Format:** The SlotsPanel shows this as "u{X} eps" where "u" is meant to represent the mu (mean) symbol. The value is truncated to integer display (`:0f` format).
>
> **Gap Identified:** No health threshold coloring is applied - the value is always displayed in cyan regardless of whether the lifespan is healthy or concerning.

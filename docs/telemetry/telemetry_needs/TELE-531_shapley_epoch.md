# Telemetry Record: [TELE-531] Shapley Epoch

> **Status:** `[x] Planned` `[ ] In Progress` `[ ] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-531` |
| **Name** | Shapley Epoch |
| **Category** | `attribution` |
| **Priority** | `P2-moderate` |

## 2. Purpose

### What question does this answer?

> "When were the current Shapley values computed? How stale is this attribution data?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [ ] Researcher (analysis)
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
| **Type** | `int` |
| **Units** | epoch number |
| **Range** | `[0, max_epochs]` |
| **Precision** | Integer (exact) |
| **Default** | `0` (before first computation) |

### Semantic Meaning

> The epoch when Shapley values were last computed. Used to:
> 1. Display staleness context in ShapleyPanel (e.g., "[Epoch 42]")
> 2. Detect if attribution data is outdated relative to current training state
> 3. Correlate Shapley snapshots with other epoch-indexed telemetry
>
> **Staleness interpretation:**
> - `current_epoch - shapley_epoch <= 5`: Fresh data
> - `current_epoch - shapley_epoch > 20`: Stale - slot contributions may have changed
> - `shapley_epoch == 0`: No Shapley computation has occurred yet

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Fresh** | `current_epoch - shapley_epoch <= 5` | Attribution reflects recent state |
| **Aging** | `5 < current_epoch - shapley_epoch <= 20` | Consider recomputation |
| **Stale** | `current_epoch - shapley_epoch > 20` | Attribution may be outdated |

**Threshold Source:** Application-specific; these are suggested defaults.

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Shapley computation captures current epoch at computation time |
| **File** | `NOT YET IMPLEMENTED` |
| **Function/Method** | TBD - set when Shapley computation completes |
| **Line(s)** | N/A |

```python
# PLANNED IMPLEMENTATION (not yet wired)
# When Shapley values are computed:
shapley_snapshot = ShapleySnapshot(
    slot_ids=active_slot_ids,
    values=computed_values,
    epoch=current_epoch,  # Capture epoch at computation time
    timestamp=datetime.now(),
)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Included in ShapleySnapshot payload | TBD |
| **2. Collection** | ANALYTICS_SNAPSHOT event | `karn/sanctum/aggregator.py` |
| **3. Aggregation** | Extracted from event payload | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Written to `env_state.shapley_snapshot.epoch` | `karn/sanctum/schema.py` |

```
[Shapley Computation]
  --shapley_computed(epoch=N)-->
  [SanctumAggregator]
  ---->
  [EnvState.shapley_snapshot.epoch = N]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `ShapleySnapshot` |
| **Field** | `epoch` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].shapley_snapshot.epoch` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 279 |
| **Default Value** | `0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| ShapleyPanel | `widgets/shapley_panel.py` (lines 65-66) | Displays "[Epoch N]" or "Latest" in header |

---

## 5. Wiring Verification

### Checklist

- [ ] **Emitter exists** - NO emitter implemented yet (part of Shapley computation)
- [ ] **Transport works** - Aggregator handler TBD
- [x] **Schema field exists** - `ShapleySnapshot.epoch: int = 0` at line 279
- [x] **Default is correct** - `0` indicates no computation yet
- [x] **Consumer reads it** - ShapleyPanel displays epoch in header
- [x] **Display is correct** - Shows "[Epoch N]" or "Latest" (line 65)
- [ ] **Thresholds applied** - No staleness warning implemented yet

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | N/A | Emitter not implemented | `[ ]` |
| Unit (aggregator) | N/A | Aggregator handler not implemented | `[ ]` |
| Widget (ShapleyPanel) | `tests/karn/sanctum/widgets/test_shapley_panel.py` | Epoch display | `[ ]` |

### Manual Verification Steps

1. Start training with multiple active seeds
2. Wait for Shapley computation at episode boundary
3. Observe ShapleyPanel header - should show "[Epoch N]" where N is computation epoch
4. Continue training past computation epoch
5. Verify displayed epoch remains at computation time (not current epoch)

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Shapley computation | system | Epoch set when values are computed |
| Current epoch tracking | state | Must capture epoch at computation time |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| ShapleyPanel header | widget | Shows when attribution was computed |
| Staleness detection | future | Could warn if attribution is outdated |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation (planned, not wired) |

---

## 8. Notes

> **Wiring Status:** Schema field exists, ShapleyPanel reads it, but no emitter populates it yet. This field is populated as part of the Shapley computation feature (TELE-530).
>
> **Design Rationale:** Epoch is captured at computation time (not updated each epoch) to reflect when the attribution was actually computed. This allows operators to assess data freshness.
>
> **Future Enhancement:** Consider adding staleness indicator (yellow/red if `current_epoch - shapley_epoch > threshold`) to ShapleyPanel.

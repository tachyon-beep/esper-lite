# Telemetry Record: [TELE-500] Total Slots

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-500` |
| **Name** | Total Slots |
| **Category** | `seed` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "How many total slot positions exist across all environments for seed modules?"

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
| **Units** | count (slots) |
| **Range** | `[0, ∞)` — typically `n_envs * n_slots_per_env` |
| **Precision** | integer |
| **Default** | `0` |

### Semantic Meaning

> Total slots represents the capacity for seed modules across the training environment. Computed as:
>
> `total_slots = len(envs) * len(slot_ids)`
>
> This is static for a training run after TRAINING_STARTED event is received, as both `num_envs` and `slot_ids` are locked at that point. The value represents the theoretical maximum number of concurrent seed modules that could be active.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `value > 0` | Training has configured slots |
| **Warning** | N/A | N/A |
| **Critical** | `value == 0` | No slots configured; seed system inactive |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Computed in aggregator from env/slot configuration |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `SanctumAggregator.snapshot()` |
| **Line(s)** | 504 |

```python
# Computed from env and slot configuration
total_slots = len(self._envs) * len(self._slot_ids)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | TRAINING_STARTED event populates `num_envs` and `slot_ids` | `aggregator.py` L656-662 |
| **2. Collection** | Aggregator stores in `_envs` dict and `_slot_ids` list | `aggregator.py` L228, L252 |
| **3. Aggregation** | Computed on each `snapshot()` call | `aggregator.py` L504 |
| **4. Delivery** | Written to `SanctumSnapshot.total_slots` | `aggregator.py` L588 |

```
[TRAINING_STARTED] --> [_envs, _slot_ids stored] --> [snapshot()] --> [SanctumSnapshot.total_slots]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SanctumSnapshot` |
| **Field** | `total_slots` |
| **Path from SanctumSnapshot** | `snapshot.total_slots` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1367 |

**Secondary Location (also stored in SeedLifecycleStats):**

| Property | Value |
|----------|-------|
| **Dataclass** | `SeedLifecycleStats` |
| **Field** | `total_slots` |
| **Path from SanctumSnapshot** | `snapshot.seed_lifecycle.total_slots` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 183 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| SlotsPanel | `/home/john/esper-lite/src/esper/karn/sanctum/widgets/tamiyo_brain/slots_panel.py` | Border title: "CURRENT SLOTS - {total_slots} across {n_envs} envs" (L39) |
| SlotsPanel | `/home/john/esper-lite/src/esper/karn/sanctum/widgets/tamiyo_brain/slots_panel.py` | Used as denominator for stage bar width calculation (L49, L71) |
| SlotsPanel | `/home/john/esper-lite/src/esper/karn/sanctum/widgets/tamiyo_brain/slots_panel.py` | Lifecycle display: "Active: {active_count}/{total_slots}" (L95) |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — Computed in `snapshot()` from env/slot configuration
- [x] **Transport works** — Value computed fresh on each snapshot
- [x] **Schema field exists** — `SanctumSnapshot.total_slots: int = 0` (L1367)
- [x] **Default is correct** — 0 appropriate before TRAINING_STARTED
- [x] **Consumer reads it** — SlotsPanel accesses `snapshot.total_slots` and `snapshot.seed_lifecycle.total_slots`
- [x] **Display is correct** — Value renders in border title and lifecycle section
- [x] **Thresholds applied** — "[no environments]" shown when total == 0 (L51-52)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | — | — | `[ ]` |
| Unit (aggregator) | — | — | `[ ]` |
| Integration (end-to-end) | — | — | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe SlotsPanel border title shows "CURRENT SLOTS - N across M envs"
4. Verify total_slots = N is consistent with n_envs * n_slots_per_env
5. Confirm lifecycle section shows "Active: X/N" where N matches border title

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| TRAINING_STARTED event | event | Provides `n_envs` and `slot_ids` to populate the computation |
| `num_envs` configuration | config | Number of parallel environments |
| `slot_ids` configuration | config | List of slot identifiers per environment |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `TELE-501` active_slots | telemetry | Computed as `total_slots - DORMANT_count` |
| SlotsPanel stage bars | display | Used as denominator for proportional bar width |
| Lifecycle active ratio | display | Shows `active_count/total_slots` |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Initial creation during telemetry audit |

---

## 8. Notes

> **Design Decision:** `total_slots` is computed dynamically in `snapshot()` rather than cached because it's cheap to compute and avoids staleness issues if env/slot configuration were ever to change mid-run.
>
> **Dual Storage:** The value appears in both `SanctumSnapshot.total_slots` (L1367) and `SeedLifecycleStats.total_slots` (L183). Both are populated from the same computation in `snapshot()` (L504, L526). The SeedLifecycleStats copy is used for the lifecycle section display.
>
> **Static After Startup:** While computed dynamically, the value is effectively static after TRAINING_STARTED because:
> - `_slot_ids_locked` is set to True (L657)
> - `num_envs` is set from the event payload (L631)
> - No mechanism exists to add/remove envs or slots mid-training
>
> **Test Gap:** No unit or integration tests verify this computation. Since it's a simple multiplication of list lengths, the risk is low, but test coverage would improve confidence.

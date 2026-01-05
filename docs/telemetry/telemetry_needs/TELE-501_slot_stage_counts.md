# Telemetry Record: [TELE-501] Slot Stage Counts

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-501` |
| **Name** | Slot Stage Counts |
| **Category** | `seed` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "What is the distribution of slots across lifecycle stages (DORMANT, GERMINATED, TRAINING, BLENDING, HOLDING, FOSSILIZED) across all environments?"

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
| **Type** | `dict[str, int]` |
| **Units** | count (number of slots) |
| **Range** | `[0, total_slots]` per key |
| **Precision** | integer |
| **Default** | `{}` (empty dict) |

### Semantic Meaning

> Aggregated count of slots in each lifecycle stage across all environments. The dict keys are the six primary lifecycle stages:
>
> - **DORMANT**: Inactive slots waiting for germination
> - **GERMINATED**: Newly spawned modules beginning development
> - **TRAINING**: Modules actively learning from host errors
> - **BLENDING**: Modules being integrated into the host
> - **HOLDING**: Modules waiting after training before blend decision
> - **FOSSILIZED**: Modules permanently fused with host
>
> Transition states (PRUNED, EMBARGOED, RESETTING) are mapped to DORMANT for aggregation simplicity.
>
> Total slots = `n_envs * n_slots_per_env`. Active slots = `total_slots - DORMANT count`.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | DORMANT < 80% of total | Slots are being utilized |
| **Warning** | DORMANT > 90% of total | Very few seeds germinating |
| **Critical** | All slots DORMANT for extended period | No seed activity |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Aggregated from per-environment seed states in snapshot generation |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `SanctumAggregator.snapshot()` |
| **Line(s)** | ~477-505 |

```python
# Aggregate slot states across all environments for TamiyoBrain slot summary
slot_stage_counts: dict[str, int] = {
    "DORMANT": 0,
    "GERMINATED": 0,
    "TRAINING": 0,
    "BLENDING": 0,
    "HOLDING": 0,
    "FOSSILIZED": 0,
}

for env in self._envs.values():
    for slot_id in self._slot_ids:
        seed = env.seeds.get(slot_id)
        if seed is None:
            slot_stage_counts["DORMANT"] += 1
        else:
            stage = seed.stage
            if stage in slot_stage_counts:
                slot_stage_counts[stage] += 1
            else:
                # Handle transition states (PRUNED, EMBARGOED, RESETTING) as DORMANT
                slot_stage_counts["DORMANT"] += 1
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Seed lifecycle events (SEED_GERMINATED, SEED_STAGE_CHANGED, SEED_FOSSILIZED, SEED_PRUNED) | `src/esper/kasmina/slots/*.py` |
| **2. Collection** | Event handlers update `env.seeds` dict | `aggregator.py` lines 990-1120 |
| **3. Aggregation** | `snapshot()` method iterates all envs | `aggregator.py` lines 477-505 |
| **4. Delivery** | Written to `SanctumSnapshot.slot_stage_counts` | `aggregator.py` line 587 |

```
[Seed Slots] --lifecycle events--> [Aggregator Event Handlers] --> [env.seeds dict]
                                                                        |
[snapshot() call] --> [Iterate all envs.seeds] --> [slot_stage_counts] --> [SanctumSnapshot]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SanctumSnapshot` |
| **Field** | `slot_stage_counts` |
| **Path from SanctumSnapshot** | `snapshot.slot_stage_counts` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | ~1366 |

```python
# Aggregate slot state across all environments (for TamiyoBrain slot summary)
# Keys: DORMANT, GERMINATED, TRAINING, BLENDING, HOLDING, FOSSILIZED
slot_stage_counts: dict[str, int] = field(default_factory=dict)
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| SlotsPanel | `/home/john/esper-lite/src/esper/karn/sanctum/widgets/tamiyo_brain/slots_panel.py` | Stage distribution with proportional bars |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** -- Aggregator computes from per-env seed states
- [x] **Transport works** -- Lifecycle events update env.seeds, snapshot aggregates
- [x] **Schema field exists** -- `SanctumSnapshot.slot_stage_counts: dict[str, int]`
- [x] **Default is correct** -- Empty dict appropriate before any seeds exist
- [x] **Consumer reads it** -- SlotsPanel accesses `snapshot.slot_stage_counts`
- [x] **Display is correct** -- Renders stage distribution with colored bars
- [x] **Thresholds applied** -- Color coding uses `STAGE_COLORS` from leyline

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | -- | -- | `[ ]` |
| Unit (aggregator) | -- | -- | `[ ]` |
| Integration (end-to-end) | -- | -- | `[ ]` |
| Visual (TUI snapshot) | -- | Manual verification | `[ ]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe SlotsPanel (CURRENT SLOTS section)
4. Verify stage counts update as seeds germinate, train, and fossilize
5. Verify DORMANT count decreases when seeds are created
6. Verify FOSSILIZED count increases when seeds complete lifecycle

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Seed lifecycle events | event | SEED_GERMINATED, SEED_STAGE_CHANGED, SEED_FOSSILIZED, SEED_PRUNED |
| `EnvState.seeds` dict | state | Per-environment seed tracking |
| `SeedState.stage` field | state | Individual seed stage string |
| `STAGE_COLORS` | constant | Color mapping from leyline |
| `STAGE_ABBREVIATIONS` | constant | Abbreviation mapping from leyline |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `SanctumSnapshot.total_slots` | derived | Computed alongside slot_stage_counts |
| `SanctumSnapshot.active_slots` | derived | `total_slots - DORMANT count` |
| `SanctumSnapshot.avg_epochs_in_stage` | derived | Mean epochs for non-dormant slots |
| SlotsPanel widget | display | Primary consumer for visualization |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Telemetry Audit | Initial record creation |

---

## 8. Notes

> **Data Flow Pattern:** Unlike most telemetry that flows from training loop -> emitter -> aggregator, this metric is **computed in the aggregator** during snapshot generation by iterating over the already-tracked `env.seeds` state. The seed states themselves are populated by lifecycle event handlers (`_handle_seed_germinated`, `_handle_seed_stage_changed`, etc.).
>
> **Transition State Handling:** Transition states (PRUNED, EMBARGOED, RESETTING) are collapsed into DORMANT for display simplicity. This means the DORMANT count includes both truly empty slots and slots in transitional states.
>
> **Missing Tests:** No unit or integration tests currently cover this metric. The wiring is verified by code inspection, not automated tests.
>
> **Display Integration:** The SlotsPanel uses `STAGE_COLORS` and `STAGE_ABBREVIATIONS` from leyline for consistent visual language across all Sanctum widgets. Stages with zero count are dimmed, active stages are colored.

# Telemetry Record: [TELE-502] Active Seed Count

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-502` |
| **Name** | Active Seed Count |
| **Category** | `seed` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "How many seeds are currently active (non-terminal) across all environments?"

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
| **Units** | count (number of seeds) |
| **Range** | `[0, total_slots]` where total_slots = num_envs * slots_per_env |
| **Precision** | integer |
| **Default** | `0` |

### Semantic Meaning

> Number of seeds currently in non-terminal states. A seed is "active" if it is NOT in a terminal or dormant state. Computed as:
>
> active_count = total_slots - dormant_slots
>
> Where dormant_slots includes truly dormant slots plus transition states (PRUNED, EMBARGOED, RESETTING) which are treated as dormant for counting purposes. Active seeds include those in GERMINATED, TRAINING, BLENDING, HOLDING, or FOSSILIZED states.
>
> **Note:** This is a cross-environment aggregate. For per-environment active counts, see individual EnvRecord.seeds filtering.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `active_count > 0` | Seeds are being germinated and developing |
| **Warning** | `active_count == 0 && training_started` | No active seeds - system may be stalled |
| **Critical** | N/A | No critical threshold defined |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Aggregated from per-environment seed state |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `SanctumAggregator.snapshot()` |
| **Line(s)** | ~476-505 (slot stage counting), ~521-535 (SeedLifecycleStats construction) |

```python
# Slot stage counting (lines ~477-505)
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

total_slots = len(self._envs) * len(self._slot_ids)
active_slots = total_slots - slot_stage_counts["DORMANT"]

# SeedLifecycleStats construction (line ~525)
seed_lifecycle = SeedLifecycleStats(
    ...
    active_count=active_slots,
    ...
)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Seed state updates via telemetry events | `simic/telemetry/emitters.py` |
| **2. Collection** | Aggregator maintains `_envs` dict with EnvRecord containing seeds | `karn/sanctum/aggregator.py` |
| **3. Aggregation** | Computed in `snapshot()` from all env seed states | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Written to `SanctumSnapshot.seed_lifecycle.active_count` | `karn/sanctum/schema.py` |

```
[Per-env seed states] --> [Aggregator._envs] --> [snapshot() slot counting] --> [SeedLifecycleStats.active_count]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SeedLifecycleStats` |
| **Field** | `active_count` |
| **Path from SanctumSnapshot** | `snapshot.seed_lifecycle.active_count` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | ~182 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| SlotsPanel | `widgets/tamiyo_brain/slots_panel.py` | Displays as "Active: X/Y" format showing active_count/total_slots |
| HistoricalEnvDetail | `widgets/historical_env_detail.py` | Computes its own local active_count from record.seeds (does NOT use snapshot.seed_lifecycle) |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — Aggregator computes active_slots from slot stage counts
- [x] **Transport works** — Value computed from _envs state during snapshot()
- [x] **Schema field exists** — `SeedLifecycleStats.active_count: int = 0`
- [x] **Default is correct** — 0 appropriate before any seeds germinate
- [x] **Consumer reads it** — SlotsPanel accesses `snapshot.seed_lifecycle.active_count`
- [x] **Display is correct** — Value renders as "Active: X/Y" with cyan styling
- [ ] **Thresholds applied** — No threshold-based coloring for active_count itself

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/kasmina/test_multi_seed.py` | `test_fossilize_decrements_active_count` | `[x]` |
| Unit (aggregator) | N/A | No aggregator test for seed_lifecycle | `[ ]` |
| Integration (end-to-end) | N/A | No integration test | `[ ]` |
| Visual (TUI snapshot) | N/A | Manual verification | `[ ]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe SlotsPanel in TamiyoBrain section
4. Verify "Active: X/Y" updates as seeds germinate
5. Verify count decreases when seeds are pruned or fossilized

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Seed stage tracking | data | Requires accurate stage values in EnvRecord.seeds |
| `_slot_ids` initialization | config | Must know slot IDs to count total slots |
| `_envs` population | events | Requires env state events to populate seed data |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `TELE-503` total_slots | telemetry | Displayed together as "X/Y" |
| SlotsPanel display | widget | Uses for lifecycle overview section |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Telemetry Audit | Initial creation from audit |

---

## 8. Notes

> **Design Note:** The aggregator treats transition states (PRUNED, EMBARGOED, RESETTING) as DORMANT for counting purposes. This means `active_count = total_slots - dormant_count` where dormant includes these transition states.
>
> **Inconsistency Found:** `HistoricalEnvDetail` widget computes its own local `active_count` by filtering `record.seeds` for non-terminal stages, rather than using `snapshot.seed_lifecycle.active_count`. This is a different calculation (per-env vs aggregate) but could cause confusion. The local calculation excludes DORMANT, FOSSILIZED, and PRUNED stages.
>
> **Test Gap:** No aggregator-level tests verify that `seed_lifecycle.active_count` is correctly populated. The existing test in `test_multi_seed.py` tests the domain model's `count_active_seeds()` method, not the telemetry aggregation path.
>
> **FOSSILIZED Counting:** Note that FOSSILIZED is counted as "active" in the aggregator (it's in the slot_stage_counts dict and not treated as DORMANT), but in `HistoricalEnvDetail` it's excluded from local active count. This is a semantic discrepancy.

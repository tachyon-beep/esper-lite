# Telemetry Record: [TELE-505] Germination Count

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-505` |
| **Name** | Germination Count |
| **Category** | `seed` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "How many seeds have germinated during this training run?"

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
| **Range** | `[0, infinity)` - non-negative integers |
| **Precision** | integer |
| **Default** | `0` |

### Semantic Meaning

> Cumulative count of seeds that have germinated during the training run. A seed germinates when it transitions from DORMANT to GERMINATED stage, indicating a new neural module has been spawned and attached to the host network.
>
> This metric tracks the total number of germination events since training started. It increases monotonically and is used alongside prune_count and fossilize_count to assess the overall seed lifecycle health.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `germination_count >= 1` after plateau | System is responding to training stagnation |
| **Warning** | `germination_count == 0` after many epochs | No seeds germinated despite training |
| **Critical** | N/A | No critical threshold (informational metric) |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Slot germination in Kasmina module |
| **File** | `/home/john/esper-lite/src/esper/kasmina/slot.py` |
| **Function/Method** | `Slot.germinate()` (emits SEED_GERMINATED event) |
| **Line(s)** | ~1365-1381 |

```python
# Slot.germinate() emits SEED_GERMINATED event
self._emit_telemetry(
    TelemetryEventType.SEED_GERMINATED,
    data=SeedGerminatedPayload(
        slot_id=self.slot_id,
        env_id=-1,  # Sentinel - replaced by emit_with_env_context
        blueprint_id=blueprint_id,
        params=sum(p.numel() for p in self.seed.parameters() if p.requires_grad),
        alpha=self.state.alpha if self.state else 0.0,
        blend_tempo_epochs=blend_tempo_epochs,
        alpha_curve=self.state.alpha_controller.alpha_curve.name,
        grad_ratio=0.0,
        has_vanishing=False,
        has_exploding=False,
        epochs_in_stage=0,
    )
)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `_emit_telemetry()` with SEED_GERMINATED | `kasmina/slot.py` |
| **2. Collection** | TelemetryEvent with SeedGerminatedPayload | `leyline/telemetry.py` |
| **3. Aggregation** | `_handle_seed_event()` increments `_cumulative_germinated` | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Written to `snapshot.seed_lifecycle.germination_count` | `karn/sanctum/schema.py` |

```
[Slot.germinate()] --SEED_GERMINATED--> [TelemetryEmitter] --event-->
  [Aggregator._handle_seed_event()] --increments--> [_cumulative_germinated] --snapshot-->
  [SeedLifecycleStats.germination_count]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SeedLifecycleStats` |
| **Field** | `germination_count` |
| **Path from SanctumSnapshot** | `snapshot.seed_lifecycle.germination_count` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 177 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| SlotsPanel | `/home/john/esper-lite/src/esper/karn/sanctum/widgets/tamiyo_brain/slots_panel.py` | Displayed as "Germ: {count}" with green styling (line 102) |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** - `Slot.germinate()` emits SEED_GERMINATED event
- [x] **Transport works** - Event includes SeedGerminatedPayload
- [x] **Schema field exists** - `SeedLifecycleStats.germination_count: int = 0`
- [x] **Default is correct** - 0 appropriate before any germinations
- [x] **Consumer reads it** - SlotsPanel accesses `snapshot.seed_lifecycle.germination_count`
- [x] **Display is correct** - Value renders with green styling
- [ ] **Thresholds applied** - No threshold coloring (informational metric)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/tamiyo/test_heuristic_unit.py` | `test_germination_count_increments` | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_backend.py` | `test_seed_germinated_adds_seed` | `[x]` |
| Integration (seed_lifecycle) | — | — | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification | `[ ]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 50`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe SlotsPanel "Germ:" counter
4. Wait for plateau detection to trigger germination
5. Verify germination_count increments when SEED_GERMINATED event appears in EventLog

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| SEED_GERMINATED event | event | Triggered by Slot.germinate() |
| Tamiyo decision to germinate | decision | Heuristic policy decides when to germinate |
| Plateau detection | computation | Germination typically triggered by training plateau |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `germination_rate` | telemetry | Computed as `germination_count / current_episode` |
| `germination_trend` | telemetry | Trend indicator derived from rate history |
| `blend_success_rate` | telemetry | Uses fossilized/(fossilized+pruned) but context includes germination |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Claude | Initial telemetry record creation |

---

## 8. Notes

> **Aggregator tracking:** The aggregator maintains `_cumulative_germinated` as an instance variable, incrementing it each time a SEED_GERMINATED event is processed (line 1016 in aggregator.py). This value is then copied to `SeedLifecycleStats.germination_count` when creating the snapshot (line 522).
>
> **Tamiyo internal counter:** Note that `tamiyo/heuristic.py` also tracks its own `_germination_count` (line 127, 199) for internal decision-making. This is a separate counter used by the heuristic policy, not the telemetry counter in the aggregator.
>
> **Test gap:** While unit tests exist for both the Tamiyo heuristic counter and the aggregator's seed handling, there is no explicit integration test that verifies `snapshot.seed_lifecycle.germination_count` is populated correctly after SEED_GERMINATED events. The aggregator tests verify the seed is added to env.seeds but don't explicitly check the cumulative count field on SeedLifecycleStats.

# Telemetry Record: [TELE-506] Germination Rate

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-506` |
| **Name** | Germination Rate |
| **Category** | `seed` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "How frequently are seeds germinating relative to episode count? Is the system spawning new modules at a healthy rate?"

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
| **Units** | germinations per episode |
| **Range** | `[0.0, +inf)` — typically 0.0 to ~10.0 in practice |
| **Precision** | 1 decimal place for display |
| **Default** | `0.0` |

### Semantic Meaning

> Germination rate measures how many seeds are spawned per episode on average across the entire training run. Computed as:
>
> germination_rate = cumulative_germinated / current_episode
>
> Higher values indicate active module spawning; very low values may indicate the policy isn't selecting GERMINATE actions or all slots are already occupied. This is a cumulative average (not windowed), providing a stable long-term metric.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `rate > 0.0` | Seeds are being spawned |
| **Warning** | N/A | No explicit warning threshold defined |
| **Critical** | N/A | No explicit critical threshold defined |

**Note:** Germination rate does not have explicit health thresholds. The value is context-dependent on the number of slots and training configuration. The `germination_trend` indicator provides directional guidance.

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Slot germination in Kasmina |
| **File** | `/home/john/esper-lite/src/esper/kasmina/slot.py` |
| **Function/Method** | `Slot.germinate()` |
| **Line(s)** | ~1365-1381 |

```python
# Germination event emission (slot.py:1365-1381)
self._emit_telemetry(
    TelemetryEventType.SEED_GERMINATED,
    data=SeedGerminatedPayload(
        slot_id=self.slot_id,
        env_id=-1,  # Sentinel - will be replaced by emit_with_env_context
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
| **1. Emission** | `TelemetryEventType.SEED_GERMINATED` event with `SeedGerminatedPayload` | `kasmina/slot.py` |
| **2. Collection** | Event collected by telemetry infrastructure | `leyline/telemetry.py` |
| **3. Aggregation** | `_cumulative_germinated` counter incremented on `SEED_GERMINATED` event | `karn/sanctum/aggregator.py:1016` |
| **4. Delivery** | Rate computed as `_cumulative_germinated / current_episode` in `get_snapshot()` | `karn/sanctum/aggregator.py:527` |

```
[Slot.germinate()] --SEED_GERMINATED--> [TelemetryEmitter] --event-->
[SanctumAggregator._cumulative_germinated++] --> [get_snapshot() computes rate] -->
[SeedLifecycleStats.germination_rate]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SeedLifecycleStats` |
| **Field** | `germination_rate` |
| **Path from SanctumSnapshot** | `snapshot.seed_lifecycle.germination_rate` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 186 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| SlotsPanel | `/home/john/esper-lite/src/esper/karn/sanctum/widgets/tamiyo_brain/slots_panel.py` | Displayed as "Germ{trend_arrow}{rate:.1f}/ep" on line 119-120 |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `Slot.germinate()` emits `SEED_GERMINATED` event
- [x] **Transport works** — Aggregator receives event and increments `_cumulative_germinated`
- [x] **Schema field exists** — `SeedLifecycleStats.germination_rate: float = 0.0`
- [x] **Default is correct** — 0.0 appropriate before any germinations
- [x] **Consumer reads it** — SlotsPanel accesses `snapshot.seed_lifecycle.germination_rate`
- [x] **Display is correct** — Value renders as "{rate:.1f}/ep" with trend arrow
- [ ] **Thresholds applied** — No health thresholds; trend arrow provides directional color

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/kasmina/test_kasmina_telemetry.py` | Tests emit SEED_GERMINATED | `[x]` |
| Unit (aggregator) | — | No specific germination_rate test | `[ ]` |
| Integration (end-to-end) | `tests/integration/test_transformer_integration.py` | `test_transformer_with_seed_lifecycle` | `[x]` |
| Visual (TUI snapshot) | — | Manual verification | `[ ]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe SlotsPanel "CURRENT SLOTS" section
4. Verify "Germ{arrow}{rate}/ep" updates after germinations occur
5. Verify trend arrow changes (green ↗ for rising, dim → for stable, red ↘ for falling) as training progresses

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `SEED_GERMINATED` events | event | Each germination increments the cumulative counter |
| `current_episode` | context | Denominator for rate calculation |
| `TELE-505` germination_count | telemetry | Numerator for rate calculation (`_cumulative_germinated`) |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `TELE-509` germination_trend | telemetry | Trend computed from `_germination_rate_history` deque |
| SlotsPanel display | widget | Renders rate with trend indicator |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Initial telemetry record creation |

---

## 8. Notes

> **Design Decision:** Germination rate is a cumulative average (total germinations / total episodes) rather than a windowed or rolling average. This provides stability but may be slow to reflect recent changes. The `germination_trend` indicator (`_germination_rate_history` deque with maxlen=20) compensates by detecting directional changes.
>
> **Rate History Update:** The rate history is updated in `_handle_batch_ended()` (aggregator.py:1207-1212) after each batch, ensuring trend detection reflects recent batches.
>
> **Display Format:** Rendered as `{rate:.1f}/ep` with 1 decimal place. The trend arrow (↗, →, ↘) provides at-a-glance directional guidance.
>
> **Related Metrics:** Part of the `SeedLifecycleStats` family alongside `prune_rate` (TELE-507) and `fossilize_rate` (TELE-508), all sharing the same computation pattern.

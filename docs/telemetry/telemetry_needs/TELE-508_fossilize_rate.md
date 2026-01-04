# Telemetry Record: [TELE-508] Fossilize Rate

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-508` |
| **Name** | Fossilize Rate |
| **Category** | `seed` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "How frequently are seeds being permanently integrated (fossilized) into the host network?"

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
| **Units** | fossilizations per episode |
| **Range** | `[0.0, +inf)` — typically `[0.0, 1.0]` for healthy training |
| **Precision** | 2 decimal places for display |
| **Default** | `0.0` |

### Semantic Meaning

> Fossilize rate measures the frequency of permanent seed integration into the host network, computed as:
>
> `fossilize_rate = cumulative_fossilized / current_episode`
>
> - **High rate (>0.5/ep):** Seeds are integrating quickly, possibly too aggressively
> - **Low rate (<0.1/ep):** Few seeds reaching fossilization, may indicate pruning dominance
> - **Zero rate:** No fossilizations yet (early training) or policy not triggering fossilize actions
>
> This is a cumulative rate, not a windowed average. The `fossilize_trend` field (TELE-511) tracks rate direction.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | Context-dependent | Rate depends on task complexity and slot configuration |
| **Warning** | `rate = 0.0` after many episodes | No seeds integrating (possible policy issue) |
| **Critical** | N/A | No critical threshold defined for this metric |

**Note:** Unlike entropy or clip fraction, fossilize rate does not have hardcoded thresholds in `TUIThresholds`. The "healthy" range depends heavily on the training configuration (number of slots, episode length, task difficulty).

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Slot state transition to FOSSILIZED stage |
| **File** | `/home/john/esper-lite/src/esper/kasmina/slot.py` |
| **Function/Method** | `Slot._apply_gate_result()` (stage transition logic) |
| **Line(s)** | ~1487-1502 |

```python
# When transitioning to FOSSILIZED stage:
if target_stage == SeedStage.FOSSILIZED:
    self._emit_telemetry(
        TelemetryEventType.SEED_FOSSILIZED,
        data=SeedFossilizedPayload(
            slot_id=self.slot_id,
            env_id=-1,  # Sentinel - replaced by emit_with_env_context
            blueprint_id=blueprint_id,
            improvement=improvement,
            params_added=sum(...),
            alpha=self.state.alpha,
            epochs_total=epochs_total,
            counterfactual=counterfactual,
        )
    )
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `SEED_FOSSILIZED` event with `SeedFossilizedPayload` | `kasmina/slot.py` |
| **2. Collection** | Event dispatched via telemetry bus | `leyline/telemetry.py` |
| **3. Aggregation** | `SanctumAggregator` increments `_cumulative_fossilized` | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Computed as `_cumulative_fossilized / current_episode` in `get_snapshot()` | `karn/sanctum/aggregator.py` |

```
[Slot._apply_gate_result()] --SEED_FOSSILIZED--> [TelemetryBus] --> [SanctumAggregator]
                                                                         |
                                                           _cumulative_fossilized++
                                                                         |
                                                    get_snapshot() --> fossilize_rate = count/ep
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SeedLifecycleStats` |
| **Field** | `fossilize_rate` |
| **Path from SanctumSnapshot** | `snapshot.seed_lifecycle.fossilize_rate` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 188 |

```python
@dataclass
class SeedLifecycleStats:
    ...
    fossilize_rate: float = 0.0  # Fossilizations per episode
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| SlotsPanel | `/home/john/esper-lite/src/esper/karn/sanctum/widgets/tamiyo_brain/slots_panel.py` | Displayed as "Foss{trend_arrow} {rate:.2f}/ep" with trend indicator |

```python
# Line 125-126 in slots_panel.py
result.append(f"Foss{f_arrow}", style=f_style)
result.append(f"{lifecycle.fossilize_rate:.2f}/ep", style="dim")
```

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `Slot._apply_gate_result()` emits `SEED_FOSSILIZED` on stage transition
- [x] **Transport works** — Event reaches aggregator via telemetry bus
- [x] **Schema field exists** — `SeedLifecycleStats.fossilize_rate: float = 0.0`
- [x] **Default is correct** — 0.0 appropriate before any fossilizations
- [x] **Consumer reads it** — SlotsPanel accesses `snapshot.seed_lifecycle.fossilize_rate`
- [x] **Display is correct** — Rendered as "{rate:.2f}/ep" with dim styling
- [ ] **Thresholds applied** — No color coding based on rate value (trend arrow only)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/leyline/test_lifecycle_fix.py` | `test_fossilization_emits_telemetry` | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_backend.py` | `test_seed_fossilized_updates_counts` | `[x]` |
| Integration (rate computation) | — | — | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification | `[ ]` |

**Note:** Existing tests verify the `SEED_FOSSILIZED` event emission and counter increment, but do not explicitly test the `fossilize_rate` computation (`_cumulative_fossilized / current_episode`).

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe SlotsPanel lifecycle statistics line
4. Verify "Foss" rate updates after fossilization events occur
5. Confirm rate = fossilize_count / current_episode

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `SEED_FOSSILIZED` events | event | Counter increments on each fossilization event |
| `current_episode` | counter | Used as denominator for rate computation |
| Episode lifecycle | event | Rate only meaningful after episodes complete |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `TELE-511` fossilize_trend | telemetry | Trend computed from rate history |
| `TELE-506` blend_success_rate | telemetry | Uses fossilize_count in numerator |
| SlotsPanel display | display | Shows rate with trend indicator |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Telemetry Audit | Initial creation during TELE-508 audit |

---

## 8. Notes

> **Design Decision:** The rate is computed as a cumulative average (total fossilizations / total episodes) rather than a windowed average. This provides a stable long-term view but may hide recent rate changes. The `fossilize_trend` field (using `_fossilize_rate_history` deque) provides insight into recent direction.
>
> **Related Metrics:**
> - `TELE-505` `fossilize_count`: Raw cumulative count
> - `TELE-511` `fossilize_trend`: Direction indicator ("rising", "stable", "falling")
> - `TELE-506` `blend_success_rate`: fossilized / (fossilized + pruned)
>
> **Display Context:** In SlotsPanel, fossilize_rate is shown alongside germination_rate and prune_rate on the lifecycle statistics line, providing a complete view of seed lifecycle throughput.

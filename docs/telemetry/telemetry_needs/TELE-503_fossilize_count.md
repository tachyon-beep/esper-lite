# Telemetry Record: [TELE-503] Fossilize Count

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-503` |
| **Name** | Fossilize Count |
| **Category** | `seed` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "How many seeds have successfully fossilized (permanently integrated with the host) during this training run?"

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
| **Units** | count (seeds) |
| **Range** | `[0, +inf)` (non-negative) |
| **Precision** | integer |
| **Default** | `0` |

### Semantic Meaning

> Cumulative count of seeds that have transitioned to `FOSSILIZED` stage during the training run.
> Fossilization represents successful permanent integration of a seed module with the host network.
> This is a terminal state indicating the seed passed all quality gates and was blended into the host.
>
> In the botanical lifecycle metaphor:
> - Seeds germinate, train, blend, and then either fossilize (success) or get pruned (failure)
> - A fossilized seed's parameters become part of the host permanently
> - Higher fossilize_count indicates more successful module integrations

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `fossilize_count >= prune_count` | More seeds succeeding than failing |
| **Warning** | `prune_count > fossilize_count` (displayed in "red") | More seeds being pruned than fossilized |
| **Critical** | N/A | No critical threshold defined |

**Threshold Source:** `SlotsPanel` applies conditional styling: `prune_style = "red" if lifecycle.prune_count > lifecycle.fossilize_count else "dim"`

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Seed slot stage transition to FOSSILIZED |
| **File** | `/home/john/esper-lite/src/esper/kasmina/slot.py` |
| **Function/Method** | `SeedSlot._transition_seed()` (via `transition()`) |
| **Line(s)** | ~1487-1502 |

```python
# Fossilization event emission
if target_stage == SeedStage.FOSSILIZED:
    self._emit_telemetry(
        TelemetryEventType.SEED_FOSSILIZED,
        data=SeedFossilizedPayload(
            slot_id=self.slot_id,
            env_id=-1,  # Sentinel - will be replaced by emit_with_env_context
            blueprint_id=blueprint_id,
            improvement=improvement,
            params_added=sum(
                p.numel() for p in (self.seed.parameters() if self.seed is not None else []) if p.requires_grad
            ),
            alpha=self.state.alpha,
            epochs_total=epochs_total,
            counterfactual=counterfactual,
        )
    )
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `TelemetryEventType.SEED_FOSSILIZED` event | `kasmina/slot.py` |
| **2. Collection** | Event with `SeedFossilizedPayload` | `leyline/telemetry.py` |
| **3. Aggregation** | `SanctumAggregator._cumulative_fossilized += 1` | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Written to `snapshot.seed_lifecycle.fossilize_count` | `karn/sanctum/schema.py` |

```
[SeedSlot.transition()] --SEED_FOSSILIZED--> [TelemetryEmitter] --event--> [SanctumAggregator] --> [SeedLifecycleStats.fossilize_count]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SeedLifecycleStats` |
| **Field** | `fossilize_count` |
| **Path from SanctumSnapshot** | `snapshot.seed_lifecycle.fossilize_count` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 179 |

```python
@dataclass
class SeedLifecycleStats:
    """Seed lifecycle aggregate metrics for TamiyoBrain display."""
    # Cumulative counts (entire run)
    germination_count: int = 0
    prune_count: int = 0
    fossilize_count: int = 0  # <-- Line 179
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| SlotsPanel | `/home/john/esper-lite/src/esper/karn/sanctum/widgets/tamiyo_brain/slots_panel.py` | Displays as "Foss: {count}" in blue, also used for prune comparison styling |

```python
# Line 96-99 in slots_panel.py
result.append("  Foss:", style="dim")
result.append(f"{lifecycle.fossilize_count}", style="blue")
result.append("  Prune:", style="dim")
prune_style = "red" if lifecycle.prune_count > lifecycle.fossilize_count else "dim"
```

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `SeedSlot.transition()` emits `SEED_FOSSILIZED` when transitioning to FOSSILIZED stage
- [x] **Transport works** — Event includes `SeedFossilizedPayload` with all required fields
- [x] **Schema field exists** — `SeedLifecycleStats.fossilize_count: int = 0`
- [x] **Default is correct** — 0 appropriate before any seeds fossilize
- [x] **Consumer reads it** — SlotsPanel accesses `snapshot.seed_lifecycle.fossilize_count`
- [x] **Display is correct** — Value renders with blue styling, count format
- [x] **Thresholds applied** — Comparison with prune_count affects prune styling

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | N/A | No direct fossilize_count test found | `[ ]` |
| Unit (aggregator) | N/A | No SeedLifecycleStats test found | `[ ]` |
| Integration (end-to-end) | `tests/integration/test_transformer_integration.py` | `test_transformer_with_seed_lifecycle` | `[?]` |
| Visual (TUI snapshot) | N/A | Manual verification | `[ ]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 100`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe SlotsPanel "Foss:" count
4. Wait for seeds to complete their lifecycle and fossilize
5. Verify count increments when seeds transition to FOSSILIZED stage
6. Verify prune count styling turns red if prune_count > fossilize_count

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `SEED_FOSSILIZED` event | event | Emitted when seed transitions to FOSSILIZED stage |
| Seed lifecycle | computation | Seed must complete germination, training, blending first |
| `TELE-500` total_slots | telemetry | Provides context for slot capacity |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `SeedLifecycleStats.blend_success_rate` | computation | `fossilize / (fossilize + prune)` |
| `SeedLifecycleStats.fossilize_rate` | computation | `fossilize_count / current_episode` |
| `SeedLifecycleStats.fossilize_trend` | computation | Rate history trend detection |
| SlotsPanel prune styling | display | Comparison determines if prune count is styled red |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Claude | Initial creation during telemetry audit |

---

## 8. Notes

> **Design Decision:** Fossilize count is cumulative for the entire run (never resets). This allows tracking total module integration success over training.
>
> **Widget Display:** The SlotsPanel displays fossilize_count in blue (positive outcome) and uses it as a baseline to determine if prune_count should be styled in red (negative indicator when prunes exceed fossilizations).
>
> **Related Metrics:** The aggregator also tracks:
> - `fossilize_rate` (fossilizations per episode)
> - `fossilize_trend` (rising/stable/falling)
> - `blend_success_rate` (fossilized / (fossilized + pruned))
>
> **Test Gap:** No direct unit tests for `fossilize_count` aggregation were found. The aggregator tests exist but do not specifically test the SeedLifecycleStats population from SEED_FOSSILIZED events.

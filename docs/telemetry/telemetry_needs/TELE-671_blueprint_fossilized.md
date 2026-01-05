# Telemetry Record: [TELE-671] Blueprint Fossilized

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-671` |
| **Name** | Blueprint Fossilized |
| **Category** | `seed` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "How many seeds of each blueprint type have successfully fossilized (permanently integrated with the host) in this environment?"

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
| **Type** | `dict[str, int]` |
| **Units** | count (fossilized seeds per blueprint) |
| **Range** | `{blueprint_id: [0, +inf)}` (non-negative counts) |
| **Precision** | integer |
| **Default** | `{}` (empty dict) |

### Semantic Meaning

> Per-blueprint-type fossilization counts tracking successful module integrations.
> Keys are blueprint IDs like "conv_light", "dense_heavy", etc.
> Values are the cumulative fossilization counts for that blueprint type within the current episode.
>
> In the botanical lifecycle metaphor:
> - Fossilization represents permanent integration of a seed module with the host network
> - A fossilized seed's parameters become part of the host permanently
> - Higher fossilize counts for a blueprint indicate that architecture works well for this task
>
> This field is part of the Seed Graveyard, which provides insights into which blueprint
> architectures succeed vs fail. Combined with prune counts, it calculates success rate:
> `success_rate = fossilized / (fossilized + pruned)`

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy (Green)** | `success_rate >= 0.50` | 50%+ of seeds succeed |
| **Warning (Yellow)** | `success_rate >= 0.25` | 25-50% of seeds succeed |
| **Critical (Red)** | `success_rate < 0.25` | Less than 25% of seeds succeed |

**Threshold Source:** `DisplayThresholds.BLUEPRINT_SUCCESS_GREEN = 0.50`, `DisplayThresholds.BLUEPRINT_SUCCESS_YELLOW = 0.25` (in `/home/john/esper-lite/src/esper/karn/constants.py` lines 191-192)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Seed fossilization (SEED_FOSSILIZED event) |
| **File** | `/home/john/esper-lite/src/esper/kasmina/slot.py` |
| **Function/Method** | `SeedSlot._transition_seed()` (via `transition()`) |
| **Event Type** | `TelemetryEventType.SEED_FOSSILIZED` |

```python
# Fossilization event emission triggers blueprint_fossilized increment
if target_stage == SeedStage.FOSSILIZED:
    self._emit_telemetry(
        TelemetryEventType.SEED_FOSSILIZED,
        data=SeedFossilizedPayload(
            slot_id=self.slot_id,
            env_id=-1,  # Sentinel - replaced by emit_with_env_context
            blueprint_id=blueprint_id,
            improvement=improvement,
            params_added=...,
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
| **3. Aggregation** | `env.blueprint_fossilized[blueprint_id] += 1` | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Written to `env_state.blueprint_fossilized` | `karn/sanctum/schema.py` |

```
[SeedSlot.transition()] --SEED_FOSSILIZED--> [TelemetryEmitter] --event--> [SanctumAggregator] --> [EnvState.blueprint_fossilized]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `EnvState` |
| **Field** | `blueprint_fossilized` |
| **Path from SanctumSnapshot** | `snapshot.environments[env_id].blueprint_fossilized` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 468 |

```python
@dataclass
class EnvState:
    """Environment state tracking."""
    # Seed graveyard: per-blueprint lifecycle tracking
    blueprint_fossilized: dict[str, int] = field(default_factory=dict)  # blueprint -> fossilized count
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvDetailScreen | `/home/john/esper-lite/src/esper/karn/sanctum/widgets/env_detail_screen.py` | Displays as fossilized count (green) in graveyard table "foss" column; used to calculate success rate |

```python
# Line 697 in env_detail_screen.py
fossilized = env.blueprint_fossilized.get(blueprint, 0)
...
line.append(f"    {fossilized:2d}", style="green")

# Lines 707-716: Success rate calculation
terminated = fossilized + pruned
if terminated > 0:
    success_rate = fossilized / terminated
    if success_rate >= DisplayThresholds.BLUEPRINT_SUCCESS_GREEN:
        rate_style = "green"
    elif success_rate >= DisplayThresholds.BLUEPRINT_SUCCESS_YELLOW:
        rate_style = "yellow"
    else:
        rate_style = "red"
```

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `SeedSlot.transition()` emits `SEED_FOSSILIZED` when transitioning to FOSSILIZED stage
- [x] **Transport works** — Event includes `SeedFossilizedPayload` with `blueprint_id` field
- [x] **Schema field exists** — `EnvState.blueprint_fossilized: dict[str, int] = field(default_factory=dict)`
- [x] **Default is correct** — Empty dict appropriate before any seeds fossilize
- [x] **Consumer reads it** — EnvDetailScreen accesses `env.blueprint_fossilized` in `_render_graveyard()`
- [x] **Display is correct** — Value renders with green styling in foss column
- [x] **Thresholds applied** — Success rate (fossilized/terminated) determines rate column color

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | N/A | No direct blueprint_fossilized test found | `[ ]` |
| Unit (aggregator) | N/A | Covered by aggregator tests | `[?]` |
| Integration (end-to-end) | `tests/karn/sanctum/test_env_detail_screen.py` | Graveyard rendering tests | `[x]` |
| Visual (TUI snapshot) | N/A | Manual verification | `[ ]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 100`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Navigate to EnvDetailScreen for an active environment
4. Observe Seed Graveyard panel "foss" column
5. Verify fossilized counts increment when seeds transition to FOSSILIZED
6. Verify success rate column updates and color-codes appropriately

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `SEED_FOSSILIZED` event | event | Emitted when seed transitions to FOSSILIZED stage |
| `SeedFossilizedPayload.blueprint_id` | field | Identifies which blueprint type fossilized |
| Seed lifecycle completion | computation | Seed must complete germination, training, blending first |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| Graveyard success rate | computation | `fossilized / (fossilized + pruned)` |
| Rate column styling | display | Green/yellow/red based on success_rate thresholds |
| Blueprint architecture analysis | analysis | Identifies which architectures work well |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Claude | Initial creation during telemetry audit |

---

## 8. Notes

> **Design Decision:** blueprint_fossilized is per-episode (resets on episode boundary) for EnvState, but cumulative tracking is also maintained in the aggregator via `_cumulative_blueprint_fossilized`.
>
> **Widget Display:** The EnvDetailScreen displays fossilized counts in green in the "foss" column of the Seed Graveyard table. The column header is "foss" (line 674). Green styling represents successful outcomes.
>
> **Success Rate Calculation:** The success rate is calculated as `fossilized / (fossilized + pruned)` for each blueprint. This rate determines the color of the "rate" column:
> - Green: >= 50% success (BLUEPRINT_SUCCESS_GREEN)
> - Yellow: >= 25% success (BLUEPRINT_SUCCESS_YELLOW)
> - Red: < 25% success
>
> **Semantic Meaning:** A low fossilize:prune ratio for a blueprint type indicates that architecture doesn't work well for this task. High fossilization rates suggest the blueprint architecture is well-suited.
>
> **Related Fields:** Works in conjunction with:
> - `blueprint_spawns` (TELE-670) - total seeds spawned per blueprint
> - `blueprint_prunes` (TELE-672) - failed/removed seeds per blueprint

# Telemetry Record: [TELE-672] Blueprint Prunes

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-672` |
| **Name** | Blueprint Prunes |
| **Category** | `seed` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "How many seeds of each blueprint type have been pruned (failed/removed) in this environment?"

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
| **Units** | count (pruned seeds per blueprint) |
| **Range** | `{blueprint_id: [0, +inf)}` (non-negative counts) |
| **Precision** | integer |
| **Default** | `{}` (empty dict) |

### Semantic Meaning

> Per-blueprint-type prune counts tracking failed/removed seed modules.
> Keys are blueprint IDs like "conv_light", "dense_heavy", etc.
> Values are the cumulative prune counts for that blueprint type within the current episode.
>
> In the botanical lifecycle metaphor:
> - Pruning removes seed modules that failed to improve the host network
> - A pruned seed's parameters are discarded and never integrated
> - Higher prune counts for a blueprint indicate that architecture doesn't work well for this task
>
> This field is part of the Seed Graveyard, which provides insights into which blueprint
> architectures succeed vs fail. Combined with fossilize counts, it calculates success rate:
> `success_rate = fossilized / (fossilized + pruned)`

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy (Green)** | `success_rate >= 0.50` | 50%+ of seeds succeed (low prune rate) |
| **Warning (Yellow)** | `success_rate >= 0.25` | 25-50% of seeds succeed |
| **Critical (Red)** | `success_rate < 0.25` | Less than 25% of seeds succeed (high prune rate) |

**Threshold Source:** `DisplayThresholds.BLUEPRINT_SUCCESS_GREEN = 0.50`, `DisplayThresholds.BLUEPRINT_SUCCESS_YELLOW = 0.25` (in `/home/john/esper-lite/src/esper/karn/constants.py` lines 191-192)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Seed pruning (SEED_PRUNED event) |
| **File** | `/home/john/esper-lite/src/esper/kasmina/slot.py` |
| **Function/Method** | `SeedSlot._transition_seed()` (via `transition()`) |
| **Event Type** | `TelemetryEventType.SEED_PRUNED` |

```python
# Pruning event emission triggers blueprint_prunes increment
if target_stage == SeedStage.PRUNED:
    self._emit_telemetry(
        TelemetryEventType.SEED_PRUNED,
        data=SeedPrunedPayload(
            slot_id=self.slot_id,
            env_id=-1,  # Sentinel - replaced by emit_with_env_context
            blueprint_id=blueprint_id,
            reason=reason,
            epochs_trained=epochs_trained,
            alpha_at_prune=self.state.alpha,
        )
    )
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `TelemetryEventType.SEED_PRUNED` event | `kasmina/slot.py` |
| **2. Collection** | Event with `SeedPrunedPayload` | `leyline/telemetry.py` |
| **3. Aggregation** | `env.blueprint_prunes[blueprint_id] += 1` | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Written to `env_state.blueprint_prunes` | `karn/sanctum/schema.py` |

```
[SeedSlot.transition()] --SEED_PRUNED--> [TelemetryEmitter] --event--> [SanctumAggregator] --> [EnvState.blueprint_prunes]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `EnvState` |
| **Field** | `blueprint_prunes` |
| **Path from SanctumSnapshot** | `snapshot.environments[env_id].blueprint_prunes` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 467 |

```python
@dataclass
class EnvState:
    """Environment state tracking."""
    # Seed graveyard: per-blueprint lifecycle tracking
    blueprint_prunes: dict[str, int] = field(default_factory=dict)   # blueprint -> prune count
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvDetailScreen | `/home/john/esper-lite/src/esper/karn/sanctum/widgets/env_detail_screen.py` | Displays as prune count (red) in graveyard table "prun" column; used to calculate success rate |

```python
# Line 698 in env_detail_screen.py
pruned = env.blueprint_prunes.get(blueprint, 0)
...
line.append(f"    {pruned:2d}", style="red")

# Lines 707-716: Success rate calculation (prune affects denominator)
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

- [x] **Emitter exists** — `SeedSlot.transition()` emits `SEED_PRUNED` when transitioning to PRUNED stage
- [x] **Transport works** — Event includes `SeedPrunedPayload` with `blueprint_id` field
- [x] **Schema field exists** — `EnvState.blueprint_prunes: dict[str, int] = field(default_factory=dict)`
- [x] **Default is correct** — Empty dict appropriate before any seeds are pruned
- [x] **Consumer reads it** — EnvDetailScreen accesses `env.blueprint_prunes` in `_render_graveyard()`
- [x] **Display is correct** — Value renders with red styling in prun column
- [x] **Thresholds applied** — Success rate (fossilized/terminated) determines rate column color

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | N/A | No direct blueprint_prunes test found | `[ ]` |
| Unit (aggregator) | N/A | Covered by aggregator tests | `[?]` |
| Integration (end-to-end) | `tests/karn/sanctum/test_env_detail_screen.py` | Graveyard rendering tests | `[x]` |
| Visual (TUI snapshot) | N/A | Manual verification | `[ ]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 100`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Navigate to EnvDetailScreen for an active environment
4. Observe Seed Graveyard panel "prun" column
5. Verify prune counts increment when seeds are pruned
6. Verify success rate column updates and color-codes appropriately (red when prune rate high)

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `SEED_PRUNED` event | event | Emitted when seed transitions to PRUNED stage |
| `SeedPrunedPayload.blueprint_id` | field | Identifies which blueprint type was pruned |
| Seed lifecycle termination | computation | Seed pruned due to poor performance or resource constraints |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| Graveyard success rate | computation | `fossilized / (fossilized + pruned)` |
| Rate column styling | display | Green/yellow/red based on success_rate thresholds |
| Blueprint architecture analysis | analysis | Identifies which architectures don't work well |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Claude | Initial creation during telemetry audit |

---

## 8. Notes

> **Design Decision:** blueprint_prunes is per-episode (resets on episode boundary) for EnvState, but cumulative tracking is also maintained in the aggregator via `_cumulative_blueprint_prunes`.
>
> **Widget Display:** The EnvDetailScreen displays prune counts in red in the "prun" column of the Seed Graveyard table. The column header is "prun" (line 675). Red styling represents failed outcomes.
>
> **Success Rate Calculation:** The success rate is calculated as `fossilized / (fossilized + pruned)` for each blueprint. High prune counts relative to fossilized counts result in low success rates (red styling).
>
> **Semantic Meaning:** A high prune rate for a blueprint type indicates that architecture doesn't work well for this task. The graveyard helps operators identify which blueprint architectures to avoid or investigate.
>
> **Prune Reasons:** Seeds may be pruned for various reasons including:
> - Poor improvement metrics
> - Training stagnation
> - Resource constraints
> - Quality gate failures
>
> **Related Fields:** Works in conjunction with:
> - `blueprint_spawns` (TELE-670) - total seeds spawned per blueprint
> - `blueprint_fossilized` (TELE-671) - successful integrations per blueprint

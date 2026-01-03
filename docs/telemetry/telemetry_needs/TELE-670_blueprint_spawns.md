# Telemetry Record: [TELE-670] Blueprint Spawns

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-670` |
| **Name** | Blueprint Spawns |
| **Category** | `seed` |
| **Priority** | `P2-useful` |

## 2. Purpose

### What question does this answer?

> "How many seeds of each blueprint type have germinated in this environment?"

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
| **Units** | count (seeds per blueprint) |
| **Range** | `{blueprint_id: [0, +inf)}` (non-negative counts) |
| **Precision** | integer |
| **Default** | `{}` (empty dict) |

### Semantic Meaning

> Per-blueprint-type spawn counts tracking how many seeds of each blueprint have germinated.
> Keys are blueprint IDs like "conv_light", "dense_heavy", etc.
> Values are the cumulative spawn counts for that blueprint type within the current episode.
>
> In the botanical lifecycle metaphor:
> - Each blueprint represents a specific neural module architecture
> - Spawning is the germination of a seed module from that blueprint
> - Tracking spawns by blueprint enables analysis of which architectures are being tried
>
> This field is part of the Seed Graveyard, which provides insights into which blueprint
> architectures succeed vs fail. The spawn count establishes the denominator for lifecycle analysis.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **N/A** | No threshold | Spawn count is informational, not health-indicative |

**Threshold Source:** N/A - spawn counts do not have health thresholds; they provide context for fossilize/prune ratios.

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Seed germination (SEED_GERMINATED event) |
| **File** | `/home/john/esper-lite/src/esper/kasmina/slot.py` |
| **Function/Method** | `SeedSlot._transition_seed()` (via `transition()`) |
| **Event Type** | `TelemetryEventType.SEED_GERMINATED` |

```python
# Germination event emission triggers blueprint_spawns increment
if target_stage == SeedStage.GERMINATED:
    self._emit_telemetry(
        TelemetryEventType.SEED_GERMINATED,
        data=SeedGerminatedPayload(
            slot_id=self.slot_id,
            env_id=-1,  # Sentinel - replaced by emit_with_env_context
            blueprint_id=blueprint_id,
            ...
        )
    )
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `TelemetryEventType.SEED_GERMINATED` event | `kasmina/slot.py` |
| **2. Collection** | Event with `SeedGerminatedPayload` | `leyline/telemetry.py` |
| **3. Aggregation** | `env.blueprint_spawns[blueprint_id] += 1` | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Written to `env_state.blueprint_spawns` | `karn/sanctum/schema.py` |

```
[SeedSlot.transition()] --SEED_GERMINATED--> [TelemetryEmitter] --event--> [SanctumAggregator] --> [EnvState.blueprint_spawns]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `EnvState` |
| **Field** | `blueprint_spawns` |
| **Path from SanctumSnapshot** | `snapshot.environments[env_id].blueprint_spawns` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 466 |

```python
@dataclass
class EnvState:
    """Environment state tracking."""
    # Seed graveyard: per-blueprint lifecycle tracking
    blueprint_spawns: dict[str, int] = field(default_factory=dict)  # blueprint -> spawn count
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvDetailScreen | `/home/john/esper-lite/src/esper/karn/sanctum/widgets/env_detail_screen.py` | Displays as spawn count (cyan) in graveyard table "spawn" column |

```python
# Line 696 in env_detail_screen.py
spawned = env.blueprint_spawns.get(blueprint, 0)
...
line.append(f"    {spawned:2d}", style="cyan")
```

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `SeedSlot.transition()` emits `SEED_GERMINATED` when transitioning to GERMINATED stage
- [x] **Transport works** — Event includes `SeedGerminatedPayload` with `blueprint_id` field
- [x] **Schema field exists** — `EnvState.blueprint_spawns: dict[str, int] = field(default_factory=dict)`
- [x] **Default is correct** — Empty dict appropriate before any seeds spawn
- [x] **Consumer reads it** — EnvDetailScreen accesses `env.blueprint_spawns` in `_render_graveyard()`
- [x] **Display is correct** — Value renders with cyan styling in spawn column
- [x] **Thresholds applied** — N/A (no thresholds for spawn counts)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | N/A | No direct blueprint_spawns test found | `[ ]` |
| Unit (aggregator) | N/A | Covered by aggregator tests | `[?]` |
| Integration (end-to-end) | `tests/karn/sanctum/test_env_detail_screen.py` | Graveyard rendering tests | `[x]` |
| Visual (TUI snapshot) | N/A | Manual verification | `[ ]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 100`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Navigate to EnvDetailScreen for an active environment
4. Observe Seed Graveyard panel "spawn" column
5. Verify spawn counts increment when seeds germinate
6. Verify blueprint IDs appear as row labels

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `SEED_GERMINATED` event | event | Emitted when seed transitions to GERMINATED stage |
| `SeedGerminatedPayload.blueprint_id` | field | Identifies which blueprint type was spawned |
| Blueprint registry | computation | Provides valid blueprint IDs |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `TELE-671` blueprint_fossilized | telemetry | Spawn count provides context for success rate |
| `TELE-672` blueprint_prunes | telemetry | Spawn count provides context for failure rate |
| Graveyard success rate | computation | `fossilized / (fossilized + pruned)` uses spawn for context |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Claude | Initial creation during telemetry audit |

---

## 8. Notes

> **Design Decision:** blueprint_spawns is per-episode (resets on episode boundary) for EnvState, but cumulative tracking is also maintained in the aggregator via `_cumulative_blueprint_spawns`.
>
> **Widget Display:** The EnvDetailScreen displays spawn counts in cyan in the "spawn" column of the Seed Graveyard table. The column header is "spawn" (line 673).
>
> **Graveyard Purpose:** The Seed Graveyard provides insights into which blueprint architectures succeed vs fail. Spawn counts establish how many seeds of each type have been tried, enabling calculation of success/failure rates.
>
> **Related Fields:** Works in conjunction with:
> - `blueprint_fossilized` (TELE-671) - successful integrations
> - `blueprint_prunes` (TELE-672) - failed/removed seeds

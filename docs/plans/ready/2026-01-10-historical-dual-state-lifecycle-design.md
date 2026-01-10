# Historical Dual-State Toggle + Seed Lifecycle Log

**Status:** Ready for Implementation
**Created:** 2026-01-10
**Author:** Claude (brainstorming session)

## Overview

This feature adds two capabilities to Sanctum's historical and live views:

1. **Dual-State Toggle**: Historical detail panels can switch between "Peak Accuracy State" (snapshot at best accuracy) and "End State" (snapshot at episode completion), with distinct visual styling for each.

2. **Seed Lifecycle Log**: Both live and historical seed modals display an event log showing Tamiyo's decisions that shaped each seed's journey through its lifecycle stages.

## Motivation

The historical view currently shows `best_seeds` captured at peak accuracy time, but graveyard stats from episode end. This causes confusion when seeds are in TRAINING at peak but pruned by episode end - the graveyard says "2 spawned, 1 pruned" but users only see the surviving seed.

Users need to understand Tamiyo's decision-making journey, not just the final state.

## Data Model

### New Dataclass: SeedLifecycleEvent

```python
@dataclass
class SeedLifecycleEvent:
    """A single lifecycle transition for a seed."""
    epoch: int
    action: str           # GERMINATE, ADVANCE, PRUNE, FOSSILIZE, or "[auto]"
    from_stage: str       # Previous stage (or None for initial)
    to_stage: str         # New stage
    blueprint_id: str     # Which blueprint
    slot_id: str          # Which slot
    alpha: float | None   # Alpha at transition (for BLENDING/HOLDING)
    accuracy_delta: float | None  # Accuracy change (for FOSSILIZE)
```

### EnvState Additions

```python
class EnvState:
    # ... existing fields ...
    lifecycle_events: list[SeedLifecycleEvent]       # Live tracking
    best_lifecycle_events: list[SeedLifecycleEvent]  # Snapshot at peak
```

### BestRunRecord Additions

```python
@dataclass
class BestRunRecord:
    # ... existing fields ...
    end_seeds: dict[str, SeedState]                  # Seeds at episode end
    end_reward_components: RewardComponents | None   # Rewards at end
    best_lifecycle_events: list[SeedLifecycleEvent]  # Events at peak
    end_lifecycle_events: list[SeedLifecycleEvent]   # Events at end
```

## UI Design

### Historical Panel Toggle

The historical detail panel header gains a toggle button and dynamic styling:

```
┌─[cyan]PEAK STATE[/]────────────── Episode# 70 ─── [s] Switch ─┐
│                                                                │
│  (all content shows peak accuracy snapshot)                    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
         ↓ press 's' ↓
┌─[yellow]END STATE[/]─────────────── Episode# 70 ─── [s] Switch ─┐
│                                                                  │
│  (all content shows episode end snapshot)                        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Visual Indicators:**

| State | Border Color | Title Style | Hint Text |
|-------|--------------|-------------|-----------|
| Peak  | `cyan` | `[bold cyan]PEAK STATE[/]` | "Showing state at best accuracy" |
| End   | `yellow` | `[bold yellow]END STATE[/]` | "Showing state at episode end" |

**What Changes Between States:**

| Component | Peak State | End State |
|-----------|------------|-----------|
| Seed slots grid | `best_seeds` | `end_seeds` |
| Graveyard stats | `best_blueprint_*` | `blueprint_*` (end values) |
| Lifecycle events | `best_lifecycle_events` | `end_lifecycle_events` |
| Accuracy shown | `peak_accuracy` | `final_accuracy` |
| Reward components | `best_reward_components` | `end_reward_components` |

### Lifecycle Panel in Seed Modals

```
┌─ Seed: r0c1 (conv_heavy) ───────────────────────────────────────┐
│                                                                  │
│  Stage: FOSSILIZED    Alpha: 1.00    Params: 12.4K              │
│  Accuracy Δ: +2.3%    Grad Ratio: 0.94                          │
│                                                                  │
├─ Lifecycle [f] Filter: r0c1 ─────────────────────────────────────┤
│                                                                  │
│  e12  GERMINATE(conv_heavy)  DORMANT → GERMINATED               │
│  e18  [auto]                 GERMINATED → TRAINING               │
│  e31  ADVANCE                TRAINING → BLENDING      α=0.15    │
│  e42  [auto]                 BLENDING → HOLDING       α=0.85    │
│  e58  FOSSILIZE              HOLDING → FOSSILIZED     +2.3%     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Filter Behavior (`f` key):**

| Current Filter | Press `f` | Result |
|----------------|-----------|--------|
| Single slot (e.g., `r0c1`) | Toggle | Show `All` slots |
| `All` slots | Toggle | Return to original slot |

**Unfiltered "All" View:**

```
├─ Lifecycle [f] Filter: All ──────────────────────────────────────┤
│                                                                  │
│  e05  r0c0  GERMINATE(attention)   DORMANT → GERMINATED         │
│  e08  r0c0  [auto]                 GERMINATED → TRAINING         │
│  e12  r0c1  GERMINATE(conv_heavy)  DORMANT → GERMINATED         │
│  e15  r0c0  PRUNE                  TRAINING → PRUNED             │
│  e18  r0c1  [auto]                 GERMINATED → TRAINING         │
│  ...                                                             │
└──────────────────────────────────────────────────────────────────┘
```

## Event Capture

**When Lifecycle Events Are Created:**

| Telemetry Event | Action Recorded | Stage Transition |
|-----------------|-----------------|------------------|
| `seed_germinated` | `GERMINATE({blueprint})` | `DORMANT → GERMINATED` |
| `seed_stage_changed` | `ADVANCE` / `[auto]` | Depends on stages |
| `seed_pruned` | `PRUNE` | `* → PRUNED` |
| `seed_fossilized` | `FOSSILIZE` | `HOLDING → FOSSILIZED` |

**Distinguishing `[auto]` vs Explicit Actions:**

- `GERMINATED → TRAINING`: Always `[auto]` (happens after warmup epochs)
- `BLENDING → HOLDING`: Always `[auto]` (happens when α crosses threshold)
- `TRAINING → BLENDING`: `ADVANCE` (explicit Tamiyo decision)
- `HOLDING → FOSSILIZED`: `FOSSILIZE` (explicit Tamiyo decision)

**Snapshot Timing:**

```python
# In EnvState.add_accuracy() - when new peak achieved:
self.best_lifecycle_events = list(self.lifecycle_events)

# In aggregator._update_best_runs() - at episode end:
record = BestRunRecord(
    ...
    best_lifecycle_events=env.best_lifecycle_events,   # Peak snapshot
    end_lifecycle_events=list(env.lifecycle_events),   # End snapshot
)
```

## Files to Modify

| File | Changes |
|------|---------|
| `schema.py` | Add `SeedLifecycleEvent` dataclass, add fields to `EnvState` and `BestRunRecord` |
| `aggregator.py` | Capture lifecycle events on seed telemetry, snapshot at peak/end, use `best_blueprint_*` fields |
| `historical_env_detail.py` | Add state toggle (`s` key), color coding, switch data source based on state |
| `seed_detail_modal.py` | Add lifecycle panel section, filter toggle (`f` key) |
| `historical_seed_modal.py` | Same lifecycle panel, plus respect Peak/End state from parent |

**New Widget:**

```
widgets/lifecycle_panel.py  - Reusable lifecycle log component
                            - Takes events list + optional slot filter
                            - Used by both live and historical modals
```

## Key Bindings

| Key | Context | Action |
|-----|---------|--------|
| `s` | Historical detail panel | Toggle Peak ↔ End state |
| `f` | Seed modal (live or historical) | Toggle slot filter ↔ All |

## Testing

- Unit test `SeedLifecycleEvent` creation from telemetry payloads
- Unit test snapshot timing (peak vs end captures correct events)
- Integration test: toggle between Peak/End shows different data
- Integration test: lifecycle filter shows correct events

# Sanctum Telemetry Audit - Master Index

**Audit Date:** 2025-12-26
**Scope:** Complete telemetry wiring audit for all Sanctum TUI panels
**Purpose:** Build a master reference for UX telemetry and confirm correct wiring

## Executive Summary

| Panel | Status | Wiring | Issues |
|-------|--------|--------|--------|
| [RunHeader](./01-run-header.md) | ✅ Production Ready | 95% | 2 dead fields |
| [AnomalyStrip](./02-anomaly-strip.md) | ✅ Production Ready | 100% | None |
| [EnvOverview](./03-env-overview.md) | ✅ Production Ready | ~95% | 1 dead field, 3 missing components |
| [Scoreboard](./04-scoreboard.md) | ✅ Production Ready | 100% | None |
| [RewardHealthPanel](./05-reward-health-panel.md) | ✅ Production Ready | 100% | Minor: missing @dataclass |
| [TamiyoBrain Actions](./06-tamiyo-brain-actions.md) | ✅ Production Ready | 100% | None |
| [TamiyoBrain PPO](./07-tamiyo-brain-ppo.md) | ✅ Production Ready | 95% | Per-head grad norms dead |
| [TamiyoBrain Decisions](./08-tamiyo-brain-decisions.md) | ✅ Production Ready | 100% | All fixed 2025-12-26 |
| [EventLog](./09-event-log.md) | ✅ Production Ready | 70%* | By design (lifecycle focus) |
| [TamiyoBrain Sparklines](./10-tamiyo-brain-sparklines.md) | ✅ Production Ready | 100% | None |

*EventLog coverage is intentionally focused on lifecycle events, not all telemetry.

## Overall Assessment

**Verdict:** The Sanctum TUI telemetry infrastructure is **production-ready**. All critical data paths are correctly wired from training events through aggregation to widget display.

### Key Findings

1. **Thread Safety:** All aggregator operations are properly protected by `threading.Lock`
2. **Data Flow:** Event → Handler → State → Snapshot → Widget pattern consistently applied
3. **History Tracking:** Deques with bounded maxlen prevent memory growth
4. **Trend Detection:** Sparklines include directional indicators for key metrics

## Issues by Priority

### Critical (P0): None

### High (P1): ✅ RESOLVED - Decision Cards Missing Fields
- **Location:** TamiyoBrain Decision Cards
- **Issue:** `decision_entropy`, `alternatives`, `slot_states` not captured
- **Resolution (2025-12-26):** Added fields to `AnalyticsSnapshotPayload`, updated emitter to pass values, updated aggregator to extract into `DecisionSnapshot`

### Medium (P2): Dead Code Cleanup
| File | Field | Action |
|------|-------|--------|
| schema.py:747 | `total_events_received` | Remove from schema |
| schema.py:749 | `training_thread_alive` | Remove or implement |
| schema.py:566 | `seed_contribution` | Remove (uses bounded_attribution) |
| schema.py:423-430 | `head_*_grad_norm` | Remove or implement display |

### Low (P3): Documentation & Enhancements
- Add @dataclass to RewardComponents class
- Document EventLog coverage decisions
- Document action path selection in aggregator
- Consider longer history deques (20-30 instead of 10)

## Telemetry Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Training System (Simic/Kasmina/Tamiyo)                          │
├─────────────────────────────────────────────────────────────────┤
│ Vectorized Training Loop                                        │
│ ├─ EPOCH_COMPLETED (per-env accuracy)                          │
│ ├─ SEED_* events (lifecycle: germinate/stage/fossilize/prune)  │
│ ├─ PPO_UPDATE_COMPLETED (gradients, losses, entropy)           │
│ ├─ BATCH_EPOCH_COMPLETED (episode boundaries)                  │
│ ├─ ANALYTICS_SNAPSHOT (decisions, rewards, actions)            │
│ └─ EPISODE_OUTCOME (multi-objective results)                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Leyline Telemetry (Contracts Layer)                             │
├─────────────────────────────────────────────────────────────────┤
│ TelemetryEvent envelope: event_type, timestamp, slot_id, data   │
│ Typed Payloads: EpochCompletedPayload, PPOUpdatePayload, etc.   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Nissa Event Hub (Distribution Layer)                            │
├─────────────────────────────────────────────────────────────────┤
│ Broadcasts events to registered backends                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ SanctumAggregator (State Management)                            │
├─────────────────────────────────────────────────────────────────┤
│ process_event() → route by event_type                          │
│ _handle_* methods extract payload fields                       │
│ EnvState: per-env metrics, seed states, reward components      │
│ TamiyoState: PPO metrics, action counts, decisions             │
│ Thread-safe: all operations protected by self._lock            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ SanctumSnapshot (Immutable Transfer Object)                     │
├─────────────────────────────────────────────────────────────────┤
│ Frozen copy of aggregator state for UI thread consumption      │
│ envs: dict[str, EnvState]                                      │
│ tamiyo: TamiyoState                                            │
│ vitals: SystemVitals                                           │
│ event_log: list[EventLogEntry]                                 │
│ best_runs: list[BestRunRecord]                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Sanctum Widgets (Presentation Layer)                            │
├─────────────────────────────────────────────────────────────────┤
│ RunHeader      → Episode/Epoch/Batch counters, throughput      │
│ AnomalyStrip   → Stalled/Degraded/Gradient/PPO/Memory warnings │
│ EnvOverview    → Per-env table with accuracy, reward, seeds    │
│ Scoreboard     → Top 10 leaderboard with growth ratio          │
│ RewardHealth   → PBRS fraction, anti-gaming, explained var     │
│ TamiyoBrain    → Action dist, PPO gauges, decisions, sparklines│
│ EventLog       → Append-only event timeline                    │
└─────────────────────────────────────────────────────────────────┘
```

## Event Type Coverage Matrix

| Event Type | Handler | Log Entry | Widgets Using |
|------------|---------|-----------|---------------|
| TRAINING_STARTED | ✅ | ⚠️ | RunHeader |
| EPOCH_COMPLETED | ✅ | ❌ | EnvOverview, Scoreboard, AnomalyStrip |
| PPO_UPDATE_COMPLETED | ✅ | ✅ | TamiyoBrain (all subpanels) |
| SEED_GERMINATED | ✅ | ✅ | EnvOverview, RunHeader, EventLog |
| SEED_STAGE_CHANGED | ✅ | ✅ | EnvOverview, RunHeader, EventLog |
| SEED_FOSSILIZED | ✅ | ✅ | Scoreboard, EnvOverview, EventLog |
| SEED_PRUNED | ✅ | ✅ | Scoreboard, EventLog |
| BATCH_EPOCH_COMPLETED | ✅ | ✅ | RunHeader, Scoreboard, TamiyoBrain |
| ANALYTICS_SNAPSHOT | ✅ | ❌ | EnvOverview, TamiyoBrain, RewardHealth |
| COUNTERFACTUAL_MATRIX | ✅ | ❌ | EnvOverview (detail screen) |
| EPISODE_OUTCOME | ✅ | ❌ | RewardHealth (hypervolume) |

## Files Audited

### Source Files
- `src/esper/karn/sanctum/widgets/run_header.py`
- `src/esper/karn/sanctum/widgets/anomaly_strip.py`
- `src/esper/karn/sanctum/widgets/env_overview.py`
- `src/esper/karn/sanctum/widgets/scoreboard.py`
- `src/esper/karn/sanctum/widgets/reward_health.py`
- `src/esper/karn/sanctum/widgets/tamiyo_brain.py`
- `src/esper/karn/sanctum/widgets/event_log.py`
- `src/esper/karn/sanctum/aggregator.py`
- `src/esper/karn/sanctum/schema.py`
- `src/esper/leyline/telemetry.py`
- `src/esper/simic/telemetry/emitters.py`
- `src/esper/simic/training/vectorized.py`

### Audit Reports
1. [01-run-header.md](./01-run-header.md) - Training progress header
2. [02-anomaly-strip.md](./02-anomaly-strip.md) - Anomaly summary bar
3. [03-env-overview.md](./03-env-overview.md) - Per-environment table
4. [04-scoreboard.md](./04-scoreboard.md) - Best runs leaderboard
5. [05-reward-health-panel.md](./05-reward-health-panel.md) - Reward health metrics
6. [06-tamiyo-brain-actions.md](./06-tamiyo-brain-actions.md) - Action distribution
7. [07-tamiyo-brain-ppo.md](./07-tamiyo-brain-ppo.md) - PPO health gauges
8. [08-tamiyo-brain-decisions.md](./08-tamiyo-brain-decisions.md) - Decision cards
9. [09-event-log.md](./09-event-log.md) - Event timeline
10. [10-tamiyo-brain-sparklines.md](./10-tamiyo-brain-sparklines.md) - History sparklines

### Supplementary Documentation
11. [11-dark-telemetry.md](./11-dark-telemetry.md) - Captured but not displayed telemetry catalog

## Next Steps

1. **Immediate:** No blocking issues for production use
2. **Short-term:** Extend AnalyticsSnapshotPayload for decision enrichment
3. **Medium-term:** Clean up dead fields per no-legacy-code policy
4. **Long-term:** Consider adding EPOCH_COMPLETED to EventLog for visibility

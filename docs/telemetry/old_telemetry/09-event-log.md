# EventLog Telemetry Audit

**Audit Date:** 2025-12-26
**Widget:** `src/esper/karn/sanctum/widgets/event_log.py`
**Status:** ✅ 70% Coverage (by design)

## Overview

The EventLog widget is an append-only, scrolling log displaying telemetry events. Events are grouped by second (waiting for each second to complete before display). Each event type gets its own line with count and contributing environment IDs.

## Telemetry Fields

| Field | Source Event | Status |
|-------|--------------|--------|
| `timestamp` | All events (event.timestamp) | ✅ Wired |
| `event_type` | All events (event_type parameter) | ✅ Wired |
| `env_id` | SEED_*/BATCH_EPOCH/PPO payload extraction | ✅ Wired |
| `message` | Generic per type | ✅ Wired |
| `episode` | self._current_episode | ✅ Wired |
| `relative_time` | Calculated from timestamp | ✅ Wired |
| `metadata` | Type-specific extraction | ✅ Wired |

## Event Type Coverage

**Fully covered (with metadata):**
- ✅ SEED_GERMINATED - slot_id, blueprint
- ✅ SEED_STAGE_CHANGED - slot_id, from_stage, to_stage
- ✅ SEED_FOSSILIZED - slot_id, improvement
- ✅ SEED_PRUNED - slot_id, reason
- ✅ PPO_UPDATE_COMPLETED - entropy, clip_fraction
- ✅ BATCH_EPOCH_COMPLETED - batch_idx, episodes_completed

**Not logged (by design):**
- ⚠️ EPOCH_COMPLETED - handled but no event_log entry
- ⚠️ COUNTERFACTUAL_MATRIX_COMPUTED - handled but no event_log entry
- ⚠️ ANALYTICS_SNAPSHOT - handled but no event_log entry
- ⚠️ EPISODE_OUTCOME - handled but no event_log entry

## Data Flow

```
TelemetryEvent → Aggregator.process_event()
    ↓
_process_event_unlocked() routes by event_type
    ↓
_add_event_log() for covered events:
    - Extract timestamp, calculate relative_time
    - Extract env_id from payload
    - Generate message, extract metadata
    - Append to self._event_log deque (maxlen=100)
    ↓
get_snapshot() → event_log=list(self._event_log)
    ↓
EventLog.update_snapshot()
    - Groups by timestamp
    - Skips current second (still accumulating)
    - Renders grouped lines
```

## Issues Found

### 1. Missing Event Log Entries
- **Events not logged:** EPOCH_COMPLETED, COUNTERFACTUAL, ANALYTICS_SNAPSHOT, EPISODE_OUTCOME
- **Impact:** Log shows ~30% of telemetry events
- **Severity:** Medium - intentional but undocumented design choice

### 2. Thread Safety
- **Status:** ✅ Correctly implemented with `self._lock`

### 3. 1-Second Display Latency
- **Behavior:** Widget skips current second to avoid partial data
- **Status:** Intentional, should be documented

## Recommendations

1. Consider adding EPOCH_COMPLETED entries for major milestone visibility
2. Document which event types are logged vs. skipped and why
3. Add EPISODE_OUTCOME entries to show episode boundaries

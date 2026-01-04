# TamiyoBrain: Decision Cards Telemetry Audit

**Audit Date:** 2025-12-26
**Widget:** `src/esper/karn/sanctum/widgets/tamiyo_brain.py`
**Status:** ✅ 100% Wired (fixed 2025-12-26)

## Overview

The TamiyoBrain Decision Cards subpanel displays enriched decision snapshots showing what Tamiyo saw, chose, and the outcome of each decision. Up to 8 decisions shown in a carousel with 30-second refresh intervals.

Each card shows:
- Action taken, slot target, confidence
- Host accuracy, decision entropy
- Value estimate, value residual, TD advantage
- Actual reward, prediction accuracy (HIT/MISS)
- Alternative action probabilities

## Telemetry Fields

| Field | Source Event | Aggregator Handler | Status |
|-------|--------------|-------------------|--------|
| `chosen_action` | ANALYTICS_SNAPSHOT(last_action) | `_handle_analytics_snapshot()` | ✅ Wired |
| `confidence` | ANALYTICS_SNAPSHOT → action_confidence | `_handle_analytics_snapshot()` | ✅ Wired |
| `expected_value` | ANALYTICS_SNAPSHOT → value_estimate | `_handle_analytics_snapshot()` | ✅ Wired |
| `actual_reward` | ANALYTICS_SNAPSHOT → total_reward | `_handle_analytics_snapshot()` | ✅ Wired |
| `value_residual` | Computed: reward - value_estimate | `_handle_analytics_snapshot()` | ✅ Wired |
| `td_advantage` | Computed: r + γV(s') - V(s) | `_handle_analytics_snapshot()` (pending) | ⚠️ Delayed |
| `host_accuracy` | EPOCH_COMPLETED or ANALYTICS_SNAPSHOT | `_handle_epoch_completed()` | ✅ Wired |
| `chosen_slot` | ANALYTICS_SNAPSHOT → slot_id | `_handle_analytics_snapshot()` | ✅ Wired |
| `decision_entropy` | ANALYTICS_SNAPSHOT → decision_entropy | `_handle_analytics_snapshot()` | ✅ Wired |
| `alternatives` | ANALYTICS_SNAPSHOT → alternatives | `_handle_analytics_snapshot()` | ✅ Wired |
| `slot_states` | ANALYTICS_SNAPSHOT → slot_states | `_handle_analytics_snapshot()` | ✅ Wired |

## Data Flow

```
Training Decision → AnalyticsSnapshotPayload(kind="last_action")
    ↓
SanctumAggregator._handle_analytics_snapshot()
    ├─ Creates DecisionSnapshot with available fields
    ├─ Stores in _pending_decisions[env_id]
    ↓
Next ANALYTICS_SNAPSHOT arrives
    ├─ Retrieves pending decision
    ├─ Computes: td_adv = reward + DEFAULT_GAMMA * value_s' - value_s
    └─ Removes from _pending_decisions
    ↓
Carousel updates (throttled, one per 30s)
    ↓
TamiyoBrain._render_enriched_decision()
```

## Issues Found

### 1. TD Advantage - Wired But Delayed
- **Status:** Correct - requires two consecutive decisions to compute TD(0)
- **Widget shows:** `TD:...` placeholder until computed
- **Severity:** None - expected behavior

### 2. Decision Entropy - ✅ FIXED (2025-12-26)
- **Fix:** Added `decision_entropy` field to `AnalyticsSnapshotPayload`
- **Files changed:**
  - `leyline/telemetry.py`: Added field to payload schema
  - `simic/telemetry/emitters.py`: Pass value in payload construction
  - `karn/sanctum/aggregator.py`: Extract from payload into DecisionSnapshot

### 3. Alternative Actions - ✅ FIXED (2025-12-26)
- **Fix:** Added `alternatives` field to `AnalyticsSnapshotPayload`
- **Format:** `list[tuple[str, float]]` - Top-2 alternative (action, prob) pairs
- **Files changed:** Same as above

### 4. Slot States - ✅ FIXED (2025-12-26)
- **Fix:** Added `slot_states` field to `AnalyticsSnapshotPayload`
- **Format:** `dict[str, str]` - slot_id → state string (e.g., "Training 12%", "Empty")
- **Files changed:** Same as above

## Recommendations

All previously missing fields are now wired. No outstanding recommendations.

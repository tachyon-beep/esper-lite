# TamiyoBrain: Action Distribution Telemetry Audit

**Audit Date:** 2025-12-26
**Widget:** `src/esper/karn/sanctum/widgets/tamiyo_brain.py` (lines 472-531)
**Status:** ✅ 100% Wired

## Overview

The TamiyoBrain widget displays a horizontal stacked bar chart showing the distribution of Tamiyo's lifecycle actions (WAIT, GERMINATE, SET_ALPHA_TARGET, FOSSILIZE, PRUNE, ADVANCE). The subpanel visualizes what percentage of time Tamiyo spends on each action type.

## Telemetry Fields

| Field | Source Event | Aggregator Handler | Status |
|-------|--------------|-------------------|--------|
| `tamiyo.action_counts` | ANALYTICS_SNAPSHOT (kind="last_action" or "action_distribution") | `_handle_analytics_snapshot()` lines 1049-1067 | ✅ Wired |
| `tamiyo.total_actions` | ANALYTICS_SNAPSHOT | `_handle_analytics_snapshot()` lines 1085-1088 | ✅ Wired |

## Data Flow

```
Training Emission (simic/training/vectorized.py)
    ↓
VectorizedEmitter.on_last_action() → ANALYTICS_SNAPSHOT(kind="last_action")
VectorizedEmitter.on_batch_completed() → emit_action_distribution() → ANALYTICS_SNAPSHOT(kind="action_distribution")
    ↓
SanctumAggregator._handle_analytics_snapshot()
    - Path A (action_distribution): Direct populate tamiyo.action_counts
    - Path B (last_action): Accumulate per-env, aggregate in snapshot
    ↓
_get_snapshot_unlocked() aggregates action counts across all envs
    ↓
TamiyoBrain._render_action_distribution_bar()
    - Calculates percentages: pcts[action] = (count / total) * 100
    - Renders horizontal stacked bar with color-coded segments
```

## Action Normalization

Actions are normalized at aggregation (lines 71-103):
- `GERMINATE_CONV_LIGHT/HEAVY/ATTENTION/MLP` → `GERMINATE`
- `FOSSILIZE_G0/G1/G2` → `FOSSILIZE`
- `SET_ALPHA_TARGET_*` → `SET_ALPHA_TARGET`
- `PRUNE_*` → `PRUNE`
- `WAIT` → `WAIT`
- `ADVANCE_*` → `ADVANCE`

## Issues Found

### 1. Dual Path Ambiguity
- **Issue:** Two paths can populate action_counts (direct vs accumulated)
- **Severity:** Low - design concern, not a bug
- **Recommendation:** Document which path is authoritative

### 2. No Cumulative Action History
- **Issue:** Only current-batch distribution shown, no trend over time
- **Severity:** Low - design limitation
- **Recommendation:** Consider adding action_counts_history deque

## Recommendations

1. Document action path selection in aggregator comments
2. Add integration test validating action counts persist through batch completion

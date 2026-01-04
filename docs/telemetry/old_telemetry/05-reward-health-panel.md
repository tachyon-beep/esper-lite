# RewardHealthPanel Telemetry Audit

**Audit Date:** 2025-12-26
**Widget:** `src/esper/karn/sanctum/widgets/reward_health.py`
**Status:** ✅ 100% Wired (after 2025-12-26 fix)

## Overview

The RewardHealthPanel displays four key health indicators for the DRL training system:
1. **PBRS Fraction** - Proportion of PBRS (Potential-Based Reward Shaping) bonus (target: 10-40%)
2. **Anti-Gaming Trigger Rate** - Fraction of steps where anti-gaming penalties fire (target: <5%)
3. **Explained Variance** - Value function quality from PPO (target: >0.5)
4. **Hypervolume** - Multi-objective Pareto frontier indicator

## Telemetry Fields

| Field | Source Event | Aggregator Handler | Emission Path | Status |
|-------|--------------|-------------------|---------------|--------|
| `pbrs_fraction` | ANALYTICS_SNAPSHOT | `compute_reward_health()` | vectorized.py → on_last_action | ✅ Wired |
| `anti_gaming_trigger_rate` | ANALYTICS_SNAPSHOT | `compute_reward_health()` | vectorized.py → on_last_action | ✅ Wired |
| `ev_explained` | PPO_UPDATE_COMPLETED | `_handle_ppo_update()` | simic/agent/ppo.py → TamiyoState | ✅ Wired |
| `hypervolume` | EPISODE_OUTCOME | `_compute_hypervolume()` | collector.py → episode_outcomes | ✅ Wired |

## Data Flow

```
TRAINING STEP (vectorized.py):
  - compute_contribution_reward() returns RewardComponentsTelemetry
  - Extract: stage_bonus, ratio_penalty, alpha_shock
  ↓
EMIT ANALYTICS_SNAPSHOT (on_last_action):
  - AnalyticsSnapshotPayload(kind="last_action", stage_bonus, ratio_penalty, alpha_shock)
  ↓
AGGREGATOR:
  - _handle_analytics_snapshot() updates env.reward_components
  ↓
WIDGET:
  - compute_reward_health() aggregates across envs
  - pbrs_fraction = sum(|stage_bonus|) / sum(|total_reward|)
  - anti_gaming_trigger_rate = count(penalty != 0) / total_steps
```

## Detailed Wiring

### stage_bonus → pbrs_fraction
- **Source:** `rewards.py:compute_contribution_reward()`
- **Emission:** `emitters.py:on_last_action()` line 280
- **Transport:** `AnalyticsSnapshotPayload.stage_bonus`
- **Reception:** `aggregator.py:_handle_analytics_snapshot()` lines 1099-1100
- **Status:** ✅ FULLY WIRED

### ratio_penalty + alpha_shock → anti_gaming_trigger_rate
- **Source:** `rewards.py:compute_contribution_reward()` (anti-gaming penalty)
- **Emission:** `emitters.py:on_last_action()` lines 281-282
- **Transport:** `AnalyticsSnapshotPayload.{ratio_penalty, alpha_shock}`
- **Reception:** `aggregator.py:_handle_analytics_snapshot()` lines 1101-1104
- **Status:** ✅ FULLY WIRED

## Issues Found

### 1. Missing @dataclass Decorator on RewardComponents
- **Location:** `schema.py` line 540
- **Issue:** Class missing `@dataclass` decorator
- **Severity:** Medium - works but non-idiomatic
- **Fix:** Add `@dataclass` decorator

## Recommendations

1. Add `@dataclass` decorator to RewardComponents class
2. Document telemetry config dependency ("ops_normal" required for reward health)

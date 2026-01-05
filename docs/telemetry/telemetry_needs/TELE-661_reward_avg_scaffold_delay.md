# Telemetry Record: [TELE-661] Reward Avg Scaffold Delay

> **Status:** `[x] Planned` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-661` |
| **Name** | Reward Avg Scaffold Delay |
| **Category** | `reward` |
| **Priority** | `P3-nice-to-have` |

## 2. Purpose

### What question does this answer?

> "On average, how many epochs ago did the scaffolding interactions begin for this hindsight credit?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

### When is this information needed?

- [ ] Real-time (every batch/epoch)
- [ ] Periodic (every N episodes)
- [ ] On-demand (when investigating issues)
- [x] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `float` |
| **Units** | epochs |
| **Range** | `[0, +inf]` — always non-negative |
| **Precision** | 1 decimal place for display |
| **Default** | `0.0` |

### Semantic Meaning

> Average scaffold delay measures the temporal distance between scaffolding interactions and the eventual fossilization of the beneficiary seed.
>
> This metric helps understand how far back in time the scaffolding relationships were established. A longer delay means the scaffold seeds provided early-stage support that paid off later; a shorter delay indicates more recent scaffolding.
>
> - **Zero:** No scaffolding (or scaffolding just started this epoch)
> - **Low values (< 5):** Recent scaffolding, quick credit assignment
> - **High values (> 10):** Long-term scaffolding relationships
>
> This is displayed alongside hindsight_credit as "(..., X.Xe)" where X.X is the average delay in epochs.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `value >= 0` | Normal (any delay is valid) |
| **Info** | `value > 5` | Established scaffolding relationships |
| **Note** | `value > 15` | Long-term scaffolding (may indicate slow fossilization) |

**Display Color Logic:** Displayed as metadata alongside hindsight_credit (blue styling inherits from parent)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Average delay computed during hindsight credit calculation |
| **File** | `/home/john/esper-lite/src/esper/simic/rewards/rewards.py` |
| **Function/Method** | `compute_scaffold_hindsight_credit()` |
| **Line(s)** | 1247-1320 |

```python
# Average delay computed from scaffold registry timestamps
# Note: avg_scaffold_delay is NOT currently in RewardComponentsTelemetry
# The schema field exists but transport is incomplete
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Not in `RewardComponentsTelemetry` (gap) | `simic/rewards/reward_telemetry.py` |
| **2. Collection** | N/A | `simic/telemetry/emitters.py` |
| **3. Aggregation** | Default value preserved (no transport) | `karn/sanctum/aggregator.py` (comment at line 1476) |
| **4. Delivery** | Schema field exists with default | `karn/sanctum/schema.py` (line 1130) |

```
[compute_scaffold_hindsight_credit()]
  --NOT WIRED-->
  [RewardComponentsTelemetry] (missing field)
  --N/A-->
  [EnvState.reward_components.avg_scaffold_delay = 0.0 (default)]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `RewardComponents` |
| **Field** | `avg_scaffold_delay` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].reward_components.avg_scaffold_delay` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1130 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvDetailScreen | `widgets/env_detail_screen.py` (line 616) | Displayed alongside hindsight_credit as "(..., X.Xe)" |

---

## 5. Wiring Verification

### Checklist

- [ ] **Emitter exists** — **GAP:** `avg_scaffold_delay` not in RewardComponentsTelemetry
- [ ] **Transport works** — **GAP:** No transport path exists
- [x] **Schema field exists** — `RewardComponents.avg_scaffold_delay: float = 0.0` at line 1130
- [x] **Default is correct** — `0.0` is appropriate placeholder
- [x] **Consumer reads it** — EnvDetailScreen accesses `rc.avg_scaffold_delay` at line 616
- [ ] **Display is correct** — Will always show "0.0e" until transport is wired

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_rewards.py` | Avg scaffold delay tests | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_last_action_populates_fields` | `[ ]` |
| Integration (end-to-end) | `tests/integration/test_reward_telemetry_flow.py` | Telemetry flow integration | `[ ]` |
| Widget (EnvDetailScreen) | `tests/karn/sanctum/test_env_detail_screen.py` | Avg scaffold delay display | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI (opens automatically)
3. Navigate to EnvDetailScreen for an environment with hindsight credit
4. Observe "Credits" row — currently shows "(Nx, 0.0e)" due to missing transport
5. **Expected after wiring:** Should show actual delay value

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Scaffold registry timestamps | state | Needs scaffold start epoch tracking |
| Current epoch | context | Needed to compute delay |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EnvDetailScreen Credits row | display | Shown as "(..., X.Xe)" metadata |
| hindsight_credit (TELE-659) | related | Always displayed together |
| scaffold_count (TELE-660) | related | Displayed in same metadata block |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation - identified transport gap |

---

## 8. Notes

> **KNOWN GAP:** The `avg_scaffold_delay` field exists in the schema (line 1130) and is consumed by EnvDetailScreen (line 616), but the transport path is incomplete:
> - `RewardComponentsTelemetry` does not include an `avg_scaffold_delay` field
> - The aggregator has a comment at line 1476: "avg_scaffold_delay is not in RewardComponentsTelemetry, leave as default"
>
> **TODO:** To complete wiring:
> 1. Add `avg_scaffold_delay: float = 0.0` to `RewardComponentsTelemetry` (reward_telemetry.py)
> 2. Compute and populate the value in `compute_scaffold_hindsight_credit()` (rewards.py)
> 3. Wire the transport in aggregator to copy from `rc.avg_scaffold_delay` to `env.reward_components.avg_scaffold_delay`
>
> **Design Intent:** Average scaffold delay provides temporal context for hindsight credit. A high delay indicates long-term scaffolding relationships that paid off eventually — a healthy sign of ecosystem cooperation. This metric was planned as part of Phase 3.2 but transport was not fully implemented.

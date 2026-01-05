# Telemetry Record: [TELE-651] Reward Base Accuracy Delta

> **Status:** `[x] Planned` `[x] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-651` |
| **Name** | Reward Base Accuracy Delta |
| **Category** | `reward` |
| **Priority** | `P2-important` |

## 2. Purpose

### What question does this answer?

> "How much did the base accuracy change contribute to this reward signal?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

### When is this information needed?

- [x] Real-time (every batch/epoch)
- [ ] Periodic (every N episodes)
- [ ] On-demand (when investigating issues)
- [ ] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `float` |
| **Units** | reward units (proportional to accuracy delta) |
| **Range** | `[-inf, +inf]` — typically small values around -1.0 to +1.0 |
| **Precision** | 2 decimal places for display |
| **Default** | `0.0` |

### Semantic Meaning

> Base accuracy delta represents the legacy shaped reward signal based on raw accuracy improvement.
>
> This is the simplest reward component: if accuracy went up, this is positive; if accuracy went down, this is negative.
>
> - **Positive values:** Host model accuracy improved this step
> - **Negative values:** Host model accuracy degraded this step
> - **Zero:** No accuracy change (stable)
>
> This component is computed in the reward function and reflects the instantaneous accuracy change, separate from seed-specific attribution.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `value >= 0` | Accuracy improving or stable |
| **Warning** | `-0.5 < value < 0` | Minor accuracy regression |
| **Critical** | `value <= -0.5` | Significant accuracy loss |

**Display Color Logic:** Green if >= 0, red if < 0 (see `_format_delta_acc()`)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Reward computation in shaped/contribution reward functions |
| **File** | `/home/john/esper-lite/src/esper/simic/rewards/rewards.py` |
| **Function/Method** | `compute_contribution_reward()` |
| **Line(s)** | 513-514 |

```python
# Wire base_acc_delta for Sanctum ΔAcc display (was never populated - bug fix)
components.base_acc_delta = acc_delta if acc_delta is not None else 0.0
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Included in RewardComponentsTelemetry dataclass | `simic/rewards/reward_telemetry.py` (line 21) |
| **2. Collection** | Passed via `emit_last_action()` in training loop | `simic/telemetry/emitters.py` (line 278) |
| **3. Aggregation** | Extracted in aggregator from LastActionPayload | `karn/sanctum/aggregator.py` (line 1460) |
| **4. Delivery** | Written to `env.reward_components.base_acc_delta` | `karn/sanctum/schema.py` (line 1108) |

```
[compute_contribution_reward()]
  --components.base_acc_delta-->
  [RewardComponentsTelemetry]
  --emit_last_action(reward_components=...)-->
  [LastActionPayload.reward_components]
  --SanctumAggregator.handle_last_action()-->
  [EnvState.reward_components.base_acc_delta]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `RewardComponents` |
| **Field** | `base_acc_delta` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].reward_components.base_acc_delta` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1108 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvOverview | `widgets/env_overview.py` (lines 674-681) | Displayed in "ΔAcc" column with green/red color coding |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `compute_contribution_reward()` sets `components.base_acc_delta` at line 514
- [x] **Transport works** — Value flows through RewardComponentsTelemetry -> LastActionPayload -> aggregator
- [x] **Schema field exists** — `RewardComponents.base_acc_delta: float = 0.0` at line 1108
- [x] **Default is correct** — `0.0` is appropriate default before first reward computation
- [x] **Consumer reads it** — EnvOverview `_format_delta_acc()` directly accesses `env.reward_components.base_acc_delta`
- [x] **Display is correct** — Rendered with `+.2f` format, green if >= 0, red if < 0

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_rewards.py` | Reward computation tests | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_last_action_populates_fields` | `[ ]` |
| Integration (end-to-end) | `tests/integration/test_reward_telemetry_flow.py` | Telemetry flow integration | `[ ]` |
| Widget (EnvOverview) | `tests/karn/sanctum/test_env_overview.py` | Delta acc rendering | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI (opens automatically)
3. Observe EnvOverview table — "ΔAcc" column should show current base accuracy delta
4. Verify color coding: green for positive values, red for negative values
5. After training, query telemetry: `SELECT * FROM rewards WHERE base_acc_delta != 0 LIMIT 10`

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Accuracy computation | metric | Requires valid accuracy measurement from host model |
| Reward computation | function | Only populated when reward is computed (action step) |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EnvOverview ΔAcc column | display | Direct consumer for TUI display |
| Aggregate row mean delta | display | Used in Σ row calculation |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** `base_acc_delta` is the "raw" accuracy signal before any seed-specific attribution. It represents what the host model would have experienced regardless of seed activity.
>
> **Relationship to other components:** This is one of several reward components. The total reward also includes `bounded_attribution`, `compute_rent`, and various bonuses/penalties. This component specifically isolates the base accuracy change.
>
> **Display Format:** The `_format_delta_acc()` function in EnvOverview renders this with `+.2f` format and applies green/red coloring based on sign.

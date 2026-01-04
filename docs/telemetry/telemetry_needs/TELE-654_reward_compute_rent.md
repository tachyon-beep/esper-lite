# Telemetry Record: [TELE-654] Reward Compute Rent

> **Status:** `[x] Planned` `[x] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-654` |
| **Name** | Reward Compute Rent |
| **Category** | `reward` |
| **Priority** | `P2-important` |

## 2. Purpose

### What question does this answer?

> "How much penalty is the agent paying for keeping active seeds (parameter bloat cost)?"

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
| **Units** | reward units (always negative or zero) |
| **Range** | `[-max_rent_penalty, 0]` — typically 0 to -5.0 |
| **Precision** | 2 decimal places for display |
| **Default** | `0.0` |

### Semantic Meaning

> Compute rent is a penalty for active seeds, incentivizing the agent to prune or fossilize seeds rather than keeping them indefinitely.
>
> **Computed as:** `-compute_rent_weight * log(1 + growth_ratio)` where:
> - `growth_ratio = (total_params - host_params) / host_params`
> - Logarithmic scaling prevents extreme penalties for large seeds
> - Capped at `max_rent_penalty` (default 5.0)
>
> - **Zero:** No active seeds or within grace period
> - **Negative values:** Active seeds are costing the agent reward
>
> This component encourages model efficiency by penalizing parameter bloat.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `value == 0` | No compute rent (no active seeds or grace period) |
| **Normal** | `-1.0 < value < 0` | Reasonable rent for active seeds |
| **Warning** | `value <= -1.0` | High rent — consider pruning or fossilizing |

**Display Color Logic:** Always red (it's always a penalty) — see `_format_rent()`

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Compute rent calculation in reward computation |
| **File** | `/home/john/esper-lite/src/esper/simic/rewards/rewards.py` |
| **Function/Method** | `compute_contribution_reward()` |
| **Line(s)** | 747-750 |

```python
if components:
    components.compute_rent = -rent_penalty  # Negative because it's a penalty
    components.growth_ratio = growth_ratio  # DRL Expert diagnostic field
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Included in RewardComponentsTelemetry dataclass | `simic/rewards/reward_telemetry.py` (line 31) |
| **2. Collection** | Passed via `emit_last_action()` in training loop | `simic/telemetry/emitters.py` (line 278) |
| **3. Aggregation** | Extracted in aggregator from LastActionPayload | `karn/sanctum/aggregator.py` (line 1464) |
| **4. Delivery** | Written to `env.reward_components.compute_rent` | `karn/sanctum/schema.py` (line 1092-1093) |

```
[compute_contribution_reward()]
  --components.compute_rent = -rent_penalty-->
  [RewardComponentsTelemetry.compute_rent]
  --emit_last_action(reward_components=...)-->
  [LastActionPayload.reward_components]
  --SanctumAggregator.handle_last_action()-->
  [EnvState.reward_components.compute_rent]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `RewardComponents` |
| **Field** | `compute_rent` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].reward_components.compute_rent` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1092-1093 (in docstring), field definition nearby |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvOverview | `widgets/env_overview.py` (lines 698-704) | Displayed in "Rent" column with red color coding |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — Compute rent calculation sets `components.compute_rent` at line 749
- [x] **Transport works** — Value flows through RewardComponentsTelemetry -> LastActionPayload -> aggregator
- [x] **Schema field exists** — `RewardComponents.compute_rent` documented in schema
- [x] **Default is correct** — `0.0` is appropriate when no rent is applied
- [x] **Consumer reads it** — EnvOverview `_format_rent()` directly accesses `env.reward_components.compute_rent`
- [x] **Display is correct** — Rendered with `.2f` format, always red (penalty)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_rewards.py` | Compute rent calculation tests | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_last_action_populates_fields` | `[ ]` |
| Integration (end-to-end) | `tests/integration/test_reward_telemetry_flow.py` | Telemetry flow integration | `[ ]` |
| Widget (EnvOverview) | `tests/karn/sanctum/test_env_overview.py` | Rent column rendering | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI (opens automatically)
3. Observe EnvOverview table — "Rent" column should show compute rent penalty
4. Verify color: always red for non-zero values
5. Verify format: "X.XX" (positive number displayed for the magnitude)
6. Note: Column shows "─" when compute_rent is 0 (no active seeds)

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Active seed parameters | state | Requires total_params from active seeds |
| Host parameters | state | Requires host_params for growth_ratio calculation |
| Configuration | config | compute_rent_weight, max_rent_penalty, grace_epochs |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EnvOverview Rent column | display | Direct consumer for TUI display |
| Aggregate row mean rent | display | Used in Σ row calculation |
| Total reward | computation | Subtracts from final reward signal |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** Compute rent uses logarithmic scaling to prevent extreme penalties for large seeds while still discouraging parameter bloat. The formula `log(1 + growth_ratio)` provides smooth, bounded growth in penalty.
>
> **Grace Period:** New seeds have a `grace_epochs` period (default 3) where no rent is charged. This allows seeds time to prove their value before incurring costs.
>
> **Configuration:**
> - `compute_rent_weight`: Scales the rent penalty (default 0.05, simplified uses 0.01)
> - `max_rent_penalty`: Caps maximum penalty (default 5.0)
> - `grace_epochs`: Rent-free period for new seeds (default 3)
>
> **Display Format:** The `_format_rent()` function in EnvOverview renders this with `.2f` format and always applies red styling since it's always a penalty. The value is stored as negative in the schema but displayed as positive magnitude.
>
> **Growth Ratio Context:** The `growth_ratio` field in RewardComponentsTelemetry is also emitted alongside compute_rent for diagnostic purposes. This shows `(total_params - host_params) / host_params` and helps debug why rent is high.

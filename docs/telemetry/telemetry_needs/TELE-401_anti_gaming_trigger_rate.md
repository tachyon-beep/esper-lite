# Telemetry Record: [TELE-401] Anti-Gaming Trigger Rate

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-401` |
| **Name** | Anti-Gaming Trigger Rate |
| **Category** | `reward` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Is the policy exploiting reward hacking loopholes, triggering anti-gaming penalties too frequently?"

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
| **Units** | fraction (0-1) |
| **Range** | `[0.0, 1.0]` |
| **Precision** | 2 decimal places for display (percentage format) |
| **Default** | `0.0` |

### Semantic Meaning

> Anti-gaming trigger rate measures the fraction of reward computation steps where either `ratio_penalty` or `alpha_shock` penalties were applied. These penalties are designed to prevent reward hacking:
>
> - **ratio_penalty**: Applied when seed contribution vastly exceeds actual improvement (suspicious dependency gaming)
> - **alpha_shock**: Convex penalty on alpha (blending weight) changes, preventing rapid oscillation
>
> Computed as:
> ```
> gaming_rate = count(steps where ratio_penalty != 0 OR alpha_shock != 0) / total_steps
> ```
>
> High trigger rate indicates the policy may be exploiting reward shaping rather than learning genuine improvement strategies.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `< 0.05` (5%) | Normal operation, occasional edge cases triggering penalties |
| **Warning** | `0.05 <= value < 0.10` | Policy may be drifting toward exploitation |
| **Critical** | `>= 0.10` (10%+) | Policy actively gaming the reward function |

**Threshold Source:** `RewardHealthData.is_gaming_healthy` property checks `< 0.05` (5%)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Reward computation in simic rewards module |
| **File** | `/home/john/esper-lite/src/esper/simic/rewards/rewards.py` |
| **Function/Method** | `compute_contribution_reward()` (ratio_penalty) and `compute_rent_and_shaping()` (alpha_shock) |
| **Line(s)** | ~527-568 (ratio_penalty), ~754-759 (alpha_shock) |

```python
# ratio_penalty computation (lines 554-561)
if ratio > config.hacking_ratio_threshold:
    ratio_penalty = -min(
        -config.prune_good_seed_penalty,
        0.1 * (ratio - config.hacking_ratio_threshold) / config.hacking_ratio_threshold
    )
# ...
components.ratio_penalty = ratio_penalty

# alpha_shock computation (lines 754-759)
alpha_shock = 0.0
if alpha_delta_sq_sum > 0 and config.alpha_shock_coef != 0.0 and not config.disable_anti_gaming:
    alpha_shock = -config.alpha_shock_coef * alpha_delta_sq_sum
    reward += alpha_shock
components.alpha_shock = alpha_shock
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `RewardComponents` dataclass with `ratio_penalty` and `alpha_shock` fields | `simic/rewards/reward_telemetry.py` |
| **2. Collection** | `RewardComponentsTelemetry` payload in telemetry events | `leyline/telemetry.py` |
| **3. Aggregation** | `SanctumAggregator` copies to `EnvState.reward_components`, tracks `gaming_trigger_count` | `karn/sanctum/aggregator.py` |
| **4. Delivery** | `compute_reward_health()` calculates aggregate rate across all envs | `karn/sanctum/aggregator.py` |

```
[compute_contribution_reward()] --> [RewardComponents]
        |
        v
[TelemetryEvent with RewardComponentsTelemetry payload]
        |
        v
[SanctumAggregator._handle_step_completed()]
        |-- Updates env.reward_components.ratio_penalty
        |-- Updates env.reward_components.alpha_shock
        |-- Increments env.gaming_trigger_count if either != 0
        v
[SanctumAggregator.compute_reward_health()]
        |-- Iterates all envs
        |-- Counts components where ratio_penalty != 0 OR alpha_shock != 0
        |-- Returns RewardHealthData(anti_gaming_trigger_rate=gaming_rate)
        v
[RewardHealthData.anti_gaming_trigger_rate]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `RewardHealthData` |
| **Field** | `anti_gaming_trigger_rate` |
| **Path from SanctumSnapshot** | Not in SanctumSnapshot; computed on-demand via `SanctumBackend.compute_reward_health()` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/widgets/reward_health.py` |
| **Schema Line** | 24 |

```python
@dataclass
class RewardHealthData:
    """Aggregated reward health metrics."""

    pbrs_fraction: float = 0.0  # |PBRS| / |total_reward|
    anti_gaming_trigger_rate: float = 0.0  # Fraction of steps with penalties
    ev_explained: float = 0.0  # Value function explained variance
    hypervolume: float = 0.0  # Pareto hypervolume indicator
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| RewardHealthPanel | `widgets/reward_health.py` | Displays as "Game:X%" with green/red color based on health |
| ActionContext | `widgets/tamiyo_brain/action_distribution.py` | Displays as "Gam:X%" with checkmark/X indicator |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `ratio_penalty` and `alpha_shock` computed in rewards.py
- [x] **Transport works** — Values flow via `RewardComponentsTelemetry` payload
- [x] **Schema field exists** — `RewardHealthData.anti_gaming_trigger_rate: float = 0.0`
- [x] **Default is correct** — 0.0 appropriate (no penalties triggered yet)
- [x] **Consumer reads it** — Both RewardHealthPanel and ActionContext display the value
- [x] **Display is correct** — Rendered as percentage with color coding
- [x] **Thresholds applied** — `is_gaming_healthy` property checks `< 0.05`

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (dataclass) | `tests/karn/sanctum/test_reward_health.py` | `test_reward_health_data_from_components` | `[x]` |
| Unit (boundaries) | `tests/karn/sanctum/test_reward_health.py` | `test_reward_health_warnings` | `[x]` |
| Unit (defaults) | `tests/karn/sanctum/test_reward_health.py` | `test_reward_health_default_values` | `[x]` |
| Unit (ablation) | `tests/simic/test_ablation_flags.py` | `test_disable_anti_gaming_zeroes_ratio_penalty` | `[x]` |
| Unit (ablation) | `tests/simic/test_ablation_flags.py` | `test_disable_anti_gaming_zeroes_alpha_shock` | `[x]` |
| Integration (aggregator) | — | `test_compute_reward_health_gaming_rate` | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe ActionContext panel "Reward Signal" section
4. Look for "Gam:X%" display with green checkmark (healthy) or red X (unhealthy)
5. Also visible in RewardHealthPanel (if displayed) as "Game:X%"
6. To trigger unhealthy state, use ablation: `--disable-anti-gaming=false` and run with aggressive policy

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `ratio_penalty` computation | reward | Requires seed contribution and total improvement data |
| `alpha_shock` computation | reward | Requires alpha_delta_sq_sum from blending |
| `RewardComponentsTelemetry` | payload | Carries penalty values from simic to aggregator |
| Per-env reward_components | state | Each EnvState stores latest reward breakdown |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `RewardHealthData.is_gaming_healthy` | property | Threshold check for display coloring |
| ActionContext widget | display | Shows gaming rate with health indicator |
| RewardHealthPanel widget | display | Shows gaming rate with health coloring |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Telemetry record created for TELE-401 |

---

## 8. Notes

> **Design Decision:** The trigger rate is computed across ALL environments at snapshot time, not per-environment. This provides a global view of policy behavior rather than per-env noise.
>
> **Ablation Support:** The `disable_anti_gaming` config flag (in `ContributionRewardConfig`) can disable both `ratio_penalty` and `alpha_shock` for ablation experiments. When disabled, this metric will always be 0.0.
>
> **Computation Detail:** The aggregator tracks `gaming_trigger_count` per-env incrementally, but `compute_reward_health()` re-computes from the latest reward_components snapshot for simplicity. This means the displayed rate reflects the current batch, not the cumulative history.
>
> **Related Metrics:**
> - `EnvState.gaming_trigger_count` — Cumulative count per-env
> - `EnvState.gaming_rate` — Computed property: `gaming_trigger_count / total_reward_steps`
> - `RewardComponents.ratio_penalty` — Individual penalty value
> - `RewardComponents.alpha_shock` — Individual penalty value

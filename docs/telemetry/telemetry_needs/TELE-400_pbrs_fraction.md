# Telemetry Record: [TELE-400] PBRS Fraction

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-400` |
| **Name** | PBRS Fraction |
| **Category** | `reward` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Is Potential-Based Reward Shaping contributing the right proportion of total reward, or is it dominating/underwhelming the signal?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [x] Automated system (alerts/intervention)

### When is this information needed?

- [x] Real-time (every batch/epoch)
- [ ] Periodic (every N episodes)
- [ ] On-demand (when investigating issues)
- [x] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `float` |
| **Units** | ratio (0-1, displayed as percentage) |
| **Range** | `[0.0, 1.0]` |
| **Precision** | 2 decimal places (displayed as whole percent) |
| **Default** | `0.0` |

### Semantic Meaning

> Ratio of Potential-Based Reward Shaping (PBRS) magnitude to total reward magnitude.
>
> Computed as: `pbrs_fraction = |stage_bonus| / |total_reward|`
>
> PBRS rewards are computed using the telescoping property:
> `F(s,a,s') = gamma * phi(s') - phi(s)`
>
> Where `phi(s)` is the potential function based on lifecycle stage and epochs-in-stage.
> The `stage_bonus` field in reward components captures this PBRS shaping reward.
>
> **Interpretation:**
> - Low PBRS fraction (< 10%): Shaping signal too weak to guide learning
> - Healthy PBRS fraction (10-40%): Shaping provides gradient without dominating
> - High PBRS fraction (> 40%): Risk of reward hacking on shaping terms

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `0.1 <= pbrs_fraction <= 0.4` | PBRS shapes without dominating (DRL Expert recommendation) |
| **Warning** | `pbrs_fraction < 0.1` | Shaping signal too weak, may slow learning |
| **Critical** | `pbrs_fraction > 0.4` | Shaping dominates; risk of reward hacking |

**Threshold Source:** `DisplayThresholds.PBRS_HEALTHY_MIN = 0.1`, `DisplayThresholds.PBRS_HEALTHY_MAX = 0.4` in `/home/john/esper-lite/src/esper/karn/constants.py:177-178`

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Reward computation in contribution reward function |
| **File** | `/home/john/esper-lite/src/esper/simic/rewards/rewards.py` |
| **Function/Method** | `_contribution_pbrs_bonus()` and `compute_contribution_reward()` |
| **Line(s)** | ~709-717 (pbrs_bonus assignment), ~1175-1217 (_contribution_pbrs_bonus) |

```python
# In compute_contribution_reward():
# === 2. PBRS: Stage Progression ===
pbrs_bonus = 0.0
if seed_info is not None and not config.disable_pbrs:
    pbrs_bonus = _contribution_pbrs_bonus(seed_info, config)
    reward += pbrs_bonus
if components:
    components.pbrs_bonus = pbrs_bonus
```

The `stage_bonus` field in `RewardComponentsTelemetry` is populated from `pbrs_bonus`:
```python
# In RewardComponentsTelemetry (reward_telemetry.py:37-38):
stage_bonus: float = 0.0
pbrs_bonus: float = 0.0
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `reward_components` in `EpochPayload.reward_components` | `leyline/telemetry.py:1296` |
| **2. Collection** | `TelemetryEvent` with `EPOCH_COMPLETED` type | `leyline/telemetry.py:72` |
| **3. Aggregation** | `SanctumAggregator._handle_last_action()` updates `env.reward_components.stage_bonus` | `karn/sanctum/aggregator.py:1465` |
| **4. Delivery** | `SanctumAggregator.compute_reward_health()` computes `pbrs_fraction` | `karn/sanctum/aggregator.py:1642-1677` |

```
[rewards.py] --compute_contribution_reward()--> [RewardComponentsTelemetry.stage_bonus]
    --> [EpochPayload.reward_components] --TelemetryEvent--> [Aggregator._handle_last_action()]
    --> [EnvState.reward_components.stage_bonus] --compute_reward_health()--> [RewardHealthData.pbrs_fraction]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `RewardHealthData` |
| **Field** | `pbrs_fraction` |
| **Path from computation** | `aggregator.compute_reward_health().pbrs_fraction` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/widgets/reward_health.py` |
| **Schema Line** | 23 |

**Aggregation Logic** (aggregator.py:1654-1657):
```python
# PBRS proxy: stage_bonus is the PBRS shaping reward
pbrs_total = sum(abs(c.stage_bonus) for c in components)
reward_total = sum(abs(c.total) for c in components)
pbrs_fraction = pbrs_total / max(1e-8, reward_total)
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| RewardHealthPanel | `widgets/reward_health.py:70-71` | Displays "PBRS:XX%" with green/red coloring |
| ActionDistributionWidget | `widgets/tamiyo_brain/action_distribution.py:304-308` | Shows "PBRS:XX%{icon}" in health metrics row |
| HistoricalEnvDetail | `widgets/historical_env_detail.py:297-307` | Shows PBRS fraction in reward breakdown |
| EnvDetailScreen | `widgets/env_detail_screen.py:550-560` | Shows PBRS fraction in live reward breakdown |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `_contribution_pbrs_bonus()` computes PBRS and stores in `components.pbrs_bonus`
- [x] **Transport works** — `EpochPayload.reward_components` carries the data
- [x] **Schema field exists** — `RewardHealthData.pbrs_fraction: float = 0.0`
- [x] **Default is correct** — 0.0 appropriate before first reward computation
- [x] **Consumer reads it** — RewardHealthPanel, ActionDistributionWidget consume it
- [x] **Display is correct** — Value renders as percentage with health icons
- [x] **Thresholds applied** — 10-40% range with green/red coloring

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (data) | `tests/karn/sanctum/test_reward_health.py` | `test_reward_health_data_from_components` | `[x]` |
| Unit (boundaries) | `tests/karn/sanctum/test_reward_health.py` | `test_reward_health_pbrs_boundary_low/high` | `[x]` |
| Unit (at boundaries) | `tests/karn/sanctum/test_reward_health.py` | `test_reward_health_pbrs_at_boundaries` | `[x]` |
| Unit (warnings) | `tests/karn/sanctum/test_reward_health.py` | `test_reward_health_warnings` | `[x]` |
| Unit (defaults) | `tests/karn/sanctum/test_reward_health.py` | `test_reward_health_default_values` | `[x]` |
| Unit (panel) | `tests/karn/sanctum/test_reward_health.py` | `test_reward_health_panel_*` | `[x]` |
| Integration (aggregator) | — | — | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe RewardHealthPanel in the TamiyoBrain widget
4. Verify PBRS:XX% updates after each action step
5. Check coloring: green for 10-40%, red for outside range
6. Observe ActionDistributionWidget health metrics row for "PBRS:XX%" display

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `stage_bonus` in `RewardComponentsTelemetry` | data | Source field for PBRS magnitude |
| `total_reward` in `RewardComponentsTelemetry` | data | Denominator for fraction calculation |
| Seed lifecycle stages | computation | PBRS potentials depend on `SeedStage` progression |
| `ContributionRewardConfig.pbrs_weight` | config | Weight applied to PBRS bonus (default 0.3) |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| RewardHealthPanel display | display | Primary consumer in Sanctum TUI |
| ActionDistributionWidget health row | display | Shows PBRS health status |
| Reward hacking detection | analysis | High PBRS fraction may indicate exploitation |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Telemetry Audit | Initial creation documenting existing wiring |

---

## 8. Notes

> **Design Decision:** PBRS fraction uses `stage_bonus` as a proxy for PBRS contribution. The `stage_bonus` field captures the potential-based shaping reward computed by `_contribution_pbrs_bonus()`. This is accurate for the CONTRIBUTION reward family but may not reflect all PBRS components if other reward families are used.
>
> **DRL Expert Recommendation:** The 10-40% healthy range is based on DRL expert guidance that PBRS should shape learning without dominating the primary reward signal. Below 10%, the shaping gradient is too weak to accelerate learning. Above 40%, the agent may optimize for stage transitions rather than actual performance improvement.
>
> **Note on `pbrs_bonus` vs `stage_bonus`:** The codebase has both fields in `RewardComponentsTelemetry`. The aggregator uses `stage_bonus` for the fraction calculation, but `pbrs_bonus` is also populated. These should have the same value when using CONTRIBUTION rewards.
>
> **Ablation Support:** PBRS can be disabled via `ContributionRewardConfig.disable_pbrs = True` for ablation experiments. When disabled, `pbrs_fraction` will be 0.0.

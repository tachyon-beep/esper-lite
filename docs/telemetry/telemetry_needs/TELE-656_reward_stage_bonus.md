# Telemetry Record: [TELE-656] Reward Stage Bonus

> **Status:** `[x] Planned` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-656` |
| **Name** | Reward Stage Bonus |
| **Category** | `reward` |
| **Priority** | `P1-critical` |

## 2. Purpose

### What question does this answer?

> "How much PBRS (Potential-Based Reward Shaping) bonus did this seed receive for reaching an advanced lifecycle stage?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [x] Automated system (alerts/intervention)

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
| **Units** | reward units (PBRS shaping bonus) |
| **Range** | `[0, +inf]` — typically 0.0 to 0.5 depending on stage |
| **Precision** | 3 decimal places for display |
| **Default** | `0.0` |

### Semantic Meaning

> Stage bonus is a PBRS (Potential-Based Reward Shaping) component that rewards seeds for progressing through the lifecycle.
>
> Seeds in advanced stages (BLENDING and beyond) receive a bonus to encourage progression toward fossilization. This is a "potential difference" reward that incentivizes lifecycle advancement without distorting the optimal policy.
>
> - **Zero:** Seed in early stages (DORMANT, GERMINATED, TRAINING) or no active seed
> - **Positive:** Seed in advanced stage (BLENDING, HOLDING, GRAFTED)
> - **Higher values:** More advanced stages receive larger bonuses
>
> The stage bonus is used to calculate the PBRS fraction: `abs(stage_bonus) / abs(total)`. A healthy PBRS fraction is 10-40% — too low means shaping isn't guiding behavior, too high suggests reward hacking.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | PBRS fraction 10-40% | Shaping is guiding behavior appropriately |
| **Warning** | PBRS fraction < 10% or > 40% | Shaping may be too weak or dominant |
| **Critical** | PBRS fraction > 60% | Agent may be optimizing for shaping bonuses |

**Display Color Logic:** Blue styling for stage bonuses (PBRS category)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PBRS stage bonus computation |
| **File** | `/home/john/esper-lite/src/esper/simic/rewards/rewards.py` |
| **Function/Method** | `compute_pbrs_stage_bonus()` |
| **Line(s)** | 1562-1621 |

```python
def compute_pbrs_stage_bonus(
    seed_info: SeedInfo,
    config: RewardConfig,
    ...
) -> float:
    # Computes stage-dependent bonus based on seed lifecycle stage
    reward += compute_pbrs_stage_bonus(seed_info, config)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Stored in `RewardComponentsTelemetry.stage_bonus` | `simic/rewards/reward_telemetry.py` (line 37) |
| **2. Collection** | Passed via `emit_last_action()` in training loop | `simic/telemetry/emitters.py` |
| **3. Aggregation** | Extracted in aggregator from LastActionPayload | `karn/sanctum/aggregator.py` (line 1465) |
| **4. Delivery** | Written to `env.reward_components.stage_bonus` | `karn/sanctum/schema.py` (line 1120) |

```
[compute_pbrs_stage_bonus()]
  --components.stage_bonus-->
  [RewardComponentsTelemetry]
  --emit_last_action(reward_components=...)-->
  [LastActionPayload.reward_components]
  --SanctumAggregator.handle_last_action()-->
  [EnvState.reward_components.stage_bonus]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `RewardComponents` |
| **Field** | `stage_bonus` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].reward_components.stage_bonus` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1120 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvDetailScreen | `widgets/env_detail_screen.py` (lines 550-560) | Used to calculate PBRS fraction: `abs(stage_bonus) / abs(total)` |
| EnvDetailScreen | `widgets/env_detail_screen.py` (lines 621-625) | Displayed in "Credits" row as "Stage: +0.123" with blue styling |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `compute_pbrs_stage_bonus()` returns stage bonus, stored in components at line 1621
- [x] **Transport works** — Value flows through RewardComponentsTelemetry -> LastActionPayload -> aggregator
- [x] **Schema field exists** — `RewardComponents.stage_bonus: float = 0.0` at line 1120
- [x] **Default is correct** — `0.0` is appropriate default (no bonus for early stages)
- [x] **Consumer reads it** — EnvDetailScreen accesses `rc.stage_bonus` at lines 550 and 621
- [x] **Display is correct** — PBRS fraction shown with healthy/warning icon, stage bonus shown in Credits row with blue styling

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_rewards.py` | PBRS stage bonus tests | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_last_action_populates_fields` | `[ ]` |
| Integration (end-to-end) | `tests/integration/test_reward_telemetry_flow.py` | Telemetry flow integration | `[ ]` |
| Widget (EnvDetailScreen) | `tests/karn/sanctum/test_env_detail_screen.py` | PBRS fraction rendering | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI (opens automatically)
3. Navigate to EnvDetailScreen for any environment with seeds in BLENDING+ stage
4. Observe "Reward Total" row — should show PBRS fraction with healthy/warning icon
5. Observe "Credits" row — should show "Stage: +X.XXX" for non-zero stage bonus
6. After training, query telemetry: `SELECT stage_bonus FROM rewards WHERE stage_bonus != 0 LIMIT 10`

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Seed lifecycle stage | state | Stage bonus depends on current seed stage (BLENDING+) |
| PBRS configuration | config | Bonus magnitude controlled by reward config |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EnvDetailScreen PBRS fraction | display | Used as numerator: `abs(stage_bonus) / abs(total)` |
| EnvDetailScreen Credits row | display | Displayed as "Stage: +X.XXX" with blue styling |
| Reward health assessment | analysis | PBRS fraction used to detect reward hacking |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** Stage bonus implements PBRS (Potential-Based Reward Shaping) from the reward shaping literature. PBRS is theoretically guaranteed to not change the optimal policy, making it safe for shaping without distorting learning.
>
> **PBRS Fraction Thresholds:** The healthy range of 10-40% (DisplayThresholds.PBRS_HEALTHY_MIN/MAX) was tuned empirically. Below 10%, shaping isn't providing meaningful guidance. Above 40%, the agent may be optimizing for shaping bonuses rather than true value.
>
> **Stage Progression:** The botanical metaphor applies: seeds progress DORMANT -> GERMINATED -> TRAINING -> BLENDING -> GRAFTED/FOSSILIZED. Stage bonus only applies to advanced stages where the seed is nearing permanent integration.

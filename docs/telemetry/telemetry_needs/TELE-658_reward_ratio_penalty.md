# Telemetry Record: [TELE-658] Reward Ratio Penalty

> **Status:** `[x] Planned` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-658` |
| **Name** | Reward Ratio Penalty |
| **Category** | `reward` |
| **Priority** | `P1-critical` |

## 2. Purpose

### What question does this answer?

> "Is the agent gaming the system by claiming high seed contribution while total improvement is low or negative (ransomware pattern)?"

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
| **Units** | reward units (penalty) |
| **Range** | `[-inf, 0]` — always negative or zero (penalty) |
| **Precision** | 3 decimal places for display |
| **Default** | `0.0` |

### Semantic Meaning

> Ratio penalty is an anti-gaming mechanism that detects "ransomware" behavior: seeds that claim high contribution while the total improvement is low or negative.
>
> This pattern indicates a seed that has made itself essential (high contribution) while actually degrading overall performance — holding the system "hostage" for credit. The penalty is proportional to the ratio of seed_contribution / total_improvement when this ratio exceeds a threshold.
>
> - **Zero:** Healthy ratio — seed contribution is proportional to improvement (no gaming)
> - **Negative:** Penalty applied for suspicious contribution/improvement ratio
> - **Larger magnitude:** More egregious ransomware behavior
>
> The penalty is computed when `seed_contribution / total_improvement > 5.0` (500% of total improvement) or when contribution is high but total improvement is negative.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `value == 0` | Contribution proportional to improvement |
| **Warning** | `-0.05 < value < 0` | Minor ratio violation detected |
| **Critical** | `value <= -0.05` | Significant ransomware behavior, may need intervention |

**Display Color Logic:** Red styling when triggered (penalty), contributes to "Gaming" indicator

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Ratio penalty computation in contribution reward |
| **File** | `/home/john/esper-lite/src/esper/simic/rewards/rewards.py` |
| **Function/Method** | `compute_contribution_reward()` |
| **Line(s)** | 527-618, 652 |

```python
# Pre-compute attribution_discount and ratio_penalty for ALL seeds
ratio_penalty = 0.0
...
if total_imp > safe_threshold:
    ratio = seed_contribution / total_imp
    if ratio > config.max_contribution_ratio:
        ratio_penalty = -min(
            config.max_ratio_penalty,
            (ratio - config.max_contribution_ratio) * 0.1
        )
...
components.ratio_penalty = ratio_penalty
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Stored in `RewardComponentsTelemetry.ratio_penalty` | `simic/rewards/reward_telemetry.py` (line 28) |
| **2. Collection** | Passed via `emit_last_action()` in training loop | `simic/telemetry/emitters.py` |
| **3. Aggregation** | Extracted in aggregator from LastActionPayload | `karn/sanctum/aggregator.py` (line 1466) |
| **4. Delivery** | Written to `env.reward_components.ratio_penalty` | `karn/sanctum/schema.py` (line 1117) |

```
[compute_contribution_reward()]
  --components.ratio_penalty-->
  [RewardComponentsTelemetry]
  --emit_last_action(reward_components=...)-->
  [LastActionPayload.reward_components]
  --SanctumAggregator.handle_last_action()-->
  [EnvState.reward_components.ratio_penalty]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `RewardComponents` |
| **Field** | `ratio_penalty` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].reward_components.ratio_penalty` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1117 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvDetailScreen | `widgets/env_detail_screen.py` (lines 584-588) | Displayed in "Signals" row as "Ratio: -0.05" with red styling |
| EnvDetailScreen | `widgets/env_detail_screen.py` (line 594) | Triggers "Gaming: X% (RATIO)" indicator when non-zero |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `compute_contribution_reward()` sets `components.ratio_penalty` at line 652
- [x] **Transport works** — Value flows through RewardComponentsTelemetry -> LastActionPayload -> aggregator
- [x] **Schema field exists** — `RewardComponents.ratio_penalty: float = 0.0` at line 1117
- [x] **Default is correct** — `0.0` is appropriate default (no penalty when not triggered)
- [x] **Consumer reads it** — EnvDetailScreen accesses `rc.ratio_penalty` at lines 584 and 594
- [x] **Display is correct** — Rendered with `.3f` format, red styling, triggers RATIO indicator

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_rewards.py` | Ratio penalty tests | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_last_action_populates_fields` | `[ ]` |
| Integration (end-to-end) | `tests/integration/test_reward_telemetry_flow.py` | Telemetry flow integration | `[ ]` |
| Widget (EnvDetailScreen) | `tests/karn/sanctum/test_env_detail_screen.py` | Gaming indicator rendering | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI (opens automatically)
3. Navigate to EnvDetailScreen for any environment
4. Observe "Signals" row — if gaming detected, shows "Ratio: -X.XXX" with red styling
5. Observe gaming indicator — shows "Gaming: X% (RATIO)" when ratio_penalty is non-zero
6. After training, query telemetry: `SELECT ratio_penalty FROM rewards WHERE ratio_penalty != 0 LIMIT 10`

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Seed contribution | metric | Measures how much the seed claims to contribute |
| Total improvement | metric | Measures actual accuracy improvement |
| Anti-gaming config | config | Controlled by `max_contribution_ratio` and `max_ratio_penalty` |
| `disable_anti_gaming` flag | config | Can be disabled for debugging |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EnvDetailScreen Signals row | display | Displayed as "Ratio: -X.XXX" |
| EnvDetailScreen gaming indicator | display | Triggers "(RATIO)" state in gaming display |
| Gaming rate calculation | derived | Contributes to `gaming_trigger_count` when non-zero |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** Ratio penalty detects the "ransomware" anti-pattern where a seed makes itself appear essential while actually hurting overall performance. This is analogous to ransomware that encrypts files — the seed creates dependencies (high contribution) while degrading value (low/negative total improvement).
>
> **Anti-Gaming Mechanism:** Together with `alpha_shock`, ratio penalty forms the anti-gaming defense. Ratio penalty targets contribution inflation, while alpha shock targets alpha manipulation. Both are displayed in the "Signals" row and contribute to the gaming rate.
>
> **Threshold Tuning:** The default `max_contribution_ratio = 5.0` means a seed claiming more than 500% of total improvement is suspicious. This is generous enough to allow legitimate high-contribution scenarios while catching obvious gaming.
>
> **Sign Convention:** Always negative or zero. A non-zero value always represents a penalty (never a bonus).

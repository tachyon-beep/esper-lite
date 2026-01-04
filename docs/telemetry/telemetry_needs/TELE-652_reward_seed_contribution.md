# Telemetry Record: [TELE-652] Reward Seed Contribution

> **Status:** `[x] Planned` `[x] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-652` |
| **Name** | Reward Seed Contribution |
| **Category** | `reward` |
| **Priority** | `P2-important` |

## 2. Purpose

### What question does this answer?

> "What percentage of accuracy improvement is causally attributable to the active seed?"

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
| **Units** | percentage points (%) |
| **Range** | `[-inf, +inf]` — typically -10% to +10% |
| **Precision** | 1 decimal place for display |
| **Default** | `0.0` |

### Semantic Meaning

> Seed contribution measures the causal impact of the active seed on accuracy through counterfactual validation.
>
> **Computed as:** `real_accuracy - baseline_accuracy` where:
> - `real_accuracy` = accuracy with seed blended at current alpha
> - `baseline_accuracy` = accuracy with seed disabled (alpha=0)
>
> - **Positive values:** Seed is helping — it causally improves accuracy
> - **Negative values:** Seed is hurting — it causally degrades accuracy (toxic seed)
> - **Zero:** Seed has no measurable impact (possibly alpha=0 or pre-blending)
>
> This is the PRIMARY signal for contribution-based reward functions, replacing heuristic accuracy-delta approaches.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `value > 0` | Seed is contributing positively |
| **Neutral** | `value == 0` | No measurable seed impact |
| **Warning** | `value < 0` | Seed may be interfering |

**Display Color Logic:** Green if > 0, red if < 0 (see `_format_seed_delta()`)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Counterfactual validation in reward computation |
| **File** | `/home/john/esper-lite/src/esper/simic/rewards/rewards.py` |
| **Function/Method** | `compute_contribution_reward()` |
| **Line(s)** | 430-431, 531 |

```python
def compute_contribution_reward(
    action: LifecycleOp,
    seed_contribution: float | None,  # Counterfactual delta (real_acc - baseline_acc)
    ...
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Included in RewardComponentsTelemetry dataclass | `simic/rewards/reward_telemetry.py` (line 24) |
| **2. Collection** | Passed via `emit_last_action()` in training loop | `simic/telemetry/emitters.py` (line 278) |
| **3. Aggregation** | Extracted in aggregator (not currently wired to schema) | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Available via `env.reward_components.seed_contribution` | `karn/sanctum/schema.py` (line 1112) |

```
[counterfactual_validation()]
  --seed_contribution-->
  [RewardComponentsTelemetry.seed_contribution]
  --emit_last_action(reward_components=...)-->
  [LastActionPayload.reward_components]
  --SanctumAggregator.handle_last_action()-->
  [EnvState.reward_components.seed_contribution]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `RewardComponents` |
| **Field** | `seed_contribution` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].reward_components.seed_contribution` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1112 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvOverview | `widgets/env_overview.py` (lines 683-696) | Displayed in "Seed Δ" column with green/red color coding |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — Counterfactual validation computes `seed_contribution`
- [x] **Transport works** — Value flows through RewardComponentsTelemetry -> LastActionPayload
- [x] **Schema field exists** — `RewardComponents.seed_contribution: float = 0.0` at line 1112
- [x] **Default is correct** — `0.0` is appropriate when no counterfactual available
- [x] **Consumer reads it** — EnvOverview `_format_seed_delta()` accesses `env.reward_components.seed_contribution`
- [x] **Display is correct** — Rendered with `+.1f%` format, green if > 0, red if < 0

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_rewards.py` | Contribution reward tests | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_last_action_populates_fields` | `[ ]` |
| Integration (end-to-end) | `tests/integration/test_reward_telemetry_flow.py` | Telemetry flow integration | `[ ]` |
| Widget (EnvOverview) | `tests/karn/sanctum/test_env_overview.py` | Seed delta rendering | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI (opens automatically)
3. Observe EnvOverview table — "Seed Δ" column should show seed contribution percentage
4. Verify format: "+X.X%" with green for positive, red for negative
5. Note: Column shows "─" when no seed is active or alpha=0

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Counterfactual validation | computation | Requires baseline model forward pass |
| Active seed with alpha > 0 | state | Only computed during BLENDING/HOLDING stages |
| Validation accuracy | metric | Requires current validation accuracy |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EnvOverview Seed Δ column | display | Primary consumer (when seed_contribution != 0) |
| Reward attribution | computation | Used in bounded_attribution calculation |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** `seed_contribution` is the raw counterfactual delta before any bounding or attribution logic is applied. It represents "what did the seed actually do to accuracy?"
>
> **Relationship to bounded_attribution:** When `seed_contribution` is non-zero, it feeds into the `bounded_attribution` calculation which applies geometric mean capping and anti-gaming protections. The display logic in EnvOverview prefers `seed_contribution` when available, falling back to `bounded_attribution` when `seed_contribution` is 0.
>
> **Display Fallback:** The `_format_seed_delta()` function checks `seed_contribution` first. If it's 0 or unavailable, it displays `bounded_attribution` instead. This provides meaningful output even when counterfactual validation isn't running.
>
> **Counterfactual Dependency:** This metric requires counterfactual validation infrastructure. If counterfactual is disabled or unavailable, this will be 0 and the display will fall back to `bounded_attribution`.

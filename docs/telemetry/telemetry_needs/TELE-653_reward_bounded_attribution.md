# Telemetry Record: [TELE-653] Reward Bounded Attribution

> **Status:** `[x] Planned` `[x] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-653` |
| **Name** | Reward Bounded Attribution |
| **Category** | `reward` |
| **Priority** | `P2-important` |

## 2. Purpose

### What question does this answer?

> "What is the bounded, anti-gaming-protected attribution reward for the active seed?"

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
| **Units** | reward units (bounded attribution score) |
| **Range** | `[-inf, +inf]` — typically -1.0 to +1.0 after bounding |
| **Precision** | 2 decimal places for display |
| **Default** | `0.0` |

### Semantic Meaning

> Bounded attribution is the contribution-primary reward signal with anti-gaming protections applied.
>
> **Computed from `seed_contribution` with:**
> - Geometric mean capping (prevents credit for more than the seed contributed)
> - Attribution discount (sigmoid penalty for negative total_improvement history)
> - Ratio penalty protection (guards against ransomware-like gaming)
>
> - **Positive values:** Seed is genuinely contributing value
> - **Negative values:** Seed is causing harm (penalized)
> - **Zero:** No measurable impact or pre-blending stage
>
> This is the PRIMARY signal used in the reward function after applying all bounding logic.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `value > 0` | Genuine positive contribution |
| **Neutral** | `value == 0` | No measurable impact |
| **Warning** | `value < 0` | Seed causing harm |

**Display Color Logic:** Green if > 0, red if < 0 (see `_format_seed_delta()`)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Attribution bounding logic in reward computation |
| **File** | `/home/john/esper-lite/src/esper/simic/rewards/rewards.py` |
| **Function/Method** | `compute_contribution_reward()` |
| **Line(s)** | 520-580 (bounded attribution calculation block) |

```python
bounded_attribution = 0.0
# ... attribution logic with geometric mean capping and anti-gaming protections
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Included in RewardComponentsTelemetry dataclass | `simic/rewards/reward_telemetry.py` (line 25) |
| **2. Collection** | Passed via `emit_last_action()` in training loop | `simic/telemetry/emitters.py` (line 278) |
| **3. Aggregation** | Extracted in aggregator from LastActionPayload | `karn/sanctum/aggregator.py` (lines 1462-1463) |
| **4. Delivery** | Written to `env.reward_components.bounded_attribution` | `karn/sanctum/schema.py` (line 1111) |

```
[compute_contribution_reward()]
  --bounded_attribution-->
  [RewardComponentsTelemetry.bounded_attribution]
  --emit_last_action(reward_components=...)-->
  [LastActionPayload.reward_components]
  --SanctumAggregator.handle_last_action()-->
  [EnvState.reward_components.bounded_attribution]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `RewardComponents` |
| **Field** | `bounded_attribution` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].reward_components.bounded_attribution` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1111 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvOverview | `widgets/env_overview.py` (lines 683-696) | Displayed in "Seed Δ" column as fallback when seed_contribution is 0 |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — Attribution bounding logic computes `bounded_attribution`
- [x] **Transport works** — Value flows through RewardComponentsTelemetry -> LastActionPayload -> aggregator
- [x] **Schema field exists** — `RewardComponents.bounded_attribution: float = 0.0` at line 1111
- [x] **Default is correct** — `0.0` is appropriate when no attribution computed
- [x] **Consumer reads it** — EnvOverview `_format_seed_delta()` accesses as fallback
- [x] **Display is correct** — Rendered with `+.2f` format, green if > 0, red if < 0

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_rewards.py` | Attribution bounding tests | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_last_action_populates_fields` | `[ ]` |
| Integration (end-to-end) | `tests/integration/test_reward_telemetry_flow.py` | Telemetry flow integration | `[ ]` |
| Widget (EnvOverview) | `tests/karn/sanctum/test_env_overview.py` | Seed delta rendering | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI (opens automatically)
3. Observe EnvOverview table — "Seed Δ" column shows this when seed_contribution is 0
4. Verify format: "+X.XX" with green for positive, red for negative
5. Compare with seed_contribution values to see bounding effects

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| seed_contribution | computation | Raw counterfactual delta (input to bounding) |
| total_improvement history | state | Used for attribution discount calculation |
| Configuration | config | Anti-gaming settings control bounding behavior |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EnvOverview Seed Δ column | display | Fallback when seed_contribution is 0 |
| Total reward | computation | Contributes to final reward signal |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** `bounded_attribution` applies bounding and anti-gaming protections to the raw `seed_contribution`. This prevents the agent from gaming the reward by:
> - Artificially inflating seed contribution claims
> - Taking credit for host-driven improvements
> - Ransomware-like behavior (hurting then helping)
>
> **Relationship to seed_contribution:** `seed_contribution` is the raw counterfactual delta. `bounded_attribution` is `seed_contribution` after geometric mean capping, attribution discount, and ratio penalty protections.
>
> **Display Logic:** The `_format_seed_delta()` function in EnvOverview checks `seed_contribution` first. If `seed_contribution` is 0 (e.g., LOSS reward family, pre-blending), it falls back to displaying `bounded_attribution`. This ensures the Seed Δ column always shows meaningful data when available.
>
> **LOSS Family Note:** For the LOSS reward family, `bounded_attribution` may be None (not computed) as it uses a different reward signal. The aggregator handles this by leaving the field at its default value.

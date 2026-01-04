# Telemetry Record: [TELE-660] Reward Scaffold Count

> **Status:** `[x] Planned` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-660` |
| **Name** | Reward Scaffold Count |
| **Category** | `reward` |
| **Priority** | `P3-nice-to-have` |

## 2. Purpose

### What question does this answer?

> "How many scaffold seeds contributed to this hindsight credit?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

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
| **Type** | `int` |
| **Units** | count (number of scaffold seeds) |
| **Range** | `[0, +inf]` — always non-negative |
| **Precision** | integer (no decimal places) |
| **Default** | `0` |

### Semantic Meaning

> Scaffold count is a debugging/analysis field that tracks how many scaffold seeds contributed to a hindsight credit assignment.
>
> When a beneficiary seed fossilizes, it may have received scaffolding support from multiple other seeds. The scaffold count tells operators how many distinct scaffolds contributed to the beneficiary's success.
>
> - **Zero:** No scaffolding relationship (standalone fossilization)
> - **Positive:** Number of distinct scaffold contributors
> - **Higher values:** More cooperative scaffolding ecosystem
>
> This is displayed alongside hindsight_credit as "(Nx, ...)" where N is the scaffold count.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `value >= 0` | Normal (any count is valid) |
| **Info** | `value > 0` | Active scaffolding cooperation |
| **Note** | `value > 3` | Dense scaffolding network |

**Display Color Logic:** Displayed as metadata alongside hindsight_credit (blue styling inherits from parent)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Scaffold count from hindsight credit computation |
| **File** | `/home/john/esper-lite/src/esper/simic/rewards/rewards.py` |
| **Function/Method** | `compute_scaffold_hindsight_credit()` |
| **Line(s)** | 1247-1320 |

```python
# Count is derived from scaffold registry during hindsight credit computation
# Stored in RewardComponentsTelemetry.num_fossilized_seeds (mapped to scaffold_count)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Stored in `RewardComponentsTelemetry.num_fossilized_seeds` | `simic/rewards/reward_telemetry.py` (line 44) |
| **2. Collection** | Passed via `emit_last_action()` in training loop | `simic/telemetry/emitters.py` |
| **3. Aggregation** | Extracted in aggregator from LastActionPayload | `karn/sanctum/aggregator.py` (line 1475) |
| **4. Delivery** | Written to `env.reward_components.scaffold_count` | `karn/sanctum/schema.py` (line 1129) |

```
[compute_scaffold_hindsight_credit()]
  --components.num_fossilized_seeds-->
  [RewardComponentsTelemetry]
  --emit_last_action(reward_components=...)-->
  [LastActionPayload.reward_components]
  --SanctumAggregator.handle_last_action()-->
  [EnvState.reward_components.scaffold_count]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `RewardComponents` |
| **Field** | `scaffold_count` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].reward_components.scaffold_count` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1129 |
| **Default Value** | `0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvDetailScreen | `widgets/env_detail_screen.py` (lines 615-616) | Displayed alongside hindsight_credit as "(Nx, ...)" |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — Scaffold count computed during hindsight credit calculation
- [x] **Transport works** — Value flows via `num_fossilized_seeds` -> aggregator -> `scaffold_count`
- [x] **Schema field exists** — `RewardComponents.scaffold_count: int = 0` at line 1129
- [x] **Default is correct** — `0` is appropriate default (no scaffolds)
- [x] **Consumer reads it** — EnvDetailScreen accesses `rc.scaffold_count` at line 615
- [x] **Display is correct** — Rendered as "(Nx, ...)" alongside hindsight_credit

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_rewards.py` | Scaffold count tests | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_last_action_populates_fields` | `[ ]` |
| Integration (end-to-end) | `tests/integration/test_reward_telemetry_flow.py` | Telemetry flow integration | `[ ]` |
| Widget (EnvDetailScreen) | `tests/karn/sanctum/test_env_detail_screen.py` | Scaffold count display | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI (opens automatically)
3. Navigate to EnvDetailScreen for an environment with hindsight credit
4. Observe "Credits" row — should show "(Nx, Y.Ye)" where N is scaffold count
5. Verify that scaffold_count > 0 whenever hindsight_credit > 0
6. After training, query telemetry: `SELECT scaffold_count, hindsight_credit FROM rewards WHERE hindsight_credit != 0 LIMIT 10`

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Scaffold registry | state | Tracks scaffold relationships |
| Hindsight credit computation | function | Count is derived during credit calculation |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EnvDetailScreen Credits row | display | Shown as "(Nx, ...)" metadata |
| hindsight_credit (TELE-659) | related | Always displayed together |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** Scaffold count is a debugging/analysis field that helps operators understand the scaffolding dynamics. It's not a reward component itself but provides context for interpreting hindsight_credit.
>
> **Field Mapping:** Note that the telemetry field is `num_fossilized_seeds` but the schema field is `scaffold_count`. The aggregator maps between them at line 1475.
>
> **Display Format:** The format "(2x, 3.5e)" means 2 scaffold seeds contributed, with an average delay of 3.5 epochs. This compact format provides scaffolding context without cluttering the display.
>
> **Relationship to hindsight_credit:** scaffold_count > 0 implies hindsight_credit > 0 (if scaffolds contributed, credit was assigned). However, the inverse is not guaranteed due to edge cases in credit calculation.

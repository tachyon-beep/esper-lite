# Telemetry Record: [TELE-659] Reward Hindsight Credit

> **Status:** `[x] Planned` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-659` |
| **Name** | Reward Hindsight Credit |
| **Category** | `reward` |
| **Priority** | `P2-important` |

## 2. Purpose

### What question does this answer?

> "How much retroactive credit did this seed receive for scaffolding contributions when a beneficiary seed successfully fossilized?"

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
| **Type** | `float` |
| **Units** | reward units (bonus) |
| **Range** | `[0, +inf]` — always positive or zero (bonus) |
| **Precision** | 3 decimal places for display |
| **Default** | `0.0` |

### Semantic Meaning

> Hindsight credit is a retroactive bonus applied when a beneficiary seed successfully fossilizes. It rewards scaffold seeds that contributed to the fossilized seed's success, even if those scaffolds were pruned earlier.
>
> This implements the "credit assignment" problem solution: seeds that help other seeds succeed deserve credit for that contribution, even if the benefit wasn't immediately apparent. The credit is applied "in hindsight" when the beneficiary reaches a terminal success state.
>
> - **Zero:** No scaffold contribution recognized (seed didn't help others fossilize)
> - **Positive:** Retroactive credit for scaffolding contribution
> - **Higher values:** More significant scaffolding contribution or multiple beneficiaries
>
> The display includes metadata: "(Nx, Y.Ye)" where N is scaffold_count and Y.Y is avg_scaffold_delay (epochs since scaffolding began).

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `value >= 0` | Normal credit assignment (including zero) |
| **Note** | `value > 0.2` | Significant scaffolding contribution detected |
| **Info** | Frequent non-zero | Healthy ecosystem with scaffold cooperation |

**Display Color Logic:** Blue styling (bonus category)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Scaffold hindsight credit computation |
| **File** | `/home/john/esper-lite/src/esper/simic/rewards/rewards.py` |
| **Function/Method** | `compute_scaffold_hindsight_credit()` |
| **Line(s)** | 1247-1320 |

```python
def compute_scaffold_hindsight_credit(
    seed_info: SeedInfo,
    scaffold_registry: ScaffoldRegistry,
    ...
) -> float:
    """Compute retroactive credit for scaffolding contributions.

    When a beneficiary seed fossilizes, scaffold seeds that contributed
    to its success receive hindsight credit proportional to their contribution.
    """
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Stored in `RewardComponentsTelemetry.hindsight_credit` | `simic/rewards/reward_telemetry.py` (line 43) |
| **2. Collection** | Passed via `emit_last_action()` in training loop | `simic/telemetry/emitters.py` |
| **3. Aggregation** | Extracted in aggregator from LastActionPayload | `karn/sanctum/aggregator.py` (line 1468) |
| **4. Delivery** | Written to `env.reward_components.hindsight_credit` | `karn/sanctum/schema.py` (line 1128) |

```
[compute_scaffold_hindsight_credit()]
  --components.hindsight_credit-->
  [RewardComponentsTelemetry]
  --emit_last_action(reward_components=...)-->
  [LastActionPayload.reward_components]
  --SanctumAggregator.handle_last_action()-->
  [EnvState.reward_components.hindsight_credit]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `RewardComponents` |
| **Field** | `hindsight_credit` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].reward_components.hindsight_credit` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1128 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvDetailScreen | `widgets/env_detail_screen.py` (lines 613-619) | Displayed in "Credits" row as "Hind: +0.12 (2x, 3.5e)" with blue styling and scaffold metadata |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `compute_scaffold_hindsight_credit()` computes credit, stored in components
- [x] **Transport works** — Value flows through RewardComponentsTelemetry -> LastActionPayload -> aggregator
- [x] **Schema field exists** — `RewardComponents.hindsight_credit: float = 0.0` at line 1128
- [x] **Default is correct** — `0.0` is appropriate default (no credit when not triggered)
- [x] **Consumer reads it** — EnvDetailScreen accesses `rc.hindsight_credit` at line 613
- [x] **Display is correct** — Rendered with `+.3f` format, blue styling, includes scaffold metadata

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_rewards.py` | Hindsight credit tests | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_last_action_populates_fields` | `[ ]` |
| Integration (end-to-end) | `tests/integration/test_reward_telemetry_flow.py` | Telemetry flow integration | `[ ]` |
| Widget (EnvDetailScreen) | `tests/karn/sanctum/test_env_detail_screen.py` | Hindsight credit rendering | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI (opens automatically)
3. Navigate to EnvDetailScreen for an environment where fossilization occurred
4. Observe "Credits" row — should show "Hind: +X.XXX (Nx, Y.Ye)" when scaffold credit is assigned
5. Verify the metadata shows scaffold count and average delay
6. After training, query telemetry: `SELECT hindsight_credit FROM rewards WHERE hindsight_credit != 0 LIMIT 10`

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Scaffold registry | state | Tracks which seeds scaffolded which beneficiaries |
| Fossilization events | event | Credit is assigned when beneficiary fossilizes |
| Scaffold contribution metrics | metric | Determines credit magnitude |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EnvDetailScreen Credits row | display | Displayed as "Hind: +X.XXX (Nx, Y.Ye)" |
| scaffold_count (TELE-660) | related | Number of scaffolds shown in display |
| avg_scaffold_delay (TELE-661) | related | Delay epochs shown in display |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** Hindsight credit solves the temporal credit assignment problem. A scaffold seed may be pruned long before its beneficiary fossilizes, but it still deserves credit for enabling that success. This retroactive credit encourages cooperative behavior.
>
> **Phase 3.2 Feature:** Hindsight credit was added in Phase 3.2 of the reward engineering roadmap as part of the scaffold contribution bonus system.
>
> **Display Format:** The format "Hind: +0.12 (2x, 3.5e)" means: hindsight credit of +0.12 from 2 scaffold seeds that began scaffolding an average of 3.5 epochs ago. This helps operators understand the scaffolding dynamics.
>
> **Sign Convention:** Always positive or zero. A non-zero value always represents a bonus (never a penalty). The "hindsight" is always positive — we reward good scaffolding, we don't penalize failed scaffolding.

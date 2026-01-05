# Telemetry Record: [TELE-631] Environment Reward Mode

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-631` |
| **Name** | Environment Reward Mode |
| **Category** | `env` |
| **Priority** | `P2-medium` |

## 2. Purpose

### What question does this answer?

> "Which A/B test cohort or reward shaping strategy is this environment using?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

### When is this information needed?

- [x] Real-time (every batch/epoch)
- [ ] Periodic (every N episodes)
- [x] On-demand (when investigating issues)
- [x] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `str | None` |
| **Units** | categorical identifier |
| **Range** | `"shaped"`, `"simplified"`, `"sparse"`, or `None` |
| **Precision** | N/A |
| **Default** | `None` |

### Semantic Meaning

> Reward mode identifies the A/B test cohort for this environment. Different reward shaping strategies can be compared side-by-side within the same training run:
>
> - **shaped:** Full reward shaping with PBRS, compute rent, stage bonuses
> - **simplified:** Reduced reward signal with fewer components
> - **sparse:** Minimal reward signal (e.g., terminal accuracy only)
> - **None:** No A/B testing active; using default reward configuration
>
> This enables visual comparison of different reward strategies in the EnvOverview table via colored pips next to env IDs.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| N/A | N/A | This is an informational field, not a health metric |

**Threshold Source:** N/A (categorical field)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | TRAINING_STARTED payload configuration |
| **File** | `/home/john/esper-lite/src/esper/leyline/telemetry.py` |
| **Function/Method** | `TrainingStartedPayload.reward_mode` |
| **Line(s)** | (varies) |

```python
@dataclass
class TrainingStartedPayload:
    ...
    reward_mode: str = ""  # A/B test cohort identifier
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Set in TrainingStartedPayload at training init | `leyline/telemetry.py` |
| **2. Collection** | Captured by aggregator during TRAINING_STARTED | `aggregator.py` (line 638) |
| **3. Aggregation** | Stored in aggregator `_reward_mode`, propagated to all EnvState instances | `aggregator.py` (lines 1686-1689) |
| **4. Delivery** | Available at `snapshot.envs[env_id].reward_mode` | `schema.py` (line 542) |

```
[TrainingStartedPayload.reward_mode]
  --TRAINING_STARTED-->
  [SanctumAggregator._handle_training_started()]
  --self._reward_mode-->
  [SanctumAggregator._ensure_env()]
  --reward_mode=self._reward_mode-->
  [EnvState.reward_mode]
  --snapshot-->
  [SanctumSnapshot.envs[env_id].reward_mode]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `EnvState` |
| **Field** | `reward_mode` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].reward_mode` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 542 |
| **Default Value** | `None` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvOverview | `widgets/env_overview.py` (lines 29-33, 491-524) | Colored pip next to env ID: blue=shaped, yellow=simplified, white=sparse |
| BestRunRecord | `schema.py` (line 1270) | Stored for historical A/B cohort tracking |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — TrainingStartedPayload includes reward_mode field
- [x] **Transport works** — Aggregator captures reward_mode and propagates to EnvState
- [x] **Schema field exists** — `EnvState.reward_mode: str | None = None` at line 542
- [x] **Default is correct** — `None` indicates no A/B testing active
- [x] **Consumer reads it** — EnvOverview._format_env_id() reads `env.reward_mode`
- [x] **Display is correct** — Colored pip renders based on reward_mode value
- [x] **Thresholds applied** — N/A (categorical field)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| E2E (telemetry) | `tests/telemetry/test_tele_env_state.py` | `TestTELE631EnvRewardMode` (7 tests) | `[x]` |
| Unit (emitter) | `tests/leyline/test_telemetry.py` | `test_training_started_payload` | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_training_started_sets_reward_mode` | `[ ]` |
| Widget (EnvOverview) | `tests/karn/sanctum/widgets/test_env_overview.py` | A/B pip rendering | `[ ]` |

### Manual Verification Steps

1. Start training with a non-default reward mode (e.g. dual-policy A/B): `PYTHONPATH=src uv run python -m esper.scripts.train ppo --dual-ab shaped-vs-simplified`
2. Launch Sanctum TUI
3. Observe EnvOverview Env column — colored pips should appear next to env IDs
4. Verify pip colors match reward mode (blue=shaped, yellow=simplified, white=sparse)

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| TRAINING_STARTED event | event | Provides initial reward_mode configuration |
| A/B testing configuration | config | Must be enabled in training CLI |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EnvOverview Env column pip | display | Colored cohort indicator |
| BestRunRecord | data | Historical cohort tracking for leaderboard |
| Post-hoc analysis | analysis | Comparing cohort performance |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** Reward mode is set once at training start and remains constant for the env's lifetime. This enables clean A/B comparison without mid-experiment switching.
>
> **Visual Design:** Colored pips (circles) provide quick visual cohort identification without cluttering the env ID column. Colors are chosen for colorblind accessibility (blue/yellow/white rather than red/green).
>
> **Wiring Status:** Fully wired and operational. Reward mode flows from CLI config through TRAINING_STARTED event to all EnvState instances.

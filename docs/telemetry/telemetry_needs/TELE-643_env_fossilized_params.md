# Telemetry Record: [TELE-643] Environment Fossilized Parameters

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-643` |
| **Name** | Environment Fossilized Parameters |
| **Category** | `env` |
| **Priority** | `P1-high` |

## 2. Purpose

### What question does this answer?

> "How many additional parameters have been permanently added to this environment through fossilized seeds?"

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
| **Type** | `int` |
| **Units** | parameter count |
| **Range** | `[0, param_budget * max_seeds]` |
| **Precision** | Integer |
| **Default** | `0` |

### Semantic Meaning

> Fossilized parameters tracks the cumulative parameter count added to the host model through fossilized seeds:
>
> - Incremented when SEED_FOSSILIZED event adds params_added
> - Represents "model growth" from successful mutations
> - Used as numerator component in growth_ratio calculation
> - Reset at episode start (BATCH_EPOCH_COMPLETED)
>
> This is distinct from seed_params (per-seed) which tracks individual seed sizes.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | Low fossilized_params relative to host | Minimal model bloat |
| **Warning** | fossilized_params approaching host_params | Significant growth |
| **Critical** | fossilized_params >> host_params | Model size explosion |

**Threshold Source:** Indirectly via growth_ratio thresholds in `_format_growth_ratio()`

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | SEED_FOSSILIZED telemetry event |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `SanctumAggregator._handle_seed_event()` |
| **Line(s)** | 1080 |

```python
elif event_type == "SEED_FOSSILIZED" and isinstance(event.data, SeedFossilizedPayload):
    ...
    env.fossilized_params += fossilized_payload.params_added
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Seed fossilization emits SEED_FOSSILIZED with params_added | `simic/seed.py` |
| **2. Collection** | Aggregator receives event | `aggregator.py` (line 1055) |
| **3. Aggregation** | Handler adds params_added to env.fossilized_params | `aggregator.py` (line 1080) |
| **4. Delivery** | Available at `snapshot.envs[env_id].fossilized_params` | `schema.py` (line 463) |

```
[SeedFossilizedPayload.params_added]
  --SEED_FOSSILIZED-->
  [SanctumAggregator._handle_seed_event()]
  --env.fossilized_params += params_added-->
  [EnvState.fossilized_params]
  --snapshot-->
  [SanctumSnapshot.envs[env_id].fossilized_params]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `EnvState` |
| **Field** | `fossilized_params` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].fossilized_params` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 463 |
| **Default Value** | `0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| growth_ratio property | `schema.py` (lines 569-578) | Numerator component for ratio |
| EnvOverview Growth column | `widgets/env_overview.py` (lines 547-567) | Displayed via growth_ratio |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — SEED_FOSSILIZED event includes params_added field
- [x] **Transport works** — Aggregator accumulates params_added in _handle_seed_event()
- [x] **Schema field exists** — `EnvState.fossilized_params: int = 0` at line 463
- [x] **Default is correct** — `0` at start of each episode
- [x] **Consumer reads it** — growth_ratio property uses fossilized_params in calculation
- [x] **Display is correct** — Shown indirectly via growth_ratio column
- [x] **Thresholds applied** — Via growth_ratio thresholds

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_seed_fossilized_accumulates_params` | `[ ]` |
| Unit (schema) | `tests/karn/sanctum/test_schema.py` | `test_growth_ratio_with_fossilized_params` | `[ ]` |
| Widget (EnvOverview) | `tests/karn/sanctum/widgets/test_env_overview.py` | Growth ratio display | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI
3. Wait for seeds to fossilize (check slot columns for "Foss" status)
4. Observe Growth column — should increase from 1.0x when fossilization occurs
5. Verify reset to 1.0x when new episode starts

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| SEED_FOSSILIZED events | event | Provides params_added |
| Seed fossilization logic | system | Must track params_added accurately |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| growth_ratio (TELE-644) | derived | Uses fossilized_params in numerator |
| Model size tracking | analysis | Total model size = host_params + fossilized_params |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** fossilized_params is reset at each BATCH_EPOCH_COMPLETED (aggregator line 1292). This provides a per-episode view of fossilization impact.
>
> **Accuracy Fix:** This field was added specifically to fix the scoreboard param count display (see schema.py comment at line 462).
>
> **Related Fields:** Used with host_params (TELE-642) to compute growth_ratio (TELE-644).
>
> **Wiring Status:** Fully wired via SEED_FOSSILIZED events.

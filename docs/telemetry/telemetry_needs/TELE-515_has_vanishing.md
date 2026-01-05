# Telemetry Record: [TELE-515] Seed Has Vanishing Gradients

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-515` |
| **Name** | Seed Has Vanishing Gradients |
| **Category** | `seed` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Is this seed experiencing vanishing gradients that may impair its ability to learn?"

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
| **Type** | `bool` |
| **Units** | N/A (boolean flag) |
| **Range** | `{True, False}` |
| **Precision** | N/A |
| **Default** | `False` |

### Semantic Meaning

> Indicates whether any of the seed module's parameter gradients fall below the vanishing threshold during the most recent backward pass. The gradient collector computes this as:
>
> `has_vanishing = n_vanishing > 0`
>
> where `n_vanishing` counts gradients with norm below `DEFAULT_VANISHING_THRESHOLD = 1e-7`.
>
> When `True`, the seed is receiving very weak learning signal, which may indicate:
> - The seed is not connected to the loss path effectively
> - Gradient scaling issues in deep networks
> - The seed has converged or is dormant
>
> **PPO Context:** Collection happens BEFORE gradient clipping. The threshold is PPO-tuned (see `simic/telemetry/gradient_collector.py`).

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `has_vanishing == False` | All gradients above vanishing threshold |
| **Warning** | `has_vanishing == True` | Some gradients below 1e-7 - seed may not be learning effectively |
| **Critical** | `has_vanishing == True` + `gradient_health < 0.5` | Significant gradient issues requiring attention |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Gradient collector after backward pass |
| **File** | `/home/john/esper-lite/src/esper/simic/telemetry/gradient_collector.py` |
| **Function/Method** | `materialize_grad_stats()` and `collect_seed_gradients()` |
| **Line(s)** | ~235-240 (collect_seed_gradients), ~358 (GradientHealthMetrics) |

```python
# From materialize_grad_stats():
return {
    'gradient_norm': gradient_norm,
    'gradient_health': health,
    'has_vanishing': n_vanishing > 0,  # Line ~238
    'has_exploding': n_exploding > 0,
}
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Gradient stats collected per seed after backward | `simic/telemetry/gradient_collector.py` |
| **2. Sync** | `SlotState.sync_telemetry(has_vanishing=...)` | `kasmina/slot.py:483-484` |
| **3. Payload** | Included in lifecycle event payloads | `leyline/telemetry.py` (SeedTelemetry, payloads) |
| **4. Aggregation** | Aggregator writes to `SeedState.has_vanishing` | `karn/sanctum/aggregator.py:736,1009,1047` |
| **5. Schema** | `SeedState.has_vanishing: bool = False` | `karn/sanctum/schema.py:400` |

```
[GradientCollector] --collect()--> [SlotState.telemetry] --sync_telemetry()--> [SeedTelemetry]
    --> [Payload(has_vanishing)] --emit()--> [Aggregator] --> [SeedState.has_vanishing]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SeedState` |
| **Field** | `has_vanishing` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].seeds[slot_id].has_vanishing` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 400 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvOverview | `widgets/env_overview.py:742` | Yellow down arrow indicator (`[yellow]▼[/yellow]`) |
| EnvDetailScreen | `widgets/env_detail_screen.py:156` | "VANISHING" label with bold yellow style |
| AnomalyStrip | `widgets/anomaly_strip.py:95` | Counted as gradient issue for alert aggregation |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `SeedGradientCollector.collect()` computes `has_vanishing`
- [x] **Transport works** — `SlotState.sync_telemetry()` propagates to `SeedTelemetry.has_vanishing`
- [x] **Schema field exists** — `SeedState.has_vanishing: bool = False`
- [x] **Default is correct** — `False` appropriate (no vanishing until detected)
- [x] **Consumer reads it** — EnvOverview, EnvDetailScreen, AnomalyStrip all read field
- [x] **Display is correct** — Yellow down arrow (EnvOverview), "VANISHING" label (EnvDetailScreen)
- [x] **Thresholds applied** — Color coding in widgets matches spec

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_gradient_collector.py` | `test_vanishing_gradients_detected` | `[x]` |
| Unit (slot sync) | `tests/kasmina/test_slot_telemetry.py` | Tests with `has_vanishing=True` | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_backend.py` | `has_vanishing` assertions | `[x]` |
| Unit (schema) | `tests/karn/sanctum/test_schema.py` | `has_vanishing=True` construction | `[x]` |
| Unit (EnvOverview) | `tests/karn/sanctum/test_env_overview.py` | `has_vanishing=True` indicator | `[x]` |
| Unit (EnvDetailScreen) | `tests/karn/sanctum/test_env_detail_screen.py` | `has_vanishing=True` display | `[x]` |
| Unit (AnomalyStrip) | `tests/karn/sanctum/test_anomaly_strip.py` | Gradient issue counting | `[x]` |
| Property (gradients) | `tests/simic/properties/test_gradient_properties.py` | Multiple vanishing assertions | `[x]` |
| Integration (PPO) | `tests/simic/test_ppo.py` | Feature extraction `has_vanishing` | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI
3. Observe EnvOverview slot cells
4. When vanishing gradients occur, verify yellow `▼` indicator appears
5. Click on seed row to expand EnvDetailScreen
6. Verify "VANISHING" appears in bold yellow in gradient health line
7. Check AnomalyStrip gradient issue count increases

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Seed backward pass | event | Gradients must exist after loss.backward() |
| `DEFAULT_VANISHING_THRESHOLD` | constant | 1e-7 threshold from `gradient_collector.py` |
| `TELE-516` has_exploding | telemetry | Computed in same pass, mutually informative |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `TELE-517` gradient_health | telemetry | Health score penalized by vanishing ratio |
| AnomalyStrip alert count | display | Aggregates gradient issues across all seeds |
| Tamiyo observations (V3) | feature | Embedded as binary feature in policy observations |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2024-11-29 | Initial | Created with seed telemetry implementation |
| 2025-01-03 | Audit | Verified wiring in telemetry audit |

---

## 8. Notes

> **Design Decision:** `has_vanishing` is a binary flag rather than a continuous metric (like `vanishing_ratio`) to simplify UI display and alerting. The continuous information is available via `gradient_health` score.
>
> **PPO Calibration:** The vanishing threshold (1e-7) is specifically tuned for PPO training. It represents gradients too small to provide meaningful learning signal, not just "small" gradients.
>
> **Multi-Event Emission:** This field is included in multiple event types:
> - `EPOCH_COMPLETED` (per-seed telemetry dict)
> - `SEED_GERMINATED` (SeedGerminatedPayload)
> - `SEED_STAGE_CHANGED` (SeedStageChangedPayload)
>
> **Feature Encoding:** For policy observations (Tamiyo V3), `has_vanishing` is encoded as 1.0 (True) or 0.0 (False) in the slot feature vector at index 25 (per `tamiyo/policy/features.py:250`).

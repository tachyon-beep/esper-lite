# Telemetry Record: [TELE-514] Seed Has Exploding Gradients

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-514` |
| **Name** | Seed Has Exploding Gradients |
| **Category** | `seed` |
| **Priority** | `P0-critical` |

## 2. Purpose

### What question does this answer?

> "Is this seed experiencing gradient explosion that could destabilize training or lead to NaN losses?"

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
| **Units** | N/A (binary flag) |
| **Range** | `{True, False}` |
| **Precision** | N/A |
| **Default** | `False` |

### Semantic Meaning

> Boolean flag indicating whether any gradient tensor in the seed module has a norm exceeding the exploding threshold. The exploding threshold is defined as `10.0 * DEFAULT_MAX_GRAD_NORM` (= 5.0 with default clip norm of 0.5).
>
> When `has_exploding=True`, the seed's gradients are so large they will be heavily clipped (scaled down 10x or more). This indicates:
> - Numerical instability in the seed's contribution to the loss
> - Potential for NaN propagation if the condition persists
> - The seed may be disrupting host training by dominating gradient updates
>
> Detection happens BEFORE gradient clipping, so "exploding" means "will be heavily clipped" rather than "has caused catastrophic failure".

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `has_exploding == False` | Gradients within normal operating range |
| **Warning** | N/A (binary flag) | N/A |
| **Critical** | `has_exploding == True` | Seed gradients exceed 10x clip norm - immediate attention required |

**Threshold Source:** `DEFAULT_EXPLODING_THRESHOLD = 10.0 * DEFAULT_MAX_GRAD_NORM` in `src/esper/simic/telemetry/gradient_collector.py:29`

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Gradient statistics collection after backward pass |
| **File** | `/home/john/esper-lite/src/esper/simic/telemetry/gradient_collector.py` |
| **Function/Method** | `SeedGradientCollector.collect_async()` and `materialize_grad_stats()` |
| **Line(s)** | ~135-183, ~186-240 |

```python
# In collect_async() - async tensor computation
'_n_exploding': (all_norms > self.exploding_threshold).sum(),

# In materialize_grad_stats() - final materialization
n_exploding = int(n_exploding_val.item())
return {
    ...
    'has_exploding': n_exploding > 0,
}
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Collected per-seed in training loop | `simic/training/vectorized.py:2494-2504` |
| **2. Collection** | Synced to `SeedState.telemetry.has_exploding` via `sync_telemetry()` | `kasmina/slot.py:485-486` |
| **3. Aggregation** | Included in slot reports via `get_slot_reports()` | `simic/telemetry/emitters.py:107` |
| **4. Delivery** | Written to `SeedState.has_exploding` via aggregator | `karn/sanctum/aggregator.py:737` |

```
[gradient_collector] --> [materialize_grad_stats()] --> [sync_telemetry()] -->
[slot_reports] --> [TelemetryEmitter.on_epoch_completed()] -->
[EPOCH_COMPLETED event] --> [SanctumAggregator._handle_epoch_completed()] -->
[SeedState.has_exploding]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SeedState` |
| **Field** | `has_exploding` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].seeds[slot_id].has_exploding` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 401 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvOverview | `widgets/env_overview.py:740-741` | Displays `[red]▲[/red]` indicator in seed cell |
| EnvDetailScreen | `widgets/env_detail_screen.py:154-155` | Shows `▲ EXPLODING` in bold red for seed detail |
| AnomalyStrip | `widgets/anomaly_strip.py:95-96` | Increments `gradient_issues` count for anomaly bar |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** - `SeedGradientCollector.collect_async()` computes and returns `has_exploding`
- [x] **Transport works** - Value flows through slot reports to aggregator
- [x] **Schema field exists** - `SeedState.has_exploding: bool = False`
- [x] **Default is correct** - `False` appropriate (healthy default)
- [x] **Consumer reads it** - All 3 widgets access `seed.has_exploding` directly
- [x] **Display is correct** - Red `▲` indicator renders in TUI
- [x] **Thresholds applied** - Binary flag; True = red alert

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_gradient_collector.py` | `test_detects_exploding_gradients` | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_backend.py` | Tests seed telemetry population | `[x]` |
| Unit (widget) | `tests/karn/sanctum/test_env_overview.py` | Tests `has_exploding=True` shows `▲` | `[x]` |
| Unit (widget) | `tests/karn/sanctum/test_env_detail_screen.py` | `has_exploding=True` test case | `[x]` |
| Unit (anomaly) | `tests/karn/sanctum/test_anomaly_strip.py` | Tests gradient issues counting | `[x]` |
| Property-based | `tests/simic/properties/test_gradient_properties.py` | Multiple property tests | `[x]` |
| Integration | `tests/kasmina/test_slot_telemetry.py` | Tests telemetry sync | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe EnvOverview seed cells for `▲` indicator
4. Click on an environment row to open EnvDetailScreen
5. Verify seed detail shows "Grad: ▲ EXPLODING" in bold red when flag is True
6. Observe AnomalyStrip shows gradient issues count when seeds have exploding gradients
7. To force explosion: modify seed learning rate or inject large gradients for testing

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Seed backward pass | computation | Requires gradients to exist on seed parameters |
| `DEFAULT_EXPLODING_THRESHOLD` | constant | 10x clip norm threshold from leyline |
| `sync_telemetry()` | method | Propagates gradient stats to telemetry dataclass |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `TELE-513` has_vanishing | telemetry | Companion flag - both inform gradient health |
| `GradientHealthMetrics.is_healthy()` | computation | Returns False if `has_exploding` is True |
| AnomalyStrip.gradient_issues | display | Counts seeds with exploding or vanishing gradients |
| Observation features | policy input | `has_exploding` is feature [13] in 26-dim SeedTelemetry vector |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2024-11-29 | Initial | Created with seed telemetry design |
| 2025-01-03 | Audit | Verified complete wiring in telemetry audit |

---

## 8. Notes

> **Design Decision:** The exploding threshold (10x clip norm = 5.0) was chosen based on PPO training dynamics. At 10x the clip norm, gradient direction is preserved but magnitude is scaled down 10x during clipping. This is informative (heavy clipping occurring) but not catastrophic. True explosions (100x+) indicate numerical instability.
>
> **Relationship to `has_vanishing`:** These flags are not mutually exclusive. A seed can have both vanishing and exploding gradients simultaneously if different layers have divergent gradient magnitudes (indicating severe layer imbalance).
>
> **Policy Feature:** This flag is encoded as a binary feature (0/1) at position [13] in the SeedTelemetry feature vector, allowing the policy to learn to avoid actions that lead to gradient explosion.
>
> **TUI Display Convention:** Exploding gradients use `▲` (upward triangle, red) while vanishing gradients use `▼` (downward triangle, yellow). This visual distinction helps operators quickly identify gradient health issues.

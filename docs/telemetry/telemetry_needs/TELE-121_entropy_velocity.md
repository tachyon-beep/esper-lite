# Telemetry Record: [TELE-121] Entropy Velocity

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-121` |
| **Name** | Entropy Velocity |
| **Category** | `policy` |
| **Priority** | `P0-critical` |

## 2. Purpose

### What question does this answer?

> "Is entropy declining rapidly? Will the policy collapse soon?"

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
| **Units** | entropy units per batch (nats per batch) |
| **Range** | `(-∞, ∞)` — typically in range `[-0.1, 0.1]` |
| **Precision** | 3 decimal places for display |
| **Default** | `0.0` |

### Semantic Meaning

> Entropy velocity measures the rate of change of policy entropy over time, computed as:
>
> velocity = d(entropy) / d(batch) = slope of entropy_history over last 10 batches
>
> Calculated using least-squares linear regression over the most recent 10 entropy samples.
> Negative velocity indicates entropy is declining (policy converging); positive velocity indicates entropy is increasing (policy exploring more).
> Magnitude indicates speed: velocity < -0.03 = rapid collapse risk; -0.03 to -0.01 = concerning; -0.01 to 0.01 = stable; > 0.01 = recovering.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `abs(velocity) < 0.005` | Entropy stable, no urgent concern |
| **Stable-Declining** | `-0.01 < velocity < -0.005` | Slow entropy decline, monitor |
| **Declining** | `-0.03 < velocity <= -0.01` | Entropy dropping, yellow warning [v] |
| **Critical** | `velocity <= -0.03` | Rapid entropy decline, red critical [vv] |
| **Recovering** | `velocity > 0.01` | Entropy increasing, green status [^] |

**Display Convention in HealthStatusPanel:**
- `[--]` (green) = `abs(velocity) < 0.005` (stable)
- `[v]` (yellow) = `-0.03 < velocity < -0.01` (warning)
- `[vv]` (red) = `velocity < -0.03` (critical)
- `[^]` (green) = `velocity > 0.01` (recovering)
- `[~]` (dim) = everything else (mixed/neutral)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Aggregator computation from entropy history |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `SanctumAggregator.handle_ppo_update()` |
| **Line(s)** | ~954-956 |

```python
# Compute entropy velocity from entropy_history
self._tamiyo.entropy_velocity = compute_entropy_velocity(
    self._tamiyo.entropy_history
)
```

**Note:** `entropy_velocity` is not directly emitted by PPO; rather, it is derived from the entropy values that PPO emits each update.

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Base Data (Entropy)** | `TelemetryEmitter.emit_ppo_update()` | `simic/telemetry/emitters.py` |
| **2. Collection** | Event payload with `entropy` field | `leyline/telemetry.py` |
| **3. Accumulation** | `SanctumAggregator.handle_ppo_update()` appends entropy to `entropy_history` | `karn/sanctum/aggregator.py` ~918 |
| **4. Computation** | `compute_entropy_velocity(entropy_history)` (numpy polyfit on last 10 values) | `karn/sanctum/schema.py` ~24-44 |
| **5. Delivery** | Written to `snapshot.tamiyo.entropy_velocity` | `karn/sanctum/schema.py` |

```
[PPOAgent] --emit_ppo_update(entropy)--> [Aggregator.entropy_history] --compute_entropy_velocity()--> [TamiyoState.entropy_velocity]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `entropy_velocity` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.entropy_velocity` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | ~974 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| HealthStatusPanel | `widgets/tamiyo_brain/health_status_panel.py` | Displayed in "Entropy D" row with velocity indicator and arrow |
| PPOLossesPanel | `widgets/tamiyo_brain/ppo_losses_panel.py` | Used to calculate collapse countdown: `batches_until_critical = distance / abs(velocity)` |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — PPO emits entropy every update; aggregator accumulates it
- [x] **Transport works** — Entropy values feed into `entropy_history` deque
- [x] **Computation exists** — `compute_entropy_velocity()` function defined in schema.py
- [x] **Schema field exists** — `TamiyoState.entropy_velocity: float = 0.0`
- [x] **Default is correct** — 0.0 appropriate before 5 entropy samples accumulated
- [x] **Computation triggers** — Called in `handle_ppo_update()` after entropy_history updated
- [x] **Consumer reads it** — Both widgets access `snapshot.tamiyo.entropy_velocity`
- [x] **Display is correct** — HealthStatusPanel renders velocity with proper arrows and coloring
- [x] **Thresholds applied** — HealthStatusPanel uses -0.03/-0.01/0.01/0.005 thresholds; PPOLossesPanel uses velocity for collapse countdown

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (computation) | `tests/karn/sanctum/test_entropy_prediction.py` | `test_stable_entropy_zero_velocity` | `[x]` |
| Unit (computation) | `tests/karn/sanctum/test_entropy_prediction.py` | `test_declining_entropy_negative_velocity` | `[x]` |
| Unit (computation) | `tests/karn/sanctum/test_entropy_prediction.py` | `test_rising_entropy_positive_velocity` | `[x]` |
| Unit (computation) | `tests/karn/sanctum/test_entropy_prediction.py` | `test_short_history_returns_zero` | `[x]` |
| Unit (computation) | `tests/karn/sanctum/test_entropy_prediction.py` | `test_noisy_declining_entropy` | `[x]` |
| Integration (display) | `tests/karn/sanctum/widgets/tamiyo_brain/test_ppo_losses_panel.py` | `test_ppo_losses_collapse_warning` | `[x]` |
| Integration (display) | `tests/karn/sanctum/widgets/tamiyo_brain/test_health_status_panel.py` | Various HealthStatusPanel tests | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe HealthStatusPanel "Entropy D" row
4. Verify entropy_velocity updates after each PPO batch with arrow indicator:
   - Stable entropy → `[--]` (green)
   - Declining entropy → `[v]` (yellow) or `[vv]` (red)
   - Recovering entropy → `[^]` (green)
5. Artificially create entropy collapse conditions (reduce entropy coefficient) to verify:
   - Critical threshold coloring (red) at velocity < -0.03
   - Warning coloring (yellow) at velocity < -0.01
   - PPOLossesPanel shows "COLLAPSE ~Nb" countdown when risk > 0.7

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `TELE-001` entropy | telemetry | Requires entropy values; velocity derived from entropy history |
| PPO update cycle | event | Only computed after entropy values accumulate (requires ≥5 samples) |
| entropy_history deque | data structure | Aggregator maintains rolling 100-sample window of entropy |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `TELE-045` collapse_risk_score | telemetry | Uses entropy + velocity for collapse prediction |
| HealthStatusPanel "Entropy D" | display | Renders velocity with arrow indicators |
| PPOLossesPanel collapse countdown | display | Calculates batches until critical using distance/velocity |
| Auto-intervention system | system | May trigger early stopping when velocity < -0.03 for N consecutive batches |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2024-09-20 | Initial | Created with Tamiyo debugging enhancements |
| 2025-01-03 | Audit | Verified wiring in telemetry audit; fully functional |

---

## 8. Notes

> **Design Decision:** Entropy velocity is computed from the last 10 entropy samples using least-squares linear regression (numpy polyfit). This provides:
> - Fast computation (numpy is 10x faster than pure Python)
> - Noise resistance (linear regression averages out single-sample spikes)
> - Sliding window (always recent trend, not historical average)
>
> **Early Return:** Function returns 0.0 if entropy_history has fewer than 5 samples. This prevents meaningless velocity estimates from random initialization.
>
> **Warmup Consideration:** During the first ~50 batches, entropy may fluctuate wildly due to random initialization. HealthStatusPanel and PPOLossesPanel both account for warmup periods in their display logic.
>
> **Collapse Countdown Algorithm:** When `collapse_risk_score > 0.7` and `entropy_velocity < 0`, PPOLossesPanel calculates an estimated batches-until-critical:
> ```
> distance = entropy - ENTROPY_CRITICAL
> batches = int(distance / abs(velocity))  # Linear extrapolation
> ```
> This assumes constant velocity (which is an approximation; actual collapse often accelerates near threshold).
>
> **Known Limitation:** Velocity estimation breaks down when entropy approaches critical threshold because the slope becomes steeper (collapse is nonlinear). The countdown becomes increasingly pessimistic as entropy nears the collapse point, which is actually conservative/safe for operator alerting.
>
> **Future Improvement:** Could implement per-head entropy velocity breakdown (slot, blueprint, tempo, op distributions) for more granular collapse detection.

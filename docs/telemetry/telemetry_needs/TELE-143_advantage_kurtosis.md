# Telemetry Record: [TELE-143] Advantage Kurtosis

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-143` |
| **Name** | Advantage Kurtosis |
| **Category** | `policy` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Are advantages heavy-tailed or thin-tailed? High kurtosis means extreme advantages occur too often."

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
| **Units** | Excess kurtosis (dimensionless) |
| **Range** | `(-inf, inf)` |
| **Precision** | 1 decimal place for display |
| **Default** | `NaN` (no data until first PPO update with valid advantages) |

### Semantic Meaning

> Excess kurtosis measures tail heaviness of the advantage distribution. Computed as:
>
> Excess Kurtosis = E[(X-μ)⁴] / σ⁴ - 3
>
> - **0**: Normal distribution (Gaussian), typical bell curve
> - **> 0 (positive/leptokurtic)**: Heavy tails, extreme advantages occur more often than Gaussian (outliers), can signal reward distribution issues or skill bifurcation
> - **< 0 (negative/platykurtic)**: Light tails, more uniform distribution, rarely extreme values
>
> By convention, excess kurtosis subtracts 3 so normal distribution equals 0. Without the -3, "Kurtosis" would be 3.0 for normal data.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `-1.0 ≤ kurtosis ≤ 3.0` | Advantage distribution is well-behaved, either Gaussian-like or moderately heavy-tailed |
| **Warning** | `kurtosis < -1.0` or `kurtosis > 3.0` | Distribution shape departing from normal; investigate if extreme advantages are problematic |
| **Critical** | `kurtosis < -2.0` or `kurtosis > 6.0` | Severe tail behavior (extremely light or heavy); likely algorithmic or reward design issue |

**Threshold Source:** `HealthStatusPanel._get_kurtosis_status()` in `src/esper/karn/sanctum/widgets/tamiyo_brain/health_status_panel.py`

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PPO update step, advantage batch statistics computation |
| **File** | `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Function/Method** | `PPOAgent.train_step()` (internally within advantage stats section) |
| **Line(s)** | 438-443 |

```python
# Excess kurtosis: E[(X-μ)⁴] / σ⁴ - 3 (0 = normal, >0 = heavy tails)
adv_kurtosis = (centered ** 4).mean() / (adv_std ** 4) - 3.0

# NaN signals "undefined" - std too low for meaningful higher moments
adv_kurtosis = torch.tensor(float("nan"), device=adv_mean.device, dtype=adv_mean.dtype)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `PPOAgent` emits via `metrics` dict in `train_step()` | `simic/agent/ppo.py` (lines 448) |
| **2. Collection** | `TelemetryEmitter.emit_ppo_update()` collects metrics and creates event payload | `simic/telemetry/emitters.py` (line 800) |
| **3. Aggregation** | `SanctumAggregator.handle_ppo_update()` receives payload and updates snapshot | `karn/sanctum/aggregator.py` (line 828) |
| **4. Delivery** | Written to `snapshot.tamiyo.advantage_kurtosis` field | `karn/sanctum/schema.py` |

```
[PPOAgent] --metrics["advantage_kurtosis"]--> [TelemetryEmitter] --event payload--> [SanctumAggregator] --> [TamiyoState.advantage_kurtosis]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `advantage_kurtosis` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.advantage_kurtosis` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 858 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| HealthStatusPanel | `widgets/tamiyo_brain/health_status_panel.py` (lines 61, 85-91) | Displayed as "kt:" field with color-coded status (ok/warning/critical) |
| Karn MCP telemetry view | `karn/mcp/views.py` (line 118) | Exported for offline analysis via telemetry database |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `PPOAgent.train_step()` computes excess kurtosis from centered advantage moments at line 439
- [x] **Transport works** — `metrics["advantage_kurtosis"]` passed to `TelemetryEmitter.emit_ppo_update()` at line 448
- [x] **Schema field exists** — `TamiyoState.advantage_kurtosis: float = float("nan")` at line 858
- [x] **Default is correct** — `NaN` appropriate before first valid PPO update (matches skewness pattern)
- [x] **Consumer reads it** — HealthStatusPanel accesses `tamiyo.advantage_kurtosis` at lines 61, 85-91
- [x] **Display is correct** — Value renders as `"{kurtosis:+.1f}"` (1 decimal, signed) or "---" for NaN
- [x] **Thresholds applied** — `_get_kurtosis_status()` applies -2.0/-1.0/3.0/6.0 thresholds at lines 417-421

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_vectorized.py` | Implicitly tested via advantage stats | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | Implicitly tested via PPO update handling | `[x]` |
| Unit (schema deserialization) | `tests/leyline/test_telemetry.py` (lines 163-167, 477, 537) | PPOUpdatePayload parsing with kurtosis values | `[x]` |
| Unit (emitter payload) | `tests/simic/telemetry/test_emitters.py` (line 35) | Default payload includes kurtosis | `[x]` |
| Integration (end-to-end) | — | Manual verification only | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification via HealthStatusPanel | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe HealthStatusPanel "Advantage" row
4. Verify kurtosis value (displayed as "kt: X.X") updates after each PPO batch
5. Verify "---" appears before first PPO update (NaN value)
6. Trigger warning threshold by engineering advantage distribution with high outliers (kurtosis > 3.0)
7. Trigger critical threshold by engineering extreme tail behavior (kurtosis > 6.0)

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO update cycle | event | Only populated after first PPO update with valid advantages completes |
| Advantage normalization | computation | Requires valid advantage batch with std > 0.05 |
| Valid data mask | computation | If no valid advantages in batch, kurtosis is NaN |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| HealthStatusPanel status | display | Kurtosis status contributes to overall panel status color |
| Karn telemetry database | storage | Exported via MCP views for analysis and trend monitoring |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Audit | Created telemetry record, verified full wiring from emitter to widget |

---

## 8. Notes

> **Design Decision:** Excess kurtosis (with -3 subtraction) is used so that a normal distribution equals 0, making interpretation more intuitive. Without the -3, the value would be 3.0 for normal data, which is confusing.
>
> **Computation:** Kurtosis is computed only when advantage standard deviation exceeds 0.05. Below this threshold, the standard deviation is too low for reliable higher-moment estimation, and kurtosis is set to NaN to signal "undefined" rather than masking the problem with a default value.
>
> **Pairing with Skewness:** Advantage kurtosis is always emitted alongside skewness (`TELE-144`). Together they characterize the full shape of the advantage distribution: skewness describes asymmetry, kurtosis describes tail heaviness. Both use the same std > 0.05 threshold for validity.
>
> **Display Format:** The HealthStatusPanel displays kurtosis with 1 decimal place and a sign (e.g., "+3.2", "-0.5") to quickly show direction and magnitude of tail behavior. NaN values show as "---" to indicate no data yet.
>
> **Interpretation Examples:**
> - `kurtosis = 0.0`: Advantage distribution is Gaussian-like
> - `kurtosis = 2.5`: Moderately heavy-tailed, more outliers than Gaussian (still healthy)
> - `kurtosis = 5.0`: Very heavy-tailed, many extreme advantages (warning)
> - `kurtosis = -1.5`: Light-tailed, very uniform, no extreme values (atypical but not harmful)

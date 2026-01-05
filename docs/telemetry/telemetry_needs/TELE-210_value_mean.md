# Telemetry Record: [TELE-210] Value Mean

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-210` |
| **Name** | Value Mean |
| **Category** | `value` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "What is the mean value function prediction? Is the critic estimating reasonable expected returns?"

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
| **Units** | raw value (in reward scale) |
| **Range** | `(-inf, inf)` — typically [-10, 10] for normalized rewards |
| **Precision** | 3-4 decimal places |
| **Default** | `0.0` |

### Semantic Meaning

> The mean (average) of all value function predictions from the critic network in a batch of states.
>
> V(s) = E[G_t | s_t = s], where G_t is the discounted cumulative future reward.
>
> The value_mean tells us what return the critic _expects_ on average. Together with value_std, it forms the Coefficient of Variation (CoV = value_std / |value_mean|) which measures relative uncertainty in value estimates.
>
> High value_mean with low std = stable, confident value estimates. High std with low/near-zero mean = unstable critic (NaN/divergence risk). Constant value_mean = critic collapse (not learning).

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `0.5 < abs(value_mean) < 10` and `CoV < 2.0` | Critic learning meaningful returns |
| **Warning** | `CoV > 2.0` or `value_range < 0.1` | Critic uncertainty high or stuck |
| **Critical** | `CoV > 3.0` or (`value_range < 0.1` and `value_std < 0.01`) | Critic collapse or severe divergence |

**Threshold Source:** Computed in `HealthStatusPanel._get_value_status()` using relative CoV thresholds (value_std / |value_mean|)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Critic network forward pass, aggregated across batch |
| **File** | `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Function/Method** | `PPOAgent.update()` → line 907 `values.mean()` |
| **Line(s)** | 907, 921 |

```python
# Value predictions from critic network
values = self.policy.network.get_value(states)  # shape: (batch_size,)

# Aggregate: mean across entire batch
logging_tensors = torch.stack([
    ...
    values.mean(),  # Line 907 - element [7]
    values.std(),   # Line 908 - element [8]
    values.min(),   # Line 909 - element [9]
    values.max(),   # Line 910 - element [10]
]).cpu().tolist()

metrics["value_mean"].append(logging_tensors[7])  # Line 921
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `TelemetryEmitter.emit_ppo_update()` | `simic/telemetry/emitters.py` (line ~811) |
| **2. Collection** | Event payload field in `PPOUpdateEvent` | `leyline/telemetry.py` (line 726) |
| **3. Aggregation** | `SanctumAggregator.handle_ppo_update()` | `karn/sanctum/aggregator.py` (line 931) |
| **4. Delivery** | Written to `snapshot.tamiyo.value_mean` | `karn/sanctum/schema.py` |

```
[PPOAgent.update() @ line 907]
  ↓ values.mean() tensor
  ↓ logging_tensors[7]
  ↓ metrics["value_mean"] list
  ↓
[TelemetryEmitter.emit_ppo_update() @ emitters.py:811]
  ↓ PPOUpdateEvent.value_mean field
  ↓
[SanctumAggregator.handle_ppo_update() @ aggregator.py:931]
  ↓ self._tamiyo.value_mean = payload.value_mean
  ↓
[TamiyoState.value_mean field]
  ↓
[snapshot.tamiyo.value_mean]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `value_mean` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.value_mean` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 983 |

```python
@dataclass
class TamiyoState:
    ...
    # Value function statistics (for divergence detection)
    value_mean: float = 0.0
    value_std: float = 0.0
    value_min: float = 0.0
    value_max: float = 0.0
    initial_value_spread: float | None = None  # Set after warmup for relative thresholds
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| HealthStatusPanel | `widgets/tamiyo_brain/health_status_panel.py` | "Value Range" row - used for Coefficient of Variation (CoV) calculation at line 360, 370 |
| (indirectly used in value_status detection) | Same file, `_get_value_status()` method | CoV = value_std / abs(value_mean) for health status detection |

**Display Location:**
- In HealthStatusPanel (line 291-305), value_mean is accessed at line 360 within `_get_value_status(tamiyo)` method
- Used to compute CoV (line 370): `cov = v_std / abs(v_mean)`
- CoV > 3.0 = critical, CoV > 2.0 = warning (lines 371-374)
- Displayed indirectly via value range [min, max] coloring based on health status

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `PPOAgent.update()` computes values.mean() from critic predictions
- [x] **Transport works** — Event includes value_mean field and routed to aggregator
- [x] **Schema field exists** — `TamiyoState.value_mean: float = 0.0` at line 983
- [x] **Default is correct** — 0.0 appropriate before first PPO update
- [x] **Consumer reads it** — HealthStatusPanel accesses `snapshot.tamiyo.value_mean` in _get_value_status()
- [x] **Display is correct** — Value used in Coefficient of Variation calculation for health coloring
- [x] **Thresholds applied** — CoV thresholds (2.0 warning, 3.0 critical) applied in _get_value_status()

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_ppo.py` | `test_value_mean_computed` | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_ppo_update_populates_value_mean` | `[ ]` |
| Integration (end-to-end) | `tests/integration/test_telemetry_flow.py` | `test_value_mean_reaches_health_panel` | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe HealthStatusPanel "Value Range" row
4. Verify value_mean is accessed in health status calculation (values [min, max] display changes color based on CoV)
5. Trigger high variance (value_std >> |value_mean|) to verify warning coloring
6. Monitor for collapse patterns (value_range < 0.1 and value_std < 0.01 → critical red)

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Critic network forward pass | computation | Value predictions require valid policy network |
| PPO update cycle | event | Only populated after first PPO update completes |
| Batch states | data | States must be present for critic evaluation |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `TELE-211` value_std | telemetry | Used together for CoV calculation |
| `TELE-212` value_min | telemetry | Part of value range [min, max] display |
| `TELE-213` value_max | telemetry | Part of value range [min, max] display |
| HealthStatusPanel health status | display | Drives color coding of Value Range row |
| Vectorized training aggregation | system | Averaged across environments (training/vectorized.py:308) |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Initial | Created during telemetry audit 2 - verified full data flow from PPO critic to HealthStatusPanel |

---

## 8. Notes

> **Design Decision:** Value statistics are computed across entire batch before averaging epochs, capturing the distribution of predictions in a single training step. This avoids per-sample noise while maintaining batch-level variance signals.
>
> **Key Integration Point:** The Coefficient of Variation (CoV = value_std / |value_mean|) is the primary health metric for the critic. High CoV indicates either:
> - High uncertainty (critic oscillating) → warning/critical coloring
> - Near-zero mean with non-zero std → NaN/divergence risk
>
> **Vectorized Training:** In multi-environment training (src/esper/simic/training/vectorized.py:308), value_mean is averaged across environments (sum(values) / len(values)), providing a global view of critic stability.
>
> **Warmup Interaction:** During warmup (first ~10 PPO updates), value estimates may be artificially scattered due to random initialization. The initial_value_spread field (set after warmup) enables relative thresholds to distinguish warmup variance from true divergence.
>
> **No Bug-Hiding Patterns:** Uses direct field access (tamiyo.value_mean) in widget code, not defensive .get() patterns. If value_mean is missing, that's a wiring error that should surface visibly.

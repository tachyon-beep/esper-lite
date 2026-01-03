# Telemetry Record: [TELE-340] LSTM Hidden State Health

> **Status:** `[ ] Planned` `[ ] In Progress` `[x] Wired` `[x] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-340` |
| **Name** | LSTM Hidden State Health |
| **Category** | `gradient` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Is the LSTM hidden state numerically stable, or is it exploding/vanishing/corrupted?"

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
| **Type** | `LSTMHealthMetrics` (dataclass with float/bool fields) |
| **Fields** | `h_norm`, `c_norm`, `h_max`, `c_max`, `has_nan`, `has_inf` |
| **Units** | L2 norm (unitless), max absolute value, boolean flags |
| **Range** | norms: `[0, ∞)`, max: `[0, ∞)`, flags: `bool` |
| **Precision** | 3 decimal places for display |
| **Default** | `None` (if no LSTM in use) |

### Semantic Meaning

> LSTM hidden states (h, c) can drift during training:
> - **h_norm**: L2 norm of hidden state tensor, computed as `||h||₂`
> - **c_norm**: L2 norm of cell state tensor, computed as `||c||₂`
> - **h_max/c_max**: Maximum absolute value (catches localized spikes)
> - **has_nan/has_inf**: Numerical stability flags
>
> These metrics detect gradient-induced corruption during BPTT (Pascanu et al., 2013).

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `1e-6 < norm < 100.0`, no NaN/Inf | Normal operating range |
| **Warning** | `norm > 100.0` or `norm < 1e-6` | Explosion or vanishing |
| **Critical** | `has_nan=True` or `has_inf=True` | Irrecoverable corruption |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | LSTM hidden state after policy forward pass |
| **File** | `/home/john/esper-lite/src/esper/simic/telemetry/lstm_health.py` |
| **Function/Method** | `compute_lstm_health()` |
| **Line(s)** | 71-124 |

```python
def compute_lstm_health(
    hidden: tuple[torch.Tensor, torch.Tensor] | None,
) -> LSTMHealthMetrics | None:
    if hidden is None:
        return None
    h, c = hidden
    with torch.inference_mode():
        # Single GPU-CPU sync (M14 optimization)
        h_norm_t = torch.linalg.vector_norm(h)
        c_norm_t = torch.linalg.vector_norm(c)
        # ... (batched computation)
        all_values = torch.stack([...])
        result = all_values.tolist()  # Single sync!
    return LSTMHealthMetrics(...)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Called after PPO updates | `vectorized.py:3603` |
| **2. Collection** | Via `AnomalyDetector.check_lstm_health()` | `anomaly_detector.py:301` |
| **3. Aggregation** | Merged into `AnomalyReport` | `vectorized.py:3611-3614` |
| **4. Delivery** | Via `_handle_telemetry_escalation()` | `vectorized.py:3616` |

```
[batched_lstm_hidden]
  --> compute_lstm_health()
  --> anomaly_detector.check_lstm_health()
  --> anomaly_report
  --> _handle_telemetry_escalation()
  --> _emit_anomaly_diagnostics()
```

### Schema Location

| Property | Value |
|----------|-------|
| **Integration** | AnomalyDetector + AnomalyReport |
| **Anomaly Types** | `lstm_nan`, `lstm_inf`, `lstm_h_explosion`, `lstm_c_explosion`, `lstm_h_vanishing`, `lstm_c_vanishing` |
| **Escalation** | Triggers DEBUG telemetry level on anomaly |
| **File** | `/home/john/esper-lite/src/esper/simic/telemetry/anomaly_detector.py` |
| **Line** | 301-365 |

### Consumers (Display)

| Consumer | Type | Usage |
|----------|------|-------|
| AnomalyReport | Internal | Merged with other anomalies for escalation |
| TelemetryHub | Event | Emitted via `_emit_anomaly_diagnostics()` |
| Logger | Warning | Logged when anomaly detected |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `compute_lstm_health()` computes metrics
- [x] **Transport works** — Called after PPO updates in vectorized.py
- [x] **Schema field exists** — Anomaly types defined in AnomalyDetector
- [x] **Default is correct** — Returns None when no LSTM
- [x] **Consumer reads it** — AnomalyReport aggregates findings
- [x] **Display is correct** — Escalates to DEBUG level
- [x] **Thresholds applied** — 100.0 max, 1e-6 min

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `test_lstm_health.py` | `TestComputeLstmHealth` | `[x]` |
| Unit (detector) | `test_anomaly_detector.py` | `TestCheckLstmHealth` | `[x]` |
| Integration | `test_anomaly_detector.py` | `test_multiple_anomalies` | `[x]` |
| Visual (TUI) | N/A | Not displayed directly | N/A |

### Manual Verification Steps

1. Start training with LSTM policy: `uv run esper ppo --episodes 10`
2. LSTM health is checked after each PPO update
3. If anomaly detected, telemetry escalates to DEBUG level
4. Check logs for `lstm_*` anomaly warnings

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `batched_lstm_hidden` | tensor | LSTM hidden state from policy |
| PPO update cycle | event | Only checked after updates |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| Anomaly escalation | system | Triggers DEBUG telemetry on anomaly |
| Training logs | output | Warnings on unhealthy state |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Claude | Initial creation (B7-DRL-04 fix) |
| | | |

---

## 8. Notes

> **DRL Context:** LSTM hidden state drift is a critical failure mode documented by Pascanu et al., 2013.
> Hidden states can explode or vanish due to gradient issues during BPTT.
> NaN/Inf can propagate through hidden states before manifesting in loss.
>
> **Design Decision:** Checking after PPO updates (not every step) because:
> 1. This is when gradient-induced corruption occurs
> 2. 1 check per batch (~1ms) is negligible overhead
> 3. Mid-episode checks would have no actionable response
>
> **DO NOT reset hidden state when unhealthy** - this is a bug-hiding anti-pattern per CLAUDE.md.
> If LSTM health is bad, the training configuration is broken and should be diagnosed.

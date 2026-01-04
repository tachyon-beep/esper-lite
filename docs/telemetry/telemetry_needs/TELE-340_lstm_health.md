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
| **Fields** | `h_l2_total`, `c_l2_total`, `h_rms`, `c_rms`, `h_env_rms_mean`, `h_env_rms_max`, `c_env_rms_mean`, `c_env_rms_max`, `h_max`, `c_max`, `has_nan`, `has_inf` |
| **Units** | L2/RMS magnitudes (unitless), max absolute value (unitless), boolean flags |
| **Range** | magnitudes/max: `[0, ∞)`, flags: `bool` |
| **Precision** | 3 decimal places for display |
| **Default** | `None` (if no LSTM in use) |

### Semantic Meaning

> LSTM hidden states (h, c) can drift during training:
> - **h_l2_total / c_l2_total**: Total L2 norm of the full tensor `||h||₂`, `||c||₂` (NOT batch-size invariant)
> - **h_rms / c_rms**: RMS magnitude (batch-size invariant), computed as `||x||₂ / sqrt(numel(x))` = `sqrt(mean(x²))`
> - **h_env_rms_* / c_env_rms_***: Per-environment RMS stats (RMS over `[layers * hidden_dim]` per env), logs mean + max for outlier detection
> - **h_max/c_max**: Maximum absolute value (catches localized spikes)
> - **has_nan/has_inf**: Numerical stability flags
>
> These metrics detect gradient-induced corruption during BPTT (Pascanu et al., 2013).
>
> **Why RMS?** Total L2 norms scale with `sqrt(numel)` so increasing `n_envs`, `hidden_dim`, or `layers` inflates the number even when per-element magnitudes are stable. RMS removes that deployment-scaling effect.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `1e-6 < rms < 5.0`, no NaN/Inf | Normal operating range (scale-free) |
| **Warning** | `rms > 5.0` | Elevated per-element magnitude (Sanctum warning threshold) |
| **Critical** | `rms > 10.0` or `env_rms_max > 10.0` or NaN/Inf | Likely saturation/instability (Sanctum critical + anomaly threshold) |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | LSTM hidden state after policy forward pass |
| **File** | `/home/john/esper-lite/src/esper/simic/telemetry/lstm_health.py` |
| **Function/Method** | `compute_lstm_health()` |
| **Line(s)** | 93-204 |

```python
def compute_lstm_health(
    hidden: tuple[torch.Tensor, torch.Tensor] | None,
) -> LSTMHealthMetrics | None:
    if hidden is None:
        return None
    h, c = hidden
    with torch.inference_mode():
        # Capacity/load (scales with sqrt(numel))
        h_l2_total_t = torch.linalg.vector_norm(h)
        c_l2_total_t = torch.linalg.vector_norm(c)

        # Scale-free health (RMS)
        inv_sqrt_numel = 1.0 / (float(h.numel()) ** 0.5)
        h_rms_t = h_l2_total_t * inv_sqrt_numel
        c_rms_t = c_l2_total_t * inv_sqrt_numel

        # Per-env RMS stats (outlier detection across envs)
        # h_env: [batch, layers*hidden_dim]
        # h_env_rms = ||h_env||2 / sqrt(layers*hidden_dim)

        # Single GPU-CPU sync (M14 optimization)
        all_values = torch.stack([...])
        result = all_values.tolist()  # Single sync!
    return LSTMHealthMetrics(...)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Called after PPO updates | `vectorized.py:3684` |
| **2. Collection** | Via `AnomalyDetector.check_lstm_health()` | `anomaly_detector.py:303` |
| **3. Aggregation** | Merged into `AnomalyReport` | `vectorized.py:3694-3697` |
| **4. Delivery** | Via `_handle_telemetry_escalation()` | `vectorized.py:3701` |
| **5. TUI Display** | `metrics.update(lstm_health.to_dict())` flows to PPOUpdatePayload | `vectorized.py:3699` |

```
[batched_lstm_hidden]
  --> compute_lstm_health()
  |
  |--> anomaly_detector.check_lstm_health()  (anomaly path)
  |      --> anomaly_report
  |      --> _handle_telemetry_escalation()
  |      --> _emit_anomaly_diagnostics()
  |
  `--> metrics.update(lstm_health.to_dict()) (standard telemetry path)
         --> emit_ppo_update_event()
         --> PPOUpdatePayload (leyline contract)
         --> TelemetryHub.emit(PPO_UPDATE_COMPLETED)
               |
               +--> SanctumAggregator (TUI consumer)
               +--> WandBBackend (cloud consumer)
               +--> KarnCollector (analytics consumer)
               +--> [any subscriber]
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
| **Sanctum TUI** | Visual | HealthStatusPanel shows LSTM `h_rms`/`c_rms` with status colors |
| **TamiyoState** | Schema | `lstm_*_rms`, `lstm_*_env_rms_*`, `lstm_*_l2_total`, `lstm_*_max`, `lstm_has_nan`, `lstm_has_inf` |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `compute_lstm_health()` computes metrics
- [x] **Transport works** — Called after PPO updates in vectorized.py
- [x] **Schema field exists** — Anomaly types defined in AnomalyDetector
- [x] **Default is correct** — Returns None when no LSTM
- [x] **Consumer reads it** — AnomalyReport aggregates findings
- [x] **Display is correct** — Escalates to DEBUG level; Sanctum TUI shows in HEALTH panel
- [x] **Thresholds applied** — Warning `>5.0 RMS`, Critical `>10.0 RMS`, Vanishing `<1e-6 RMS`

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `test_lstm_health.py` | `TestComputeLstmHealth` | `[x]` |
| Unit (detector) | `test_anomaly_detector.py` | `TestCheckLstmHealth` | `[x]` |
| Integration | `test_anomaly_detector.py` | `test_multiple_anomalies` | `[x]` |
| Telemetry flow | `test_emitters.py` | `test_emit_ppo_update_event_includes_lstm_health` | `[x]` |
| Telemetry default | `test_emitters.py` | `test_emit_ppo_update_event_lstm_health_defaults_to_none` | `[x]` |
| Visual (TUI) | Manual | HealthStatusPanel shows LSTM h/c norms | `[x]` |

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
| 2026-01-03 | Claude | Added Sanctum TUI display via PPOUpdatePayload telemetry path |
| 2026-01-04 | Codex | Switched health signal to RMS + per-env RMS stats (batch-invariant); preserved total L2 as capacity metric |

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

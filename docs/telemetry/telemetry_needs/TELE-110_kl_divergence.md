# Telemetry Record: [TELE-110] KL Divergence

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-110` |
| **Name** | KL Divergence |
| **Category** | `policy` |
| **Priority** | `P0-critical` |

## 2. Purpose

### What question does this answer?

> "Is the policy changing too rapidly, violating the PPO trust region constraint?"

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
| **Units** | nats (natural logarithm) |
| **Range** | `[0.0, ∞)` — typically 0.0-0.1 in healthy training |
| **Precision** | 4 decimal places for display |
| **Default** | `0.0` |

### Semantic Meaning

> KL divergence measures the distance between the old (frozen) policy and new (updated) policy distributions. Computed as:
>
> KL(π_old || π_new) = E_s[Σ_a π_old(a|s) * log(π_old(a|s) / π_new(a|s))]
>
> In PPO, this is approximated as:
> KL ≈ E[(exp(log_ratio) - 1) - log_ratio] where log_ratio = log(π_new/π_old)
>
> High KL = policy changed dramatically, violating trust region
> Low KL = policy updated conservatively, within trust region

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `kl_divergence <= 0.015` | Policy changing conservatively within trust region |
| **Warning** | `0.015 < kl_divergence <= 0.03` | Policy drift detected, monitor closely |
| **Critical** | `kl_divergence > 0.03` | Excessive policy change, may destabilize training |

**Threshold Source:** `TUIThresholds.KL_WARNING = 0.015`, `TUIThresholds.KL_CRITICAL = 0.03` in `/home/john/esper-lite/src/esper/karn/constants.py`

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PPO update step, computed from old vs new policy ratio |
| **File** | `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Function/Method** | `PPOAgent._compute_policy_loss()` (per-step), aggregated into epoch metrics |
| **Line(s)** | ~733-742 |

```python
# KL computation per timestep across all action heads
kl_per_step = (torch.exp(log_ratio_clamped) - 1) - log_ratio_clamped
# Weighted average across heads by causal relevance
approx_kl = (weighted_kl_sum / total_weight.clamp(min=1e-8)).item()
metrics["approx_kl"].append(approx_kl)
```

**Note:** KL is computed within `torch.inference_mode()` to avoid gradient tracking. The computation uses clamped log_ratio to prevent NaN from numerical overflow (see line 727-731 bugfix comment).

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `TelemetryEmitter.emit_ppo_update()` | `simic/telemetry/emitters.py:788` |
| **2. Collection** | Event payload with `kl_divergence` field (required) | `leyline/telemetry.py:621` |
| **3. Aggregation** | `SanctumAggregator.handle_ppo_update()` | `karn/sanctum/aggregator.py:813-815` |
| **4. Delivery** | Written to `snapshot.tamiyo.kl_divergence` and history deque | `karn/sanctum/schema.py:835, 950` |

```
[PPOAgent._compute_policy_loss()] --emit_ppo_update()--> [TelemetryEmitter] --PPOUpdatePayload--> [SanctumAggregator] --> [TamiyoState.kl_divergence]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `kl_divergence` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.kl_divergence` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 835 (field definition), 950 (history deque) |

**Schema definitions:**
```python
kl_divergence: float = 0.0                                               # Line 835
kl_divergence_history: deque[float] = field(default_factory=lambda: deque(maxlen=10))  # Line 950
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| HealthStatusPanel | `widgets/tamiyo_brain/health_status_panel.py:137-160` | Displayed as "KL Diverge" metric with sparkline trend |
| StatusBanner | `widgets/tamiyo_brain/status_banner.py:144-150, 209-210, 230-231` | Used for critical/warning status detection; appends "!" indicator |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `PPOAgent` computes approx_kl during each update batch
- [x] **Transport works** — `PPOUpdatePayload.kl_divergence` (line 621 leyline) carries value
- [x] **Emission calls emitter** — `TelemetryEmitter.emit_ppo_update()` includes kl_divergence (line 788)
- [x] **Schema field exists** — `TamiyoState.kl_divergence: float = 0.0` (line 835 schema.py)
- [x] **Aggregator handles it** — `handle_ppo_update()` writes to both field and history (lines 813-815)
- [x] **Consumer reads it** — Both HealthStatusPanel and StatusBanner access `snapshot.tamiyo.kl_divergence`
- [x] **Display is correct** — HealthStatusPanel renders as 6.4f format with sparkline; StatusBanner shows 3f format with trend arrow
- [x] **Thresholds applied** — Both widgets use `TUIThresholds.KL_WARNING (0.015)` and `TUIThresholds.KL_CRITICAL (0.03)`
- [x] **History tracking** — Both widgets access `tamiyo.kl_divergence_history` deque (maxlen=10 for sparkline)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_ppo.py` | Tests KL computation in PPO update | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | Tests PPO_UPDATE_COMPLETED handling | `[x]` |
| Integration (end-to-end) | `tests/integration/test_telemetry_flow.py` | Tests KL reaches TUI snapshot | `[x]` |
| Visual (TUI snapshot) | — | Manual verification on HealthStatusPanel | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe HealthStatusPanel "KL Diverge" row
4. Verify KL value updates after each PPO batch update
5. Check sparkline renders with deque history (10-sample rolling window)
6. Observe trend arrow (↓ = declining, ↑ = rising; rising is bad for KL)
7. Artificially cause high KL (e.g., large learning rate) to verify warning/critical coloring
8. Verify StatusBanner shows "KL" in warning issues when KL > 0.015
9. Verify StatusBanner shows "KL" in critical issues when KL > 0.03

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO update cycle | event | Only populated after first PPO update completes |
| Policy network forward pass | computation | Requires valid π_new and π_old distributions |
| Action head masks | data | Uses HEAD_NAMES and head_masks for weighted averaging |
| Log ratio clamping | computation | Depends on clamped log_ratio to prevent NaN (line 732) |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `TELE-112` policy_drift_rate | telemetry | Computed as KL velocity (d(KL)/d(batch)) |
| `TELE-113` trust_region_health | telemetry | Uses KL vs target_kl threshold for PPO health |
| Anomaly detector | system | `check_kl_divergence()` at line 231 flags high KL spikes |
| Early stopping logic | system | Line 763: skips update if KL > 1.5 * target_kl |
| StatusBanner status | display | Drives WARN/FAIL status when thresholds exceeded |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Telemetry Audit | Initial creation; verified wiring across all layers |

---

## 8. Notes

> **Design Decision:** KL is averaged across action heads using causal weight (fraction of timesteps where head is relevant). This prevents distortion from heads with sparse relevance (e.g., style head active <5% of timesteps). See lines 737-740 for weight computation.
>
> **Numerical Stability:** The KL computation uses clamped log_ratio (line 732) to prevent NaN from exp(large_value) overflow. Without this fix, large policy changes could produce inf/NaN, silently breaking monitoring. See bugfix comment at lines 727-731.
>
> **Early Stopping:** If KL exceeds 1.5x target_kl (line 763), the entire PPO update is skipped to protect against trust region violation. This is standard in OpenAI baselines and Stable-Baselines3. The anomaly detector also monitors KL spikes as a leading indicator of training instability.
>
> **History Tracking:** Stored as deque(maxlen=10) for sparkline rendering. The 10-sample window provides ~10 batches of history (typically 100-1000 training steps depending on batch size).
>
> **Display Formatting:**
> - HealthStatusPanel: `f"{kl:.4f}"` (4 decimal places) with sparkline
> - StatusBanner: `f"{kl:.3f}"` (3 decimal places) with trend arrow
>
> **Related Metrics:**
> - `TELE-001` entropy: Policy should maintain exploration (inverse relationship to KL sometimes, but not always)
> - `TELE-102` clip_fraction: High clipping often paired with high KL (policy change constrained)
> - Pre/post-clip gradient norm indicates if clipping is active during high KL periods

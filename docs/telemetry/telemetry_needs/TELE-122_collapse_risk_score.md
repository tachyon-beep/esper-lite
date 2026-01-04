# Telemetry Record: [TELE-122] Collapse Risk Score

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-122` |
| **Name** | Collapse Risk Score |
| **Category** | `policy` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "What is the computed risk of policy collapse? Combines entropy level and velocity into a risk score."

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
| **Units** | Risk score (dimensionless) |
| **Range** | `[0.0, 1.0]` |
| **Precision** | 3 decimal places for display |
| **Default** | `0.0` |

### Semantic Meaning

> Collapse Risk Score combines entropy proximity and velocity to predict imminent policy collapse.
>
> **Formula:**
> - If entropy ≤ critical_threshold (0.1): risk = 1.0 (already collapsed)
> - If entropy > critical_threshold: risk = proximity_risk + time_risk with hysteresis
>   - **Proximity risk** (30% weight): How close current entropy is to critical (0.0-0.3)
>   - **Time risk** (70% weight): Estimated batches to collapse based on entropy velocity
>     - > 100 batches to collapse = 0.1 risk
>     - 50-100 batches = 0.25 risk
>     - 20-50 batches = 0.5 risk
>     - 10-20 batches = 0.7 risk
>     - < 10 batches = 0.9 risk
> - **Hysteresis** (0.08): Risk score only updates if change exceeds threshold to prevent flapping

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `risk < 0.3` | Low collapse risk, policy stable |
| **Warning** | `0.3 <= risk < 0.7` | Entropy declining, monitor closely |
| **Critical** | `risk >= 0.7` | Imminent collapse, intervention recommended |

**Threshold Source:** PPOLossesPanel triggers alert border `COLLAPSE ~Xb` when risk > 0.7; HealthStatusPanel displays risk in color-coded entropy diagnostic.

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Aggregated from PPO entropy history in TamiyoState |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Function** | `compute_collapse_risk()` |
| **Line(s)** | ~87-167 |

```python
def compute_collapse_risk(
    entropy_history: deque[float] | list[float],
    critical_threshold: float = 0.3,
    warning_threshold: float = 0.5,
    max_healthy_entropy: float = 1.39,
    previous_risk: float = 0.0,
    hysteresis: float = 0.08,
) -> float:
    """Compute entropy collapse risk score (0.0 to 1.0).

    Risk is based on:
    - Current distance from critical threshold (proximity)
    - Velocity (rate of decline)
    - Hysteresis to prevent risk score flapping
    """
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Computed during PPO update aggregation | `karn/sanctum/aggregator.py` |
| **2. Collection** | Added to entropy_history deque (maxlen=10) | `karn/sanctum/schema.py` TamiyoState |
| **3. Aggregation** | compute_collapse_risk() called with entropy_history | `karn/sanctum/aggregator.py` ~957-964 |
| **4. Delivery** | Written to snapshot.tamiyo.collapse_risk_score | `karn/sanctum/schema.py` TamiyoState field |

```
[entropy_history] --> compute_collapse_risk() --> [TamiyoState.collapse_risk_score]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `collapse_risk_score` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.collapse_risk_score` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | ~975 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| PPOLossesPanel | `widgets/tamiyo_brain/ppo_losses_panel.py` | Alerts when > 0.7: changes border title to "PPO LOSSES !! COLLAPSE ~Xb" |
| HealthStatusPanel | `widgets/tamiyo_brain/health_status_panel.py` | Displays in entropy diagnostic row with risk assessment |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `compute_collapse_risk()` function defined in schema.py
- [x] **Transport works** — Called in aggregator.handle_ppo_update() after entropy_history updated
- [x] **Schema field exists** — `TamiyoState.collapse_risk_score: float = 0.0`
- [x] **Default is correct** — 0.0 appropriate (no collapse risk before first PPO update)
- [x] **Consumer reads it** — Both PPOLossesPanel and HealthStatusPanel access the field
- [x] **Display is correct** — Risk > 0.7 triggers alert border; risk used in entropy diagnostics
- [x] **Thresholds applied** — Constants passed from aggregator (ENTROPY_CRITICAL, ENTROPY_WARNING, ENTROPY_MAX)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (compute_collapse_risk) | — | Manual verification in schema.py | `[x]` |
| Integration (aggregator) | — | Tested via aggregator flow | `[x]` |
| Visual (TUI snapshot) | — | Manual verification | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 100`
2. Open Sanctum TUI (auto-opens)
3. Observe PPOLossesPanel border title
4. Trigger entropy decline to verify alert (e.g., reduce entropy_coef)
5. Verify border changes to "PPO LOSSES !! COLLAPSE ~Xb" when risk > 0.7
6. Check HealthStatusPanel Entropy D row for risk assessment

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `TELE-001` entropy | telemetry | Requires entropy values to compute risk and velocity |
| `TELE-002` entropy_velocity | telemetry | Derived from entropy_history for time-to-collapse calculation |
| PPO update cycle | event | Only populated after first PPO update completes |
| TUIThresholds constants | infrastructure | ENTROPY_CRITICAL (0.1), ENTROPY_WARNING (0.3), ENTROPY_MAX (1.39 ≈ ln(4)) |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| PPOLossesPanel alert border | display | Triggers "COLLAPSE ~Xb" warning when risk > 0.7 |
| HealthStatusPanel diagnostics | display | Used in entropy health assessment |
| Training operator monitoring | manual | High-risk scores trigger human intervention |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2024-06-15 | Initial | Implemented as part of entropy diagnostics |
| 2024-09-20 | Refactor | Moved compute_collapse_risk to schema.py, added hysteresis |
| 2025-01-03 | Audit | Verified wiring in telemetry audit TELE-122 |

---

## 8. Notes

> **Design Decision:** Collapse risk combines proximity (30%) and time-to-collapse (70%) to balance static position with dynamic trend. Proximity alone would miss rapidly declining entropy; velocity alone would miss entropy already near critical.
>
> **Hysteresis (0.08):** Prevents risk score from flapping due to batch-to-batch entropy noise. Only meaningful changes update the score, reducing alert fatigue.
>
> **Critical Threshold (0.1):** Policy is <10% of max entropy for categorical action space, indicating near-deterministic behavior.
>
> **Warning Threshold (0.3):** Policy at 30% of max entropy, converging but not yet critical. Used to weight proximity vs time risk differently.
>
> **Max Healthy Entropy (1.39):** Approximately ln(4), the maximum entropy for a 4-action head. Used as reference point for proximity calculation.
>
> **PPOLossesPanel Integration:** When risk > 0.7, displays estimated batches to collapse based on current velocity. Helps operators understand urgency of intervention. Shows "999" if velocity is 0 (entropy stable).
>
> **Future Enhancement:** Per-head collapse risk could detect collapse in specific action heads (e.g., slot selection) while others remain healthy.

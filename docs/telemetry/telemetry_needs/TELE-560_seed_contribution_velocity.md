# Telemetry Record: [TELE-560] Seed Contribution Velocity

> **Status:** `[x] Planned` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-560` |
| **Name** | Seed Contribution Velocity |
| **Category** | `seed` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Is this seed's contribution improving or declining over time?"

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
| **Units** | rate of change (dimensionless) |
| **Range** | `(-∞, +∞)` |
| **Precision** | Displayed as trend arrow, not numeric |
| **Default** | `0.0` (stable) |

### Semantic Meaning

> Contribution velocity is an exponential moving average (EMA) of contribution changes over time. It indicates the direction and momentum of a seed's contribution trend:
>
> - **velocity > EPSILON:** Improving — contribution is increasing
> - **velocity ≈ 0:** Stable — contribution is steady
> - **velocity < -EPSILON:** Declining — contribution is decreasing
>
> This helps identify seeds that are gaining or losing effectiveness over training.

### Health Thresholds

| Context | Condition | Meaning |
|---------|-----------|---------|
| Any stage | velocity > CONTRIBUTION_VELOCITY_EPSILON (0.01) | Improving (green ↗) |
| Any stage | velocity < -CONTRIBUTION_VELOCITY_EPSILON (-0.01) | Declining (yellow ↘) |
| Any stage | |velocity| ≤ 0.01 | Stable (dim --) |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | EMA computation of contribution delta changes |
| **File** | `/home/john/esper-lite/src/esper/kasmina/slot.py` |
| **Class** | `SeedState` |
| **Field** | `contribution_velocity: float` (schema line 418) |

```python
@dataclass
class SeedState:
    """State of a single seed slot."""
    # Inter-slot interaction metrics (from counterfactual engine)
    contribution_velocity: float = 0.0  # EMA of contribution changes (trend direction)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Computation** | EMA of contribution deltas | `tamiyo/counterfactual.py` |
| **2. Storage** | SeedState.contribution_velocity updated | `kasmina/slot.py` |
| **3. Telemetry** | SeedTelemetry.contribution_velocity field | `leyline/telemetry.py` |
| **4. Schema** | SeedState.contribution_velocity in Sanctum | `karn/sanctum/schema.py` (line 418) |
| **5. Display** | EnvDetailScreen SeedCard | `karn/sanctum/widgets/env_detail_screen.py` (lines 186-194) |

```
[EMA computation]
  --updates-->
  [SeedState.contribution_velocity]
  --EPOCH_COMPLETED event-->
  [SeedTelemetry.contribution_velocity]
  --SanctumAggregator-->
  [schema.SeedState.contribution_velocity]
  --SeedCard-->
  ["Trend: ↗ improving"]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SeedState` (Sanctum schema) |
| **Field** | `contribution_velocity` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].seeds[slot_id].contribution_velocity` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 418 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvDetailScreen SeedCard | `widgets/env_detail_screen.py` (lines 186-194) | "Trend: ↗ improving" / "↘ declining" / "--" |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — EMA computation tracks contribution changes
- [x] **Transport works** — SeedState.contribution_velocity updated; telemetry carries value
- [x] **Schema field exists** — `SeedState.contribution_velocity: float = 0.0` at line 418
- [x] **Default is correct** — `0.0` is correct for stable/no trend
- [x] **Consumer reads it** — SeedCard._render_active() displays trend indicator
- [x] **Display is correct** — Format: "↗ improving" green, "↘ declining" yellow, "--" dim

### Display Code

```python
# From env_detail_screen.py lines 186-194
# Contribution velocity (trend indicator - always visible)
trend_text = Text("Trend: ")
if seed.contribution_velocity > DisplayThresholds.CONTRIBUTION_VELOCITY_EPSILON:
    trend_text.append("↗ improving", style="green")
elif seed.contribution_velocity < -DisplayThresholds.CONTRIBUTION_VELOCITY_EPSILON:
    trend_text.append("↘ declining", style="yellow")
else:
    trend_text.append("--", style="dim")
lines.append(trend_text)
```

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Contribution history | data | Sequence of contribution values |
| EMA computation | algorithm | Exponential moving average of deltas |
| accuracy_delta | metric | Raw contribution values for trend |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| SeedCard display | display | Shows trend direction |
| Prune decisions | training | Declining seeds may be pruned |
| Fossilization gate | lifecycle | Improving trend supports fossilization |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation for EnvDetailScreen SeedCard |

---

## 8. Notes

> **Threshold:** DisplayThresholds.CONTRIBUTION_VELOCITY_EPSILON = 0.01
> - Values above 0.01: "↗ improving" in green
> - Values below -0.01: "↘ declining" in yellow
> - Values in [-0.01, 0.01]: "--" dim (stable)
>
> **Display Context:** Always visible in SeedCard. Shows trend direction rather than raw numeric value for quick visual assessment.
>
> **EMA Smoothing:** Uses exponential moving average to reduce noise from epoch-to-epoch fluctuations. Provides a smoothed trend signal rather than instantaneous changes.
>
> **Operational Significance:**
> - **Improving:** Seed is becoming more valuable — good candidate for fossilization
> - **Declining:** Seed is losing effectiveness — may need pruning or intervention
> - **Stable:** Seed has reached equilibrium — evaluate based on absolute contribution

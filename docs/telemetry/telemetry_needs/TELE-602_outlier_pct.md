# Telemetry Record: [TELE-602] Outlier Percentage

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-602` |
| **Name** | Outlier Percentage |
| **Category** | `environment` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "What percentage of observations are outside 3 sigma? Is the observation distribution shifting unexpectedly, indicating potential data quality issues?"

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
| **Units** | percentage (0.0-1.0, displayed as X.X%) |
| **Range** | `[0.0, 1.0]` |
| **Precision** | 1 decimal place for display (X.X%) |
| **Default** | `0.0` |

### Semantic Meaning

> The percentage of observations that fall outside the 3-sigma range (mean ± 3×std) of the observation distribution.
>
> High outlier rate (>10%) indicates potential observation distribution shift, which can:
> - Destabilize value function learning (OOD regression)
> - Cause action selection issues if policy was trained on different distribution
> - Signal environment changes or bugs in observation generation
>
> Calculation approach (expected):
> ```
> outlier_pct = (count of |obs - mean| > 3*std) / total_observations
> ```

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `outlier_pct <= 0.05` | Normal observation distribution (≤5%) |
| **Warning** | `0.05 < outlier_pct <= 0.10` | Elevated outliers, monitor closely (5-10%) |
| **Critical** | `outlier_pct > 0.10` | Significant distribution shift, immediate attention (>10%) |

**Threshold Source:** `health_status_panel.py::_get_outlier_status()` (lines 341-347)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Observation batch processing during training (ops telemetry path) |
| **File** | `/home/john/esper-lite/src/esper/simic/training/vectorized_trainer.py` |
| **Function/Method** | `compute_observation_stats()` (invoked after `obs_normalizer.normalize`) |
| **Line(s)** | Near the per-step observation normalization block |

**Emission Path:** `compute_observation_stats()` (in
`/home/john/esper-lite/src/esper/simic/telemetry/observation_stats.py`)
produces `ObservationStatsTelemetry`, which is attached to
`EpochCompletedPayload` in `/home/john/esper-lite/src/esper/simic/telemetry/emitters.py`.

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `EpochCompletedPayload.observation_stats` attached per env | `simic/telemetry/emitters.py` |
| **2. Collection** | `EpochCompletedPayload` carries `observation_stats` | `leyline/telemetry.py` |
| **3. Aggregation** | `_handle_epoch_completed` updates `ObservationStats` | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Written to `snapshot.observation_stats.outlier_pct` | `karn/sanctum/schema.py` |

```
[compute_observation_stats] --> [EpochCompletedPayload.observation_stats] --> [SanctumAggregator] --> [SanctumSnapshot.observation_stats.outlier_pct]
```

**Current Implementation:** Outlier percentage is computed per-step and carried
through epoch telemetry into the Sanctum snapshot.

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `ObservationStats` |
| **Field** | `outlier_pct` |
| **Path from SanctumSnapshot** | `snapshot.observation_stats.outlier_pct` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 216 |

**Definition:**
```python
@dataclass
class ObservationStats:
    """Observation space health metrics."""
    # ... other fields ...
    outlier_pct: float = 0.0  # % of observations outside 3σ
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| HealthStatusPanel | `src/esper/karn/sanctum/widgets/tamiyo_brain/health_status_panel.py` | Displays as "Out:X.X%" with threshold-based coloring (lines 324-347) |

**Display Logic:**
```python
def _get_outlier_status(self, outlier_pct: float) -> str:
    """Check if outlier percentage is healthy."""
    if outlier_pct > 0.1:  # >10% outliers is critical
        return "critical"
    if outlier_pct > 0.05:  # >5% is warning
        return "warning"
    return "ok"

# Rendered as:
f"Out:{obs.outlier_pct:.1%}"  # Formatted as X.X%
```

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `compute_observation_stats()` in `simic/telemetry/observation_stats.py`
- [x] **Transport works** — `EpochCompletedPayload.observation_stats` carries the stats
- [x] **Schema field exists** — Field defined in dataclass — `ObservationStats.outlier_pct`
- [x] **Default is correct** — Field has appropriate default — `0.0`
- [x] **Consumer reads it** — Widget accesses the field — `HealthStatusPanel._render_observation_stats()`
- [x] **Display is correct** — Value renders as expected — Formatted as "Out:X.X%" with threshold coloring
- [x] **Thresholds applied** — Color coding matches spec — `_get_outlier_status()` implements 5%/10% thresholds

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | — | — | `[ ]` |
| Unit (aggregator) | — | — | `[ ]` |
| Integration (end-to-end) | `tests/telemetry/test_environment_metrics.py` | `TestTELE602OutlierPct::test_outlier_pct_computed_from_batch_observations` | `[x]` |
| Visual (TUI snapshot) | — | Manual verification | `[ ]` |

### Manual Verification Steps

1. `uv run esper ppo --episodes 100`
2. Open Sanctum TUI (should open automatically or via `uv run sanctum`)
3. Navigate to HealthStatusPanel
4. Verify "Out:X.X%" displays in Obs Health row
5. Trigger high-outlier scenario to verify warning/critical coloring:
   - Manually scale observation batch to introduce outliers
   - Verify coloring changes: green (≤5%) → yellow (5-10%) → red (>10%)

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Observation batch | data | Raw observation vectors from environment |
| Batch statistics | computation | Mean and standard deviation of batch observations |
| Feature grouping | config | Understanding of slot/host/context feature boundaries |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| HealthStatusPanel | display | Drives status coloring for "Obs Health" row |
| Distribution shift detection | system | Could trigger alerts if outlier rate spikes |
| Observation normalization monitoring | diagnostic | Tracks if normalization layer is functioning correctly |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Initial telemetry record creation; identified wiring gap |
| 2026-01-07 | Claude Code | Wired via EpochCompletedPayload observation_stats and added integration test |

---

## 8. Notes

> **WIRING STATUS:** Outlier percentage is computed in
> `compute_observation_stats()` and propagated via
> `EpochCompletedPayload.observation_stats` into the Sanctum snapshot.
>
> **Feature Definition Notes:**
> - **3-sigma rule:** ~99.73% of values in a normal distribution fall within 3σ, so 0.27% is baseline expected outlier rate for truly normal data
> - **Observation grouping:** Stats are computed over per-feature means/stds; slot/host/context feature groups are tracked separately for context
> - **Running normalization:** Related field `normalization_drift` indicates whether normalization stats are drifting
> - **NaN/Inf handling:** Critical issues (NaN or Inf) bypass outlier checks in the widget display
>
> **Design Rationale:**
> High outlier percentages typically indicate:
> - Environment behavior has changed (e.g., new agent policies found unexpected paths)
> - Observation space has shifted (e.g., task variant active that wasn't seen in training)
> - Data quality issues (e.g., sensors saturated or malfunctioning)
> - Numerical precision loss (e.g., feature scaling misconfiguration)
>
> Catching this early prevents cascading failures in value function learning and policy gradient computation.

# Telemetry Record: [TELE-602] Outlier Percentage

> **Status:** `[ ] Planned` `[x] In Progress` `[ ] Wired` `[ ] Tested` `[ ] Verified`

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
| **Origin** | Observation batch processing during training (simic or governor) |
| **File** | `[NOT FOUND]` — Observation statistics computation not yet implemented |
| **Function/Method** | `[NOT FOUND]` — Expected in simic batch processing or Governor |
| **Line(s)** | `[UNKNOWN]` |

```python
# EXPECTED LOGIC (not yet implemented):
# For each observation batch [N, obs_dim]:
# 1. Compute mean and std across batch
# 2. Count values where |obs - mean| > 3 * std
# 3. Divide count by total observations to get outlier_pct
outlier_count = ((obs - obs.mean()).abs() > 3 * obs.std()).sum()
outlier_pct = float(outlier_count) / len(obs)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `[NOT FOUND]` — No emitter exists yet | simic/telemetry/emitters.py |
| **2. Collection** | `[NOT FOUND]` — No event payload defined | leyline/telemetry.py |
| **3. Aggregation** | Stub: returns default ObservationStats | `karn/sanctum/aggregator.py` (line 538) |
| **4. Delivery** | Written to `snapshot.observation_stats.outlier_pct` | `karn/sanctum/schema.py` |

```
[Observation Batch] --> [NOT IMPLEMENTED] --> [Aggregator Stub] --> [SanctumSnapshot.observation_stats.outlier_pct]
```

**Current Implementation Gap:** The aggregator creates an empty `ObservationStats()` with all fields at default values. No computation or event emission is wired.

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

- [ ] **Emitter exists** — Code computes and emits this value — `NOT FOUND`
- [ ] **Transport works** — Value reaches aggregator — `NOT IMPLEMENTED`
- [x] **Schema field exists** — Field defined in dataclass — `ObservationStats.outlier_pct`
- [x] **Default is correct** — Field has appropriate default — `0.0` (stub value)
- [x] **Consumer reads it** — Widget accesses the field — `HealthStatusPanel._render_observation_stats()`
- [x] **Display is correct** — Value renders as expected — Formatted as "Out:X.X%" with threshold coloring
- [x] **Thresholds applied** — Color coding matches spec — `_get_outlier_status()` implements 5%/10% thresholds

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `[NOT FOUND]` | `test_outlier_pct_computed` | `[ ]` |
| Unit (aggregator) | `[NOT FOUND]` | `test_observation_stats_populated` | `[ ]` |
| Integration (end-to-end) | `[NOT FOUND]` | `test_outlier_pct_reaches_tui` | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification | `[ ]` |

### Manual Verification Steps

1. **When emitter is implemented:** `uv run esper ppo --episodes 100`
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

---

## 8. Notes

> **WIRING GAP IDENTIFIED:** This metric is defined in the schema and has a consumer widget, but **no emitter or aggregation path exists**. The value always defaults to `0.0` because `ObservationStats()` is instantiated with default values in the aggregator (line 538 of aggregator.py).
>
> **Next Steps to Wire:**
> 1. **Add observation stats computation** in simic batch processing (likely in `simic/telemetry/` or `tolaria/governor.py`)
> 2. **Define telemetry event** in `leyline/telemetry.py` (e.g., `OBSERVATION_STATS_COMPUTED`)
> 3. **Add emitter** to compute and emit observation stats after each batch
> 4. **Add aggregator handler** to receive event and populate `ObservationStats` fields
> 5. **Add unit/integration tests** to verify end-to-end flow
> 6. **Manual testing** with synthetic outlier injection to verify thresholds
>
> **Feature Definition Notes:**
> - **3-sigma rule:** By definition, ~99.73% of values in a normal distribution fall within 3σ, so 0.27% is baseline expected outlier rate for truly normal data
> - **Observation grouping:** The schema includes separate tracking for `slot_features`, `host_features`, and `context_features` statistics, suggesting outlier detection may need to be computed per-feature-group rather than globally
> - **Running normalization:** Related field `normalization_drift` is also stubbed and may indicate whether observation normalization is drifting from training distribution
> - **NaN/Inf handling:** Critical issues (NaN or Inf) bypass outlier checks in the widget display, suggesting they're higher priority diagnostics
>
> **Design Rationale:**
> High outlier percentages typically indicate:
> - Environment behavior has changed (e.g., new agent policies found unexpected paths)
> - Observation space has shifted (e.g., task variant active that wasn't seen in training)
> - Data quality issues (e.g., sensors saturated or malfunctioning)
> - Numerical precision loss (e.g., feature scaling misconfiguration)
>
> Catching this early prevents cascading failures in value function learning and policy gradient computation.

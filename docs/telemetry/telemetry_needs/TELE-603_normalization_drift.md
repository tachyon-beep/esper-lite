# Telemetry Record: [TELE-603] Normalization Drift

> **Status:** `[ ] Planned` `[ ] In Progress` `[x] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-603` |
| **Name** | Normalization Drift |
| **Category** | `environment` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "How much has observation running mean/std shifted from initial values? Large drift indicates environment distribution change that could degrade policy performance."

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

### When is this information needed?

- [x] Real-time (every batch/epoch)
- [ ] Periodic (every N episodes)
- [x] On-demand (when investigating issues)
- [ ] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `float` |
| **Units** | Standard deviations (σ) |
| **Range** | `[0.0, inf)` |
| **Precision** | 2 decimal places for display |
| **Default** | `0.0` |

### Semantic Meaning

> Normalization drift measures how much the running mean and/or running standard deviation of observations have shifted relative to initial values. This is expressed in units of standard deviations to make the metric scale-invariant.
>
> The running mean and std are maintained via Welford's online algorithm (or EMA for long training) in the `RunningMeanStd` class. Drift is computed as the magnitude of change:
>
> `drift = max(|current_mean - initial_mean|/std, |current_std - initial_std|/initial_std)`
>
> Large drift (>2σ or >100% change in std) indicates the environment observation distribution has shifted significantly from training start, which can cause:
> - Normalization layer to apply incorrect scaling
> - Value function to see observations outside its training distribution
> - Policy to experience out-of-distribution inputs

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `drift <= 1.0` | Observation distribution stable within 1σ |
| **Warning** | `1.0 < drift <= 2.0` | Moderate distribution shift detected (1-2σ) |
| **Critical** | `drift > 2.0` | Severe distribution shift (>2σ) — immediate attention required |

**Threshold Source:** Hardcoded in `HealthStatusPanel._get_drift_status()`:
- `DRIFT_CRITICAL = 2.0`
- `DRIFT_WARNING = 1.0`

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Observation normalization statistics during episode rollout |
| **File** | `/home/john/esper-lite/src/esper/simic/control/normalization.py` |
| **Class/Method** | `RunningMeanStd._update_from_moments()` (lines 80-121) |
| **Line(s)** | Lines 80-121 |

```python
# RunningMeanStd maintains running mean/std via Welford's algorithm
# _update_from_moments() updates statistics with each batch:
#   delta = batch_mean - self.mean
#   self.mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
#   self.var = ...
```

**Note:** The drift metric itself must be computed by comparing `RunningMeanStd.mean` and `RunningMeanStd.var` against some reference (initial values or baseline). This computation location is not yet identified — see "Currently Not Implemented" below.

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `[UNKNOWN]` — not yet emitted | `src/esper/simic/telemetry/emitters.py` |
| **2. Collection** | `[UNKNOWN]` — no telemetry event carries drift | `leyline/telemetry.py` |
| **3. Aggregation** | Stub (default values only) | `src/esper/karn/sanctum/aggregator.py` (line 538) |
| **4. Delivery** | Written to `snapshot.observation_stats.normalization_drift` | `src/esper/karn/sanctum/schema.py` |

```
[RunningMeanStd] --[MISSING EMISSION]--> [Aggregator] --> [ObservationStats.normalization_drift]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `ObservationStats` |
| **Field** | `normalization_drift` |
| **Path from SanctumSnapshot** | `snapshot.observation_stats.normalization_drift` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | ~223 |

**Field Definition:**
```python
@dataclass
class ObservationStats:
    """Observation space health metrics."""
    # ... other fields ...
    normalization_drift: float = 0.0  # How much running mean/std has shifted
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| HealthStatusPanel | `src/esper/karn/sanctum/widgets/tamiyo_brain/health_status_panel.py` (lines 307-355) | Displayed in "Obs Health" row as "Drift:X.XX" with color coding |

**Consumer Code:**
```python
def _render_observation_stats(self) -> Text:
    """Render observation space health indicators."""
    obs = self._snapshot.observation_stats
    drift_status = self._get_drift_status(obs.normalization_drift)
    result.append(
        f"Drift:{obs.normalization_drift:.2f}",
        style=self._status_style(drift_status),
    )

def _get_drift_status(self, drift: float) -> str:
    """Check if normalization drift is healthy."""
    if drift > 2.0:  # >2σ drift is critical
        return "critical"
    if drift > 1.0:  # >1σ is warning
        return "warning"
    return "ok"
```

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `RunningMeanStd` maintains running stats, but drift computation missing
- [ ] **Transport works** — No telemetry event emits observation stats (line 537 comment: "telemetry not yet wired")
- [x] **Schema field exists** — `ObservationStats.normalization_drift: float = 0.0` (line 223)
- [x] **Default is correct** — `0.0` appropriate when no drift computation occurs
- [x] **Consumer reads it** — `HealthStatusPanel._render_observation_stats()` accesses field (line 335)
- [x] **Display is correct** — Formatted as "Drift:X.XX" with status color coding (lines 335-337)
- [x] **Thresholds applied** — Color coding in `_get_drift_status()` matches spec (lines 349-355)

### Currently Not Implemented

**Critical Gap:** The drift computation itself does not exist. The metric is:
- Defined in schema ✓
- Consumed by widget ✓
- **But never computed or emitted** ✗

To wire this metric, the following must be implemented:
1. **Compute drift in RunningMeanStd or observation collector:**
   - Track initial mean/std at `RunningMeanStd.__init__()`
   - On each epoch/batch, compute `drift = max(|mean_delta|/std, |std_delta|/initial_std)`
   - Store drift value accessible to telemetry layer

2. **Emit observation stats in telemetry:**
   - Create `ObservationStatsPayload` in `leyline/telemetry.py`
   - Emit from `VectorizedEmitter` or batch completion handler
   - Include: `nan_count`, `inf_count`, `outlier_pct`, `normalization_drift`

3. **Aggregate in SanctumAggregator:**
   - Replace stub `ObservationStats()` (line 538) with aggregated data
   - Compute rolling stats or use latest value from telemetry events

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_normalization.py` | `test_normalization_drift_computation` | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_observation_stats_populated` | `[ ]` |
| Integration (end-to-end) | `tests/integration/test_telemetry_flow.py` | `test_observation_stats_reaches_tui` | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification | `[ ]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 100`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe HealthStatusPanel "Obs Health" row
4. Verify "Drift:X.XX" value appears (currently will be "Drift:0.00" until wired)
5. Artificially perturb observation distribution (environment or preprocessing) to trigger high drift
6. Verify color coding: green ≤1.0, yellow 1.0-2.0, red >2.0

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `RunningMeanStd.mean / .var` | state | Requires access to running statistics |
| Initial observation distribution | baseline | Drift is relative to initial mean/std — need to capture at start |
| Observation preprocessing | event | Depends on how observations enter the normalizer |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| HealthStatusPanel | display | Displays drift in "Obs Health" row |
| Automated intervention | system | Could trigger resets if drift exceeds threshold (not yet implemented) |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Initial record created — metric wired in display but computation missing |
| | | Identified gap: drift computed by RunningMeanStd but never emitted to telemetry |

---

## 8. Notes

> **Design Decision:** Drift is expressed in standard deviations rather than absolute change to make the metric scale-invariant. Observations with different ranges can be compared meaningfully.

> **Known Issue:** Currently a stub metric. The field exists in schema and is consumed by HealthStatusPanel, but:
> - No code computes drift during training
> - ObservationStats is always initialized with defaults (line 538 in aggregator.py has comment: "telemetry not yet wired")
> - Displays as "Drift:0.00" always

> **Future Implementation:** When wiring observation stats telemetry:
> 1. Drift should be per-feature-group (slot features, host features, context features) not global
> 2. Consider exponential weighting to give recent drift more weight
> 3. Add option to pause training if drift exceeds critical threshold (environmental shift detector)
> 4. Consider storing drift history for trend detection (e.g., gradual vs sudden shift)

> **Design Question:** Should drift be computed as:
> - (a) `max(|Δmean|/std, |Δstd|/std)` — magnitude of change in both mean and variance
> - (b) `|Δmean|/initial_std` only — only change in mean, not variance
> - (c) Mahalanobis distance or other statistical distance metric
>
> Recommend (a) because environment distribution shifts typically affect both mean and variance.

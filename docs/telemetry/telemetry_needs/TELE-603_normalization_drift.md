# Telemetry Record: [TELE-603] Normalization Drift

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[ ] Verified`

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
| **Units** | Normalized observation units (mean absolute delta) |
| **Range** | `[0.0, inf)` |
| **Precision** | 2 decimal places for display |
| **Default** | `0.0` |

### Semantic Meaning

> Normalization drift measures how much the running mean of observations has
> shifted relative to the initial mean captured at rollout start. This is
> computed as the mean absolute delta between the current normalizer mean and
> the initial mean:
>
> `drift = mean(abs(current_mean - initial_mean))`
>
> Large drift (>2.0 in normalized feature units) indicates the environment
> observation distribution has shifted significantly from training start,
> which can cause:
> - Normalization layer to apply incorrect scaling
> - Value function to see observations outside its training distribution
> - Policy to experience out-of-distribution inputs

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `drift <= 1.0` | Observation distribution stable (<= 1.0 normalized units) |
| **Warning** | `1.0 < drift <= 2.0` | Moderate distribution shift (1.0-2.0 normalized units) |
| **Critical** | `drift > 2.0` | Severe distribution shift (>2.0 normalized units) |

**Threshold Source:** Hardcoded in `HealthStatusPanel._get_drift_status()`:
- `DRIFT_CRITICAL = 2.0`
- `DRIFT_WARNING = 1.0`

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Observation normalization statistics during episode rollout |
| **File** | `/home/john/esper-lite/src/esper/simic/training/vectorized_trainer.py` |
| **Class/Method** | `compute_observation_stats()` (invoked after `obs_normalizer.normalize`) |
| **Line(s)** | Near the per-step observation normalization block |

**Emission Path:** `compute_observation_stats()` (in
`/home/john/esper-lite/src/esper/simic/telemetry/observation_stats.py`)
computes `normalization_drift` using `obs_normalizer.mean` and the initial
normalizer mean captured at rollout start. The result is attached to
`EpochCompletedPayload` in `/home/john/esper-lite/src/esper/simic/telemetry/emitters.py`.

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `EpochCompletedPayload.observation_stats` attached per env | `src/esper/simic/telemetry/emitters.py` |
| **2. Collection** | `EpochCompletedPayload` carries `observation_stats` | `leyline/telemetry.py` |
| **3. Aggregation** | `_handle_epoch_completed` updates `ObservationStats` | `src/esper/karn/sanctum/aggregator.py` |
| **4. Delivery** | Written to `snapshot.observation_stats.normalization_drift` | `src/esper/karn/sanctum/schema.py` |

```
[compute_observation_stats] --> [EpochCompletedPayload.observation_stats] --> [SanctumAggregator] --> [ObservationStats.normalization_drift]
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

- [x] **Emitter exists** — `compute_observation_stats()` computes drift from normalizer mean
- [x] **Transport works** — `EpochCompletedPayload.observation_stats` carries the stats
- [x] **Schema field exists** — `ObservationStats.normalization_drift: float = 0.0` (line 223)
- [x] **Default is correct** — `0.0` appropriate when no drift computation occurs
- [x] **Consumer reads it** — `HealthStatusPanel._render_observation_stats()` accesses field (line 335)
- [x] **Display is correct** — Formatted as "Drift:X.XX" with status color coding (lines 335-337)
- [x] **Thresholds applied** — Color coding in `_get_drift_status()` matches spec (lines 349-355)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | — | — | `[ ]` |
| Unit (aggregator) | — | — | `[ ]` |
| Integration (end-to-end) | `tests/telemetry/test_environment_metrics.py` | `TestTELE603NormalizationDrift::test_normalization_drift_tracks_mean_std_shift` | `[x]` |
| Visual (TUI snapshot) | — | Manual verification | `[ ]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 100`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe HealthStatusPanel "Obs Health" row
4. Verify "Drift:X.XX" value appears
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
| 2026-01-07 | Claude Code | Wired via EpochCompletedPayload observation_stats and added integration test |

---

## 8. Notes

> **Design Decision:** Drift is computed as the mean absolute delta between the
> current normalizer mean and the initial mean captured at rollout start. If
> we want sigma-normalized drift, update `compute_observation_stats()` to divide
> by the running standard deviation and adjust UI thresholds accordingly.

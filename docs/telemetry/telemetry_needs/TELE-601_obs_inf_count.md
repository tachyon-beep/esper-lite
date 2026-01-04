# Telemetry Record: [TELE-601] Observation Inf Count

> **Status:** `[ ] Planned` `[ ] In Progress` `[x] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-601` |
| **Name** | Observation Inf Count |
| **Category** | `environment` |
| **Priority** | `P0-critical` |

## 2. Purpose

### What question does this answer?

> "Are there Inf values in observation tensors? Inf values indicate observation overflow and signal imminent NaN propagation."

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
| **Type** | `int` |
| **Units** | count (number of Inf values in observation tensor) |
| **Range** | `[0, ∞)` — non-negative integers |
| **Precision** | integer |
| **Default** | `0` |

### Semantic Meaning

> Count of Inf (positive or negative infinity) values detected in observation tensors during the last observation collection.
>
> Inf values in observations indicate:
> - Numerical overflow in feature computation
> - Division by zero (1/0 = Inf)
> - log(0) or exp(huge_value) in preprocessing
> - Uninitialized or corrupted feature values
>
> Unlike NaN, Inf can still propagate through networks (Inf * weight = Inf),
> but it frequently precedes NaN due to downstream operations (log(Inf) = Inf, sqrt(Inf) = Inf, etc.).

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `inf_count == 0` | No Inf values in observations |
| **Warning** | `inf_count > 0` (rare) | Inf detected, investigate immediately |
| **Critical** | `inf_count > 0` (persistent) | Inf overflow - training will fail soon |

**Note:** Unlike most metrics, ANY value > 0 is critical. Inf in observations is always a bug—there is no "acceptable" amount of infinity in the input space.

**Threshold Source:** Health panel logic (line 319 in health_status_panel.py):
```python
if obs.nan_count > 0 or obs.inf_count > 0:
    # RED BOLD: Critical indicator
```

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Observation collection during environment step |
| **File** | `[NOT YET IMPLEMENTED]` — likely `src/esper/tolaria/environment.py` or vec env wrapper |
| **Function/Method** | `[NOT YET IMPLEMENTED]` — likely in step() or observation preprocessing |
| **Line(s)** | `[NOT YET FOUND]` |

**STATUS:** Observation inf/nan counting code does NOT YET EXIST in the codebase. The health panel expects it, but no emitter code has been wired.

```python
# TODO: Implement observation stat collection
# torch.isinf(observations).sum()
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `[NOT IMPLEMENTED]` — Needs event payload | `simic/telemetry/emitters.py` (estimate) |
| **2. Collection** | `[NOT IMPLEMENTED]` — Observation stats event | `leyline/telemetry.py` |
| **3. Aggregation** | `[STUB]` — Currently hardcoded empty stats | `karn/sanctum/aggregator.py:538` |
| **4. Delivery** | `[STUB]` — Writes empty ObservationStats | `karn/sanctum/schema.py:1380` |

```
[Observation Collection] --> [VectorizedEmitter] --> [event] --> [SanctumAggregator] --> [SanctumSnapshot.observation_stats]
```

**KEY FINDING:** Line 537-538 in aggregator.py shows explicit stub comment:
```python
# Stub observation and episode stats (telemetry not yet wired)
observation_stats = ObservationStats()
```

This confirms the entire data flow is NOT YET WIRED.

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `ObservationStats` |
| **Field** | `inf_count` |
| **Path from SanctumSnapshot** | `snapshot.observation_stats.inf_count` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 220 |

**Schema Definition (line 200-224):**
```python
@dataclass
class ObservationStats:
    """Observation space health metrics.

    Tracks feature statistics to catch input distribution issues
    before they propagate to NaN gradients.
    """
    # Per-feature-group statistics (mean/std across batch)
    slot_features_mean: float = 0.0
    slot_features_std: float = 0.0
    host_features_mean: float = 0.0
    host_features_std: float = 0.0
    context_features_mean: float = 0.0  # Epoch progress, action history, etc.
    context_features_std: float = 0.0

    # Outlier detection
    outlier_pct: float = 0.0  # % of observations outside 3σ

    # Numerical health
    nan_count: int = 0  # NaN values detected in observations
    inf_count: int = 0  # Inf values detected in observations

    # Normalization drift (running stats divergence)
    normalization_drift: float = 0.0  # How much running mean/std has shifted
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| HealthStatusPanel | `widgets/tamiyo_brain/health_status_panel.py:315-339` | "Obs Health" row with NaN/Inf check (line 319-321) |

**Consumer Code (line 307-339):**
```python
def _render_observation_stats(self) -> Text:
    """Render observation space health indicators."""
    if self._snapshot is None:
        return Text()

    obs = self._snapshot.observation_stats
    result = Text()

    # Check for NaN/Inf first (critical issue)
    if obs.nan_count > 0 or obs.inf_count > 0:
        result.append("Obs Health   ", style="dim")
        result.append(f"NaN:{obs.nan_count} Inf:{obs.inf_count}", style="red bold")
        return result

    # ... rest of display logic
```

---

## 5. Wiring Verification

### Checklist

- [ ] **Emitter exists** — Code does NOT YET compute and emit this value
- [ ] **Transport works** — Event payload does NOT YET include inf_count
- [x] **Schema field exists** — Field defined in ObservationStats (line 220)
- [x] **Default is correct** — Field has appropriate default (0)
- [x] **Consumer reads it** — HealthStatusPanel accesses the field (line 319)
- [x] **Display is correct** — Value renders as "Inf:X" in red bold
- [x] **Thresholds applied** — Any value > 0 triggers critical display

**SUMMARY:** 4/7 wiring stages complete. Missing: emitter implementation.

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/tolaria/test_observation_stats.py` | `test_inf_count_collected` | `[ ]` NOT IMPLEMENTED |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_observation_stats_aggregated` | `[ ]` STUB ONLY |
| Integration (end-to-end) | `tests/integration/test_telemetry_flow.py` | `test_inf_count_reaches_tui` | `[ ]` NOT IMPLEMENTED |
| Visual (TUI snapshot) | — | Manual verification | `[ ]` BLOCKED BY EMITTER |

**NOTE:** HealthStatusPanel tests likely exist but would not catch inf_count since it's always 0.

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Inject Inf into observations (modify environment preprocessing to trigger)
3. Observe HealthStatusPanel "Obs Health" row
4. Verify "Inf:X" appears in red bold when X > 0
5. Clear Inf issue and verify display returns to normal

**BLOCKER:** Cannot currently test end-to-end without implementing emitter first.

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Observation tensor creation | computation | VectorizedEnv step() output must be inspected |
| Observation normalization/preprocessing | computation | Must check AFTER preprocessing but BEFORE network input |
| Feature computation pipeline | computation | torch.isinf() must be called on finalized observation tensor |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| HealthStatusPanel rendering | display | Shows "Obs Health" row when inf_count > 0 |
| Training diagnostics | monitoring | Developer alert for observation overflow issues |
| Auto-intervention (future) | system | Could trigger governor rollback if persistent Inf detected |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Telemetry Audit | Created record; identified as NOT YET WIRED (stub only) |
| | | Aggregator explicitly stubs observation_stats (line 537) |
| | | Widget expects field, emitter does not yet provide it |

---

## 8. Notes

> **WIRING GAP IDENTIFIED:** This metric is a classic case of schema-first design without backend implementation. The widget (HealthStatusPanel) expects observation_stats, the schema (ObservationStats) is defined, but the emitter code that populates it does not exist.
>
> **IMPLEMENTATION PLAN:**
> 1. Add observation stat collection in environment step (likely tolaria/environment.py)
> 2. Compute inf_count = torch.isinf(observations).sum().item()
> 3. Emit ObservationStatsPayload or add field to EPOCH_COMPLETED payload
> 4. Handle event in SanctumAggregator.process_event() (remove stub)
> 5. Add tests for collection and aggregation
>
> **TIMING:** Observation stats should be collected every epoch, same cadence as EPOCH_COMPLETED (not every step, would be too expensive).
>
> **PERFORMANCE NOTE:** torch.isinf() is cheap (single pass over tensor), but should be conditional on debug mode or telemetry config to avoid hot-path overhead.
>
> **RELATED TELEMETRY:** Compare with TELE-301 (inf_grad_count) which has similar pattern. That metric IS wired and working, so can use it as reference implementation.
>
> **DIFFERENCE FROM TELE-301:**
> - TELE-301: Checks gradients during backward pass (debug_telemetry.py)
> - TELE-601: Should check observation tensor right after env.step() returns
> - TELE-301 has explicit gradient collection code
> - TELE-601 is currently a stub awaiting implementation

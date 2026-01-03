# Telemetry Record: [TELE-644] Environment Growth Ratio

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-644` |
| **Name** | Environment Growth Ratio |
| **Category** | `env` |
| **Priority** | `P1-high` |

## 2. Purpose

### What question does this answer?

> "How much larger is this environment's model compared to the original host model?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [x] Automated system (alerts/intervention)

### When is this information needed?

- [x] Real-time (every batch/epoch)
- [ ] Periodic (every N episodes)
- [ ] On-demand (when investigating issues)
- [x] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `float` (computed property) |
| **Units** | ratio (multiplier) |
| **Range** | `[1.0, inf)` typically `[1.0, 10.0]` |
| **Precision** | 2 decimal places for display |
| **Default** | `1.0` (no growth) |

### Semantic Meaning

> Growth ratio measures model size expansion from fossilized seeds:
>
> - **Formula:** `(host_params + fossilized_params) / host_params`
> - **1.0x:** No fossilized seeds, original model size
> - **2.0x:** Model has doubled in size from fossilization
> - **5.0x+:** Significant growth, may impact inference cost
>
> This is a key efficiency metric - high accuracy with low growth ratio is ideal.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `ratio < 2.0` | Minimal growth (DEFAULT_GROWTH_RATIO_GREEN_MAX) |
| **Warning** | `2.0 <= ratio < 5.0` | Moderate growth (DEFAULT_GROWTH_RATIO_YELLOW_MAX) |
| **Critical** | `ratio >= 5.0` | Significant growth, efficiency concern |

**Threshold Source:** `src/esper/leyline/constants.py` — `DEFAULT_GROWTH_RATIO_GREEN_MAX`, `DEFAULT_GROWTH_RATIO_YELLOW_MAX`
**Display:** `src/esper/karn/sanctum/widgets/env_overview.py` — `_format_growth_ratio()` (lines 547-567)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Computed from host_params and fossilized_params |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Function/Method** | `EnvState.growth_ratio` property |
| **Line(s)** | 569-578 |

```python
@property
def growth_ratio(self) -> float:
    """Compute growth ratio: (host + fossilized) / host.

    Returns 1.0 if host_params is 0 (avoids division by zero).
    """
    if self.host_params <= 0:
        return 1.0
    return (self.host_params + self.fossilized_params) / self.host_params
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | host_params set at TRAINING_STARTED | `aggregator.py` (line 633) |
| **2. Collection** | fossilized_params accumulated via SEED_FOSSILIZED | `aggregator.py` (line 1080) |
| **3. Aggregation** | Ratio computed on-demand via property | `schema.py` (lines 569-578) |
| **4. Delivery** | Available at `snapshot.envs[env_id].growth_ratio` | `schema.py` (lines 569-578) |

```
[host_params + fossilized_params]
  --property access-->
  [(host_params + fossilized_params) / host_params]
  --snapshot-->
  [SanctumSnapshot.envs[env_id].growth_ratio]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `EnvState` |
| **Field** | `growth_ratio` (property) |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].growth_ratio` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 569-578 |
| **Default Value** | `1.0` (computed when host_params=0 or fossilized_params=0) |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvOverview | `widgets/env_overview.py` (lines 547-567) | Growth column with color coding |
| EnvOverview (aggregate) | `widgets/env_overview.py` (lines 438-440, 451) | Mean growth across envs |
| BestRunRecord | `schema.py` (line 1241) | Captured for historical comparison |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — Computed property from host_params and fossilized_params
- [x] **Transport works** — Both input fields are wired correctly
- [x] **Schema field exists** — `EnvState.growth_ratio` property at lines 569-578
- [x] **Default is correct** — `1.0` when no fossilized params
- [x] **Consumer reads it** — EnvOverview._format_growth_ratio() accesses env.growth_ratio
- [x] **Display is correct** — Shown as "Nx" with color coding (green/yellow/red)
- [x] **Thresholds applied** — 2.0x and 5.0x from leyline constants

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (schema) | `tests/karn/sanctum/test_schema.py` | `test_growth_ratio_calculation` | `[ ]` |
| Unit (schema) | `tests/karn/sanctum/test_schema.py` | `test_growth_ratio_division_by_zero` | `[ ]` |
| Widget (EnvOverview) | `tests/karn/sanctum/widgets/test_env_overview.py` | Growth ratio formatting | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI
3. Observe Growth column — should show 1.0x initially
4. Wait for seeds to fossilize — growth ratio should increase
5. Verify color coding: green (<2.0x), yellow (2.0-5.0x), red (>=5.0x)

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| host_params (TELE-642) | metric | Denominator for ratio |
| fossilized_params (TELE-643) | metric | Numerator addition |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EnvOverview Growth column | display | Primary efficiency indicator |
| Aggregate mean growth | computation | Mean across all envs |
| BestRunRecord.growth_ratio | data | Captured for leaderboard |
| Pareto analysis | analysis | Accuracy vs growth tradeoff |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** Thresholds (2.0x and 5.0x) are configurable via leyline constants because "acceptable" growth varies by use case. Small models can tolerate higher ratios; large production models need stricter limits.
>
> **Threshold Rationale:** 2.0x as green limit is generous because small host models (e.g., CIFAR10 demo models) can easily double with a single attention seed. Production models would use stricter thresholds.
>
> **Division Safety:** Returns 1.0 if host_params <= 0 to avoid division by zero. This should only occur before TRAINING_STARTED.
>
> **Wiring Status:** Fully wired as computed property from two fully-wired input fields.

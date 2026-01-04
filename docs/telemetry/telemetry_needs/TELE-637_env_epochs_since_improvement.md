# Telemetry Record: [TELE-637] Epochs Since Improvement

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-637` |
| **Name** | Epochs Since Improvement |
| **Category** | `env` |
| **Priority** | `P1-high` |

## 2. Purpose

### What question does this answer?

> "How many epochs have passed since this environment achieved a new best accuracy?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [ ] Researcher (analysis)
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
| **Units** | epochs (count) |
| **Range** | `[0, max_epochs]` typically 0-75 |
| **Precision** | Integer |
| **Default** | `0` |

### Semantic Meaning

> Epochs since improvement counts how many epochs have passed since the last time a new best_accuracy was achieved. This is the primary stall detection metric:
>
> - **0:** Just achieved a new best (actively improving)
> - **1-5:** Recently improved, normal exploration
> - **6-15:** Starting to stall, may need intervention
> - **>15:** Significantly stalled, likely needs policy adjustment
>
> The counter is reset to 0 whenever accuracy exceeds best_accuracy, and incremented each epoch otherwise.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `value == 0` | Currently improving (just hit new best) |
| **Good** | `value <= 5` | Recent improvement, normal training |
| **Warning** | `6 <= value <= 15` | Stagnating, needs attention |
| **Critical** | `value > 15` | Severely stalled, intervention needed |

**Threshold Source:** `src/esper/karn/sanctum/widgets/env_overview.py` — `_format_momentum_epochs()` (lines 798-833)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Computed in EnvState.add_accuracy() |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Function/Method** | `EnvState.add_accuracy()` |
| **Line(s)** | 599, 648 |

```python
def add_accuracy(self, accuracy: float, epoch: int, episode: int = 0) -> None:
    ...
    if accuracy > self.best_accuracy:
        ...
        self.epochs_since_improvement = 0
        ...
    else:
        self.epochs_since_improvement += 1
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Updated during add_accuracy() | `schema.py` (lines 599, 648) |
| **2. Collection** | Stored in EnvState.epochs_since_improvement | `schema.py` (line 532) |
| **3. Aggregation** | Included in EnvState, passed through snapshot | `aggregator.py` (line 559) |
| **4. Delivery** | Available at `snapshot.envs[env_id].epochs_since_improvement` | `schema.py` (line 532) |

```
[EnvState.add_accuracy(accuracy)]
  --if accuracy > best_accuracy-->
  [epochs_since_improvement = 0]
  --else-->
  [epochs_since_improvement += 1]
  --snapshot-->
  [SanctumSnapshot.envs[env_id].epochs_since_improvement]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `EnvState` |
| **Field** | `epochs_since_improvement` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].epochs_since_improvement` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 532 |
| **Default Value** | `0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvOverview | `widgets/env_overview.py` (lines 798-833) | Momentum column with color-coded staleness indicator |
| EnvOverview (aggregate) | `widgets/env_overview.py` (lines 469-471) | Mean staleness across all envs |
| EnvState._update_status() | `schema.py` (lines 696-701) | Used for stall status detection (>10 epochs triggers stalled) |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — EnvState.add_accuracy() updates epochs_since_improvement
- [x] **Transport works** — Counter is computed in-place during accuracy updates
- [x] **Schema field exists** — `EnvState.epochs_since_improvement: int = 0` at line 532
- [x] **Default is correct** — `0` at start of each episode
- [x] **Consumer reads it** — EnvOverview._format_momentum_epochs() reads the value
- [x] **Display is correct** — Color-coded with prefix: +0 (green), N (white/yellow/red based on status)
- [x] **Thresholds applied** — 5/15 thresholds for warning/critical coloring; 10 for stalled status

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (schema) | `tests/karn/sanctum/test_schema.py` | `test_epochs_since_improvement_tracking` | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_stall_counter_updates` | `[ ]` |
| Widget (EnvOverview) | `tests/karn/sanctum/widgets/test_env_overview.py` | Momentum formatting | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI
3. Observe Momentum column — should show "+0" when improving, then count up when stalled
4. Verify color coding: green (0), white (1-5), yellow (6-15), red (>15)
5. Note that coloring also depends on status (healthy envs show green even with high counts)

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| host_accuracy (TELE-635) | metric | Counter updates based on accuracy comparison |
| best_accuracy (TELE-636) | metric | Compared against to determine improvement |
| add_accuracy() calls | method | Triggers counter update |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EnvOverview Momentum column | display | Primary stall indicator |
| status computation | derived | >10 epochs triggers "stalled" status |
| Aggregate mean staleness | computation | Mean across all envs |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** The momentum display uses context-aware coloring. When env status is "excellent" or "healthy", high epoch counts are shown in green (stability is good). When status is "stalled" or "degraded", the same counts show in yellow/red (stuck in bad state is concerning).
>
> **Display Prefixes:** Fixed-width ASCII prefixes for alignment:
> - `+0` = currently improving (green)
> - ` N` = normal count (white/green)
> - `!N` = warning (yellow)
> - `xN` = critical (red)
>
> **Status Interaction:** epochs_since_improvement > 10 triggers hysteresis counter for "stalled" status (requires 3 consecutive epochs).
>
> **Wiring Status:** Fully wired and operational.

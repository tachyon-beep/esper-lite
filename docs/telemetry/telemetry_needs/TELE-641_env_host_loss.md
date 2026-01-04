# Telemetry Record: [TELE-641] Environment Host Loss

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-641` |
| **Name** | Environment Host Loss |
| **Category** | `env` |
| **Priority** | `P1-high` |

## 2. Purpose

### What question does this answer?

> "What is the current validation loss of the host model in this environment?"

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
| **Type** | `float` |
| **Units** | loss units (typically cross-entropy) |
| **Range** | `[0.0, inf)` but typically `[0.0, 5.0]` |
| **Precision** | 3 decimal places for display |
| **Default** | `0.0` |

### Semantic Meaning

> Host loss is the validation loss of the base model (with integrated seeds) at the current epoch. Used to detect overfitting and training issues:
>
> - **Low loss (<0.1):** Excellent convergence
> - **Normal loss (0.1-0.5):** Typical training range
> - **Elevated loss (0.5-1.0):** May indicate overfitting or issues
> - **High loss (>=1.0):** Significant problems, possible divergence
>
> Loss is inversely related to accuracy for classification tasks.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `loss < 0.1` | Very low loss, excellent convergence |
| **Good** | `0.1 <= loss < 0.5` | Normal training range |
| **Warning** | `0.5 <= loss < 1.0` | Elevated, possible overfitting |
| **Critical** | `loss >= 1.0` | High loss, significant issues |

**Threshold Source:** `src/esper/karn/sanctum/widgets/env_overview.py` — `_format_host_loss()` (lines 526-545)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | EPOCH_COMPLETED telemetry event |
| **File** | `/home/john/esper-lite/src/esper/leyline/telemetry.py` |
| **Function/Method** | `EpochCompletedPayload.val_loss` |
| **Line(s)** | (varies) |

```python
@dataclass
class EpochCompletedPayload:
    env_id: int
    inner_epoch: int
    val_accuracy: float
    val_loss: float
    seeds: dict[str, dict[str, Any]] | None = None
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Training loop emits EPOCH_COMPLETED with val_loss | `simic/training.py` |
| **2. Collection** | Aggregator extracts val_loss from payload | `aggregator.py` (lines 708, 711) |
| **3. Aggregation** | Stored in env.host_loss | `aggregator.py` (line 711) |
| **4. Delivery** | Available at `snapshot.envs[env_id].host_loss` | `schema.py` (line 453) |

```
[EpochCompletedPayload.val_loss]
  --EPOCH_COMPLETED-->
  [SanctumAggregator._handle_epoch_completed()]
  --env.host_loss = val_loss-->
  [EnvState.host_loss]
  --snapshot-->
  [SanctumSnapshot.envs[env_id].host_loss]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `EnvState` |
| **Field** | `host_loss` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].host_loss` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 453 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvOverview | `widgets/env_overview.py` (lines 526-545) | Loss column with color coding |
| EnvOverview (aggregate) | `widgets/env_overview.py` (lines 435-436, 450) | Mean loss across envs |
| BestRunRecord | `schema.py` (line 1264) | Captured for historical detail |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — EpochCompletedPayload includes val_loss field
- [x] **Transport works** — Aggregator extracts val_loss in _handle_epoch_completed()
- [x] **Schema field exists** — `EnvState.host_loss: float = 0.0` at line 453
- [x] **Default is correct** — `0.0` before any epochs complete
- [x] **Consumer reads it** — EnvOverview._format_host_loss() reads env.host_loss
- [x] **Display is correct** — Color-coded: green (<0.1), white (0.1-0.5), yellow (0.5-1.0), red (>=1.0)
- [x] **Thresholds applied** — 0.1, 0.5, 1.0 thresholds in _format_host_loss()

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_epoch_completed_updates_loss` | `[ ]` |
| Widget (EnvOverview) | `tests/karn/sanctum/widgets/test_env_overview.py` | Loss formatting | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI
3. Observe Loss column — should show decreasing loss values
4. Verify color coding: green (<0.1), white (0.1-0.5), yellow (0.5-1.0), red (>=1.0)
5. Verify aggregate row shows mean loss

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| EPOCH_COMPLETED events | event | Provides val_loss each epoch |
| Validation evaluation | computation | Must run validation to get loss |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EnvOverview Loss column | display | Per-env loss indicator |
| Aggregate mean loss | computation | Mean across all envs |
| BestRunRecord.host_loss | data | Captured for historical detail |
| Overfitting detection | analysis | High loss + high accuracy = overfit |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** Loss is shown with 3 decimal places to distinguish fine-grained differences in the <0.1 range where convergence matters most.
>
> **Overfitting Detection:** High loss combined with high accuracy may indicate memorization/overfitting. Low loss with low accuracy indicates underfitting.
>
> **Edge Case:** Loss of 0 (or very close) may indicate numerical issues or constant predictions. Displayed as dim dash if loss <= 0.
>
> **Wiring Status:** Fully wired via EPOCH_COMPLETED events.

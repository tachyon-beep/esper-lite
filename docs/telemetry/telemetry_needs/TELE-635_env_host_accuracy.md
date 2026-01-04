# Telemetry Record: [TELE-635] Environment Host Accuracy

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-635` |
| **Name** | Environment Host Accuracy |
| **Category** | `env` |
| **Priority** | `P0-critical` |

## 2. Purpose

### What question does this answer?

> "What is the current validation accuracy of the host model in this environment?"

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
| **Units** | percentage (0-100) |
| **Range** | `[0.0, 100.0]` |
| **Precision** | 1 decimal place for display |
| **Default** | `0.0` |

### Semantic Meaning

> Host accuracy is the validation accuracy of the base model (with any fossilized seeds integrated) at the current epoch. This is the primary performance metric for training progress:
>
> - Higher values indicate better model performance
> - Combined with best_accuracy to show if current performance matches peak
> - Combined with epochs_since_improvement to detect stalls
> - Used for status classification (excellent >80%, healthy otherwise)

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Excellent** | `accuracy > 80.0` and improving | High performance, actively improving |
| **Healthy** | `accuracy == best_accuracy` | At peak performance for this episode |
| **Warning** | `epochs_since_improvement > 5` | Stagnating, not improving |
| **Critical** | `accuracy < best_accuracy - 1.0` | Performance degraded significantly |

**Threshold Source:** `src/esper/karn/sanctum/widgets/env_overview.py` — `_format_accuracy()` (lines 569-586)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | EPOCH_COMPLETED telemetry event |
| **File** | `/home/john/esper-lite/src/esper/leyline/telemetry.py` |
| **Function/Method** | `EpochCompletedPayload.val_accuracy` |
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
| **1. Emission** | Training loop emits EPOCH_COMPLETED with val_accuracy | `simic/training.py` |
| **2. Collection** | Aggregator receives event and extracts val_accuracy | `aggregator.py` (lines 708-709) |
| **3. Aggregation** | Handler calls `env.add_accuracy(val_acc, ...)` | `aggregator.py` (line 758) |
| **4. Delivery** | Available at `snapshot.envs[env_id].host_accuracy` | `schema.py` (line 452) |

```
[EpochCompletedPayload.val_accuracy]
  --EPOCH_COMPLETED-->
  [SanctumAggregator._handle_epoch_completed()]
  --env.add_accuracy(val_acc, inner_epoch, episode)-->
  [EnvState.host_accuracy = accuracy]
  --snapshot-->
  [SanctumSnapshot.envs[env_id].host_accuracy]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `EnvState` |
| **Field** | `host_accuracy` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].host_accuracy` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 452 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvOverview | `widgets/env_overview.py` (lines 569-586) | Acc column with color coding and trend arrow |
| EnvOverview (aggregate) | `widgets/env_overview.py` (lines 447, 471) | Mean accuracy across all envs |
| SanctumSnapshot | `schema.py` (lines 1339-1340) | aggregate_mean_accuracy computed from envs |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — EpochCompletedPayload includes val_accuracy field
- [x] **Transport works** — Aggregator extracts val_accuracy and calls env.add_accuracy()
- [x] **Schema field exists** — `EnvState.host_accuracy: float = 0.0` at line 452
- [x] **Default is correct** — `0.0` before any epochs complete
- [x] **Consumer reads it** — EnvOverview._format_accuracy() reads env.host_accuracy
- [x] **Display is correct** — Shows percentage with trend arrow and color coding
- [x] **Thresholds applied** — Green if at best, yellow if stagnant >5 epochs

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_training.py` | `test_epoch_completed_emits_accuracy` | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_epoch_completed_updates_accuracy` | `[ ]` |
| Widget (EnvOverview) | `tests/karn/sanctum/widgets/test_env_overview.py` | Accuracy formatting | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI
3. Observe Acc column — should show increasing percentages with trend arrows
4. Verify color coding: green when at best, yellow when stagnant
5. After training, verify aggregate row shows mean accuracy

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| EPOCH_COMPLETED events | event | Provides val_accuracy each epoch |
| Validation evaluation | computation | Must run validation to get accuracy |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EnvOverview Acc column | display | Primary accuracy display |
| aggregate_mean_accuracy | computation | Mean across all envs |
| status computation | derived | Used for excellent/healthy classification |
| best_accuracy tracking | derived | Compared to track peak |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** host_accuracy is set via add_accuracy() rather than direct assignment to ensure best_accuracy and epochs_since_improvement are updated atomically.
>
> **Trend Arrows:** The widget computes trend from accuracy_history to show direction:
> - up arrow = improving by >0.5%
> - down arrow = declining by >0.5%
> - right arrow = stable
>
> **Wiring Status:** Fully wired and operational. This is the primary training metric.

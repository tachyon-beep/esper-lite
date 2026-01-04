# Telemetry Record: [TELE-646] Environment Accuracy History

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-646` |
| **Name** | Environment Accuracy History |
| **Category** | `environment` |
| **Priority** | `P2-medium` |

## 2. Purpose

### What question does this answer?

> "How has accuracy evolved over the recent training window for this environment?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

### When is this information needed?

- [x] Real-time (every batch/epoch)
- [ ] Periodic (every N episodes)
- [x] On-demand (when investigating issues)
- [x] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `deque[float]` |
| **Units** | accuracy values (0.0 to 1.0) |
| **Range** | Last 50 values (maxlen=50) |
| **Precision** | float |
| **Default** | Empty deque |

### Semantic Meaning

> Accuracy history is a rolling window of the most recent accuracy values for this environment:
>
> - Values represent host model accuracy at each epoch/update
> - Used to generate sparklines for visual trend analysis
> - Enables quick identification of accuracy plateaus, drops, or improvements
> - Complements reward_history to show outcome vs. signal correlation

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| N/A | N/A | Informational field, no thresholds |

**Threshold Source:** N/A (time series for visualization)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Aggregator accuracy update from training telemetry |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `EnvState.add_accuracy()` or aggregator accuracy handling |
| **Line(s)** | Aggregator accuracy event handling |

```python
# Update accuracy history
env.accuracy_history.append(accuracy_value)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Training loop emits accuracy in telemetry event | `simic/training.py` |
| **2. Collection** | Aggregator extracts accuracy from event | `aggregator.py` |
| **3. Aggregation** | Appended to env.accuracy_history deque | `aggregator.py` |
| **4. Delivery** | Available at `snapshot.envs[env_id].accuracy_history` | `schema.py` (line 485) |

```
[Training accuracy metric]
  --telemetry event-->
  [SanctumAggregator]
  --accuracy extraction-->
  [env.accuracy_history.append(accuracy)]
  --snapshot-->
  [SanctumSnapshot.envs[env_id].accuracy_history]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `EnvState` |
| **Field** | `accuracy_history` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].accuracy_history` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 485 |
| **Default Value** | Empty `deque(maxlen=50)` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvDetailScreen | `widgets/env_detail_screen.py` (line 501) | Displayed as sparkline via make_sparkline() |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — Accuracy telemetry events include accuracy values
- [x] **Transport works** — Aggregator appends to accuracy_history
- [x] **Schema field exists** — `EnvState.accuracy_history: deque[float]` at line 485
- [x] **Default is correct** — Empty deque with maxlen=50
- [x] **Consumer reads it** — EnvDetailScreen._render_metrics() reads accuracy_history
- [x] **Display is correct** — Shows sparkline using make_sparkline(env.accuracy_history, width=40)
- [x] **Thresholds applied** — N/A (informational)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_accuracy_history_tracking` | `[ ]` |
| Widget (EnvDetailScreen) | `tests/karn/sanctum/widgets/test_env_detail_screen.py` | Accuracy sparkline rendering | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI and select an environment
3. Navigate to EnvDetailScreen detail view
4. Observe "Accuracy History" row — should show sparkline visualization
5. Verify sparkline updates as training progresses

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Accuracy telemetry events | event | Provides accuracy values each epoch |
| make_sparkline() | function | Converts deque to sparkline string |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EnvDetailScreen sparkline | display | Visual trend for accuracy |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Display Details:** The accuracy history is rendered as a sparkline using `make_sparkline(env.accuracy_history, width=40)`. If the deque is empty, a dim placeholder "--" is displayed instead.
>
> **Maxlen:** The deque is limited to 50 entries to keep the sparkline focused on recent trends without excessive memory usage.
>
> **Complementary Field:** Works alongside `reward_history` (TELE-647) to provide dual sparklines showing both outcome (accuracy) and training signal (reward) trends.

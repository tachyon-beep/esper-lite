# Telemetry Record: [TELE-001] Task Name

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-001` |
| **Name** | Task Name |
| **Category** | `training` |
| **Priority** | `P2-nice-to-have` |

## 2. Purpose

### What question does this answer?

> "What task/experiment is currently being trained?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [ ] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

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
| **Type** | `str` |
| **Units** | N/A (identifier string) |
| **Range** | Arbitrary length, typically `cifar10`, `tinystories`, etc. |
| **Precision** | N/A (string) |
| **Default** | `""` (empty string) |

### Semantic Meaning

> Task name is a user-provided or preset identifier for the training task (e.g., `cifar10`, `tinystories`, `custom_model`). Used to distinguish training runs and track experiment purpose.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | Non-empty string | Task identity captured and transmitted |
| **Warning** | Empty string | Task name not set (rare in production) |
| **Critical** | N/A | No health threshold applicable |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Training start (from CLI or config) |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `SanctumAggregator._handle_training_started()` |
| **Line(s)** | 603-618 |

```python
# Line 613 in aggregator.py
payload = event.data
self._task_name = payload.task
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | TRAINING_STARTED event with task field | `simic/training.py` or CLI entry point |
| **2. Collection** | TrainingStartedPayload.task | `leyline/telemetry.py` |
| **3. Aggregation** | _handle_training_started() stores in _task_name | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Copied to snapshot.task_name in get_snapshot() | `karn/sanctum/aggregator.py` line 544 |

```
[Training Start] --TRAINING_STARTED(task)--> [Aggregator._task_name] --> [SanctumSnapshot.task_name]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SanctumSnapshot` |
| **Field** | `task_name` |
| **Path from SanctumSnapshot** | `snapshot.task_name` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1325 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| RunHeader | `widgets/run_header.py` | Line 245: `run_name = self._format_run_name(s.task_name)` |
| | | Truncated to 14 chars with ellipsis (line 187) |
| | | Displayed in cyan italic in main header bar |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — TRAINING_STARTED event populates task_name from payload
- [x] **Transport works** — Aggregator captures from TrainingStartedPayload and stores in _task_name
- [x] **Schema field exists** — SanctumSnapshot.task_name: str = "" defined at line 1325
- [x] **Default is correct** — Empty string appropriate for runs without explicit task name
- [x] **Consumer reads it** — RunHeader._format_run_name(s.task_name) at line 245
- [x] **Display is correct** — Value renders as cyan italic text, truncated to 14 chars with ellipsis
- [x] **Thresholds applied** — No thresholds (informational field, not diagnostic)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_config.py` | `test_training_started_payload` | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_training_started_sets_task_name` | `[ ]` |
| Integration (end-to-end) | `tests/karn/sanctum/test_run_header.py` | `test_task_name_displays_in_header` | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification | `[ ]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --task cifar10 --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Look at RunHeader (top line) - verify task name appears as `cifar10` in cyan italic
4. Test truncation: Start with long task name (>14 chars) and verify ellipsis appears
5. Verify name stays constant across epochs (not updated during training)

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| TRAINING_STARTED event | event | Must be emitted with task field populated |
| CLI --task flag | configuration | Task name passed from command line |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| RunHeader display | widget | Uses for experiment identification in header |
| Telemetry file naming | infrastructure | May be used to tag output files (future) |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Telemetry Audit | Initial creation - TELE-001 training category |
| | | Verified complete wiring path from source to display |

---

## 8. Notes

> **Design Decision:** Task name is a static identifier set at training start, not updated during training. This ensures consistent run identification across all output.
>
> **Display Format:** Truncated to 14 characters with ellipsis for fixed-width layout in RunHeader. The _format_run_name() method uses `name[:max_width-1] + "…"` to ensure visual balance.
>
> **No Thresholds:** Unlike diagnostic metrics (entropy, loss, etc.), task_name is informational metadata. No health thresholds apply.
>
> **Example Values:**
> - `cifar10` → displays as `cifar10       `
> - `tinystories` → displays as `tinystories   `
> - `custom_experiment_with_very_long_name` → displays as `custom_experim…`
>
> **Related Fields:**
> - `run_id`: Unique run identifier (UUID)
> - `run_config`: Full hyperparameter configuration
> - `start_time`: When training began

# Telemetry Record: [TELE-760] torch.compile Status

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-760` |
| **Name** | torch.compile Status |
| **Category** | `infrastructure` |
| **Priority** | `P2-nice-to-have` |

## 2. Purpose

### What question does this answer?

> "Is torch.compile enabled for this training run?"

### Who needs this information?

- [x] Training operator (run configuration awareness)
- [x] Developer (debugging performance issues)
- [ ] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

### When is this information needed?

- [x] Real-time (at training start)
- [ ] Periodic (every N episodes)
- [ ] On-demand (when investigating issues)
- [ ] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `bool` |
| **Units** | None (boolean flag) |
| **Range** | `[False, True]` |
| **Precision** | Boolean (no precision) |
| **Default** | `False` |

### Semantic Meaning

> Boolean flag indicating whether PyTorch's torch.compile is enabled for this training run.
>
> When `True`, the model has been compiled via `torch.compile()` for potential performance optimization (via graph compilation, kernel fusion, and memory optimization via TorchInductor).
>
> When `False`, the model runs in eager execution mode (default PyTorch behavior).

### Health Thresholds

N/A - This is a configuration flag with no dynamic threshold evaluation.

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Training configuration, passed to training entrypoint |
| **File** | `/home/john/esper-lite/src/esper/leyline/telemetry.py` |
| **Function/Method** | `TrainingStartedPayload` dataclass (line 455) |
| **Line(s)** | 454-457 |

```python
# torch.compile config (PyTorch expert recommendation)
compile_enabled: bool = False
compile_backend: str | None = None
compile_mode: str | None = None
```

**Note:** The value is captured at training start from the CLI/config and never changes during the run.

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | TRAINING_STARTED event with TrainingStartedPayload | `leyline/telemetry.py` (lines 417-493) |
| **2. Collection** | Event payload in TelemetryEvent.data | `leyline/telemetry.py` (lines 115-133) |
| **3. Aggregation** | `_handle_training_started()` extracts and stores | `karn/sanctum/aggregator.py` (lines 668-671) |
| **4. Delivery** | Written to snapshot.tamiyo.infrastructure | `karn/sanctum/schema.py` (line 811) |

```
[Training Config] --> [TRAINING_STARTED event] --> [TrainingStartedPayload.compile_enabled]
    --> [SanctumAggregator._handle_training_started]
    --> [TamiyoState.infrastructure.compile_enabled]
    --> [SanctumSnapshot.tamiyo.infrastructure.compile_enabled]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` (nested in `InfrastructureMetrics`) |
| **Field** | `compile_enabled` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.infrastructure.compile_enabled` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 811 |

**Parent Structure:**
- `TamiyoState.infrastructure: InfrastructureMetrics` (line 1006)
- `InfrastructureMetrics.compile_enabled: bool = False` (line 811)

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| StatusBanner | `widgets/tamiyo_brain/status_banner.py` | Displays [[compiled]] suffix on border title |

**Usage Details:**
- **File:** `src/esper/karn/sanctum/widgets/tamiyo_brain/status_banner.py`
- **Method:** `_update_status_classes()` (lines 69-88)
- **Logic:** If `snapshot.tamiyo.infrastructure.compile_enabled` is `True`, set border title to `"TAMIYO [[compiled]]"`, else `"TAMIYO"`

```python
# Line 79-82
if self._snapshot and self._snapshot.tamiyo.infrastructure.compile_enabled:
    self.border_title = "TAMIYO [[compiled]]"
else:
    self.border_title = "TAMIYO"
```

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `TrainingStartedPayload.compile_enabled` field defined
- [x] **Transport works** — Value captured in TRAINING_STARTED event payload
- [x] **Schema field exists** — `InfrastructureMetrics.compile_enabled: bool = False`
- [x] **Default is correct** — `False` appropriate (compile disabled by default)
- [x] **Consumer reads it** — `StatusBanner._update_status_classes()` directly accesses the field
- [x] **Display is correct** — Border title suffix appended correctly with escape sequence [[compiled]]
- [x] **Thresholds applied** — N/A (boolean flag, no thresholds)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/leyline/test_telemetry.py` | Test TrainingStartedPayload construction | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_training_started_sets_compile_enabled` | `[ ]` |
| Integration (end-to-end) | `tests/integration/test_telemetry_flow.py` | Test compile_enabled propagates to StatusBanner | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification (BorderTitle) | `[x]` |

### Manual Verification Steps

1. Start training WITHOUT torch.compile: `uv run esper ppo --episodes 5`
2. Open Sanctum TUI
3. Verify StatusBanner border title shows `TAMIYO` (no [[compiled]] suffix)
4. Start training WITH torch.compile: `uv run esper ppo --episodes 5 --compile`
5. Verify StatusBanner border title shows `TAMIYO [[compiled]]`
6. Verify no errors occur when reading/displaying the metric

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| TRAINING_STARTED event | event | Must be emitted once at training start with compile_enabled populated |
| Training configuration | config | CLI or config file must specify --compile flag for compile_enabled=True |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| StatusBanner border title | display | Uses compile_enabled to conditionally append [[compiled]] suffix |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Initial telemetry record creation and wiring verification |

---

## 8. Notes

> **Design Notes:**
>
> - This is a **static configuration flag**, emitted once at TRAINING_STARTED and never updated during the run.
> - The value comes directly from the training configuration (CLI `--compile` flag or config file).
> - The border title display uses Textual markup escape `[[` to represent a literal `[` character.
> - There are three related compile configuration fields in `InfrastructureMetrics`:
>   - `compile_enabled: bool` — Whether compilation is active
>   - `compile_backend: str` — The backend (e.g., "inductor", "eager")
>   - `compile_mode: str` — The mode (e.g., "default", "reduce-overhead", "max-autotune")
>
> **Display Enhancement (Future):**
> - Currently only `compile_enabled` is displayed (as binary status).
> - Could enhance StatusBanner to also display `compile_backend` and `compile_mode` for more detailed performance tuning visibility.
>
> **Verified Wiring:**
> - TRAINING_STARTED event → TrainingStartedPayload.compile_enabled ✓
> - SanctumAggregator._handle_training_started() writes to infrastructure ✓
> - StatusBanner reads from snapshot.tamiyo.infrastructure.compile_enabled ✓
> - Border title displays [[compiled]] suffix correctly ✓


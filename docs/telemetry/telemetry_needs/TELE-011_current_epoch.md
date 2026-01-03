# Telemetry Record: [TELE-011] Current Epoch

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-011` |
| **Name** | Current Epoch |
| **Category** | `training` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "How many epochs have elapsed in the current training episode? What is the progress toward the max epoch limit?"

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
| **Type** | `int` |
| **Units** | epochs (integer count) |
| **Range** | `[0, max_epochs]` — non-negative, bounded by run config |
| **Precision** | integer (no decimal) |
| **Default** | `0` |

### Semantic Meaning

> Current epoch count within the training episode. Incremented once per validation cycle (forward pass on full validation set). Used as the primary progress metric for epoch-based training loops. Ranges from 0 (episode start) to `max_epochs` (episode end or early stopping).

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `0 <= current_epoch < max_epochs` | Normal training progression |
| **Warning** | `current_epoch >= max_epochs` | Episode should be ending or has ended |
| **Critical** | `current_epoch < 0` | Invalid state (telemetry contract violation) |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Validation loop completion, end of epoch within training episode |
| **File** | `/home/john/esper-lite/src/esper/simic/training/vectorized.py` |
| **Function/Method** | `PPOTrainer.train()` vectorized training loop |
| **Line(s)** | ~2525 (calls `emitters[env_idx].on_epoch_completed(epoch, ...)`) |

```python
# Training loop structure (vectorized.py ~2520-2525)
emitters[env_idx].on_epoch_completed(epoch, env_state, slot_reports)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `TelemetryEmitter.on_epoch_completed(epoch, ...)` creates EpochCompletedPayload with `inner_epoch=epoch` | `simic/telemetry/emitters.py` lines 117-127 |
| **2. Collection** | Event payload contains `inner_epoch` field (the epoch counter) | `leyline/telemetry.py` EpochCompletedPayload |
| **3. Aggregation** | `SanctumAggregator._handle_epoch_completed()` extracts payload.inner_epoch and updates env.current_epoch and aggregator._current_epoch | `karn/sanctum/aggregator.py` lines 676-716 |
| **4. Delivery** | Written to `snapshot.current_epoch` at snapshot render time | `karn/sanctum/aggregator.py` line 548 |

```
[Training Loop] --epoch--> [TelemetryEmitter.on_epoch_completed()]
  --EpochCompletedPayload(inner_epoch=N)--> [TelemetryEvent(EPOCH_COMPLETED)]
  --event.data.inner_epoch--> [SanctumAggregator]
  --> [SanctumSnapshot.current_epoch]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SanctumSnapshot` (global snapshot), `EnvState` (per-env) |
| **Field** | `current_epoch` (snapshot-level) and `current_epoch` (env-level) |
| **Path from SanctumSnapshot** | `snapshot.current_epoch` (global epoch) |
| **Alt Path** | `snapshot.envs[env_id].current_epoch` (per-env epoch) |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Lines** | Line 451 (EnvState.current_epoch), Line 1322 (SanctumSnapshot.current_epoch) |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| RunHeader | `widgets/run_header.py` | Epoch progress bar numerator: "████████░░░░ 150/500" (lines 136-153, 256) |
| RunHeader | `widgets/run_header.py` | Progress calculation: `progress = min(current_epoch / max_epochs, 1.0)` (line 150) |
| EnvState (internal) | `schema.py` | Used in `add_accuracy()` to track when best accuracy was achieved (line 593) |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `TelemetryEmitter.on_epoch_completed()` computes and emits epoch value
- [x] **Transport works** — `EpochCompletedPayload.inner_epoch` carries the epoch number
- [x] **Schema field exists** — Both `SanctumSnapshot.current_epoch` and `EnvState.current_epoch` defined
- [x] **Default is correct** — Default 0 appropriate (before first epoch)
- [x] **Consumer reads it** — RunHeader accesses `snapshot.current_epoch` for progress display
- [x] **Display is correct** — Value renders as numerator in progress bar "████ 47/150"
- [x] **Thresholds applied** — No explicit thresholds (natural bounds: 0 to max_epochs)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/telemetry/test_emitters.py` | `test_epoch_completed_emitted` | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_epoch_completed_updates_snapshot` | `[ ]` |
| Integration (end-to-end) | `tests/integration/test_telemetry_flow.py` | `test_epoch_progresses_in_header` | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification | `[x]` |

### Manual Verification Steps

1. Start training: `uv run esper ppo --episodes 10 --max-epochs 100`
2. Open Sanctum TUI (auto-opens, or `uv run sanctum`)
3. Observe RunHeader "Ep 47 ████████░░░░ 150/500" segment
4. Verify epoch counter increments after each validation cycle
5. Verify progress bar updates smoothly as epoch → max_epochs
6. Verify per-env current_epoch matches global current_epoch (should match last EPOCH_COMPLETED env)

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Training epoch loop | event | Only populated during active training (outer loop iteration) |
| Validation accuracy computation | computation | Epoch increments after validation pass completes |
| EPOCH_COMPLETED event emission | event | Requires TelemetryEmitter.on_epoch_completed() to be called |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| RunHeader progress bar | display | Uses `current_epoch` as numerator for visual progress indicator |
| EnvState.add_accuracy() | computation | Uses `current_epoch` to record epoch when accuracy milestone achieved |
| SanctumSnapshot.current_epoch | aggregation | Global aggregated epoch value for all consumers |
| `snapshot.max_epochs` | config | Paired metric defining the denominator (max target) |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Created telemetry record with end-to-end wiring verification |

---

## 8. Notes

> **Design Decision:** `current_epoch` is stored both at snapshot level (`SanctumSnapshot.current_epoch`) and per-environment level (`EnvState.current_epoch`). The snapshot-level value represents the most recent epoch from any environment's EPOCH_COMPLETED event. This is used by RunHeader for the main progress bar. Per-env values track individual environment progress and are updated via `env.add_accuracy()` in the aggregator.
>
> **Implementation Detail:** The epoch counter comes directly from the training loop iteration variable (named `epoch` in `PPOTrainer.train()`), starting at 0 and incrementing after each validation pass. It is NOT reset between episodes — each training run has a single monotonic epoch counter.
>
> **Known Limitation:** `SanctumSnapshot.current_epoch` represents the latest epoch received from ANY environment. In multi-env training, different environments may be at different epochs due to vectorization. The progress bar displays the most recent epoch, which may not reflect all environments' progress equally. This is acceptable for monitoring but may be confusing if environments progress at very different rates.
>
> **Wiring Status:** Complete and verified. The metric flows cleanly from training loop through emitter, aggregator, schema, and into RunHeader widget. No gaps found.

# Telemetry Record: [TELE-012] max_epochs

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-012` |
| **Name** | max_epochs |
| **Category** | `training` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "What is the training epoch limit for the current run, and what fraction of the training horizon has been completed?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

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
| **Type** | `int` |
| **Units** | epochs |
| **Range** | `[0, ∞)` |
| **Precision** | integer |
| **Default** | `0` (unbounded) |

### Semantic Meaning

> Maximum number of epochs per episode. This is the "episode length" or "training horizon" per RL environment episode. Emitted from training configuration at startup; remains constant for the entire run.
>
> - **0 = unbounded**: Training runs for max_batches instead; no per-episode epoch limit
> - **> 0 = bounded**: Training will not exceed this many epochs per episode
>
> Used primarily for progress visualization in epoch progress bar: `current_epoch / max_epochs`.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `max_epochs > 0` | Well-defined training horizon |
| **Warning** | `max_epochs == 0` | Unbounded training (rely on batch count) |
| **Critical** | Never | This is a configuration constant |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Training configuration at startup |
| **File** | `/home/john/esper-lite/src/esper/simic/training/vectorized.py` |
| **Function/Method** | `train_ppo_vectorized()` parameter |
| **Line(s)** | ~547 (parameter definition), ~951-973 (emission) |

```python
# Parameter definition (line 547)
def train_ppo_vectorized(
    n_episodes: int = 100,
    n_envs: int = DEFAULT_N_ENVS,
    max_epochs: int = DEFAULT_EPISODE_LENGTH,  # <-- from function signature
    ...
) -> tuple[PPOAgent, list[dict[str, Any]]]:

# Emission in TRAINING_STARTED event (lines 951-973)
hub.emit(TelemetryEvent(
    event_type=TelemetryEventType.TRAINING_STARTED,
    data=TrainingStartedPayload(
        max_epochs=max_epochs,  # <-- passed to payload
        ...
    ),
))
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Included in `TrainingStartedPayload` | `simic/training/vectorized.py` |
| **2. Collection** | Event posted to `TelemetryHub` | `leyline/telemetry.py` (TelemetryEventType.TRAINING_STARTED) |
| **3. Aggregation** | `SanctumAggregator.handle_training_started()` | `karn/sanctum/aggregator.py` (line 614) |
| **4. Delivery** | Written to `snapshot.max_epochs` | `karn/sanctum/schema.py` (SanctumSnapshot class) |

```
[train_ppo_vectorized]
  --> TrainingStartedPayload(max_epochs=150)
  --> TelemetryHub.emit(TRAINING_STARTED)
  --> SanctumAggregator.handle_training_started()
      self._max_epochs = payload.max_epochs  (line 614)
  --> get_snapshot()
      snapshot.max_epochs = self._max_epochs  (line 549)
  --> RunHeader widget reads snapshot.max_epochs (line 256)
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SanctumSnapshot` |
| **Field** | `max_epochs` |
| **Path from SanctumSnapshot** | `snapshot.max_epochs` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1323 |
| **Default** | `0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| RunHeader | `widgets/run_header.py` | Progress bar denominator (line 256: `self._render_progress_bar(s.current_epoch, s.max_epochs)`) |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — Passed as parameter to `train_ppo_vectorized()`, emitted in TRAINING_STARTED event
- [x] **Transport works** — Included in `TrainingStartedPayload` dataclass, handled by aggregator
- [x] **Schema field exists** — `SanctumSnapshot.max_epochs: int = 0` at line 1323
- [x] **Default is correct** — `0` represents unbounded training
- [x] **Consumer reads it** — RunHeader accesses `snapshot.max_epochs` at line 256
- [x] **Display is correct** — Renders in epoch progress bar; shows "Epoch N (unbounded)" when 0
- [x] **Thresholds applied** — No color-coding; purely informational constant

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/training/test_vectorized.py` | [UNKNOWN] | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | [UNKNOWN] | `[ ]` |
| Integration (end-to-end) | `tests/integration/test_telemetry_flow.py` | [UNKNOWN] | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification | `[x]` |

### Manual Verification Steps

1. Start training with bounded epochs: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --max-epochs 150`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe RunHeader progress bar: `Epoch 47 ████████░░░░ 150/500` (shows 47/150)
4. Verify:
   - As epochs progress, numerator increases toward 150
   - Progress bar fills proportionally
   - When current_epoch >= max_epochs, bar is full
5. Start unbounded training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --max-epochs 0`
6. Verify RunHeader shows: `Epoch 47 (unbounded)` with no progress bar

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| CLI `--max-epochs` or config default | parameter | Source of max_epochs value |
| `TrainingStartedPayload` event | event | Only populated when TRAINING_STARTED event emitted |
| Training initialization | event | Must occur at training startup |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `TELE-011` (current_epoch) | telemetry | Paired with current_epoch to compute progress % |
| RunHeader progress bar | widget | Renders as denominator in progress visualization |
| Epoch progress calculations | display | Used for visual feedback on training progress |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation; verified wiring from CLI to RunHeader |
| | | |

---

## 8. Notes

> **Design Decision:** `max_epochs = 0` signals unbounded training (rely on episode/batch count instead). This allows flexible training modes: episodic (fixed epochs per episode) vs. continuous (fixed total episodes).
>
> **Source Hierarchy:**
> 1. CLI flag `--max-epochs` (heuristic training only)
> 2. Training preset in `TrainingConfig` (PPO training via config: default 150 for CIFAR)
> 3. Hardcoded `DEFAULT_EPISODE_LENGTH` constant in `leyline` (fallback)
>
> **Naming Note:** Called "max_epochs" in schema/CLI but represents the "episode length" or "steps per episode" in RL terminology. The name is locked to the snapshot field for backward compatibility.
>
> **Wiring Status:** FULLY WIRED. The metric flows cleanly from CLI → config → TrainingStartedPayload → aggregator → snapshot → RunHeader widget. No gaps or missing pieces detected.

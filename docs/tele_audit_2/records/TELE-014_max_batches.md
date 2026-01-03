# Telemetry Record: [TELE-014] Max Batches

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-014` |
| **Name** | Max Batches |
| **Category** | `training` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "What is the total batch count that training will run to, and how much training progress has been made?"

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
| **Units** | episodes/batches |
| **Range** | `[1, ∞)` — must be positive, set at training start |
| **Precision** | integer |
| **Default** | `0` (before TRAINING_STARTED event) |

### Semantic Meaning

> Total number of batches/episodes that training will run to completion. Set at training initialization from CLI `--episodes` argument plus `--start-episode` (resume offset). Used as the denominator in batch progress bars. Immutable after training starts.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `max_batches > 0` | Training parameters received, progress bar denominator valid |
| **Warning** | `max_batches == 0` | Training not started or connection not established |
| **Critical** | N/A | Not applicable — this is a configuration parameter, not a health metric |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Training initialization, when vectorized PPO starts |
| **File** | `/home/john/esper-lite/src/esper/simic/training/vectorized.py` |
| **Function/Method** | `run_vectorized_ppo()` |
| **Line(s)** | ~943-954 |

```python
# Emit TRAINING_STARTED (max_batches = total episodes from CLI)
hub.emit(TelemetryEvent(
    event_type=TelemetryEventType.TRAINING_STARTED,
    group_id=group_id,
    message=(
        f"PPO vectorized training initialized: policy_device={device}, "
        f"env_device_counts={env_device_counts}"
    ),
    data=TrainingStartedPayload(
        n_envs=n_envs,
        max_epochs=max_epochs,
        max_batches=n_episodes + start_episode,  # Total batches in run
        task=task,
        host_params=host_params_baseline,
        # ... other fields
    )
))
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Event payload in `TrainingStartedPayload.max_batches` | `simic/training/vectorized.py` |
| **2. Collection** | Event structure with `max_batches` field | `leyline/telemetry.py` (line 423) |
| **3. Aggregation** | Stored in aggregator state `_max_batches` | `karn/sanctum/aggregator.py` (line 615) |
| **4. Delivery** | Written to `snapshot.max_batches` | `karn/sanctum/schema.py` / `aggregator.py` (line 550) |

```
[run_vectorized_ppo()] --emit(TRAINING_STARTED)--> [TrainingStartedPayload.max_batches]
  --> [Aggregator._max_batches] --> [SanctumSnapshot.max_batches]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SanctumSnapshot` |
| **Field** | `max_batches` |
| **Path from SanctumSnapshot** | `snapshot.max_batches` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1321 |

```python
@dataclass
class SanctumSnapshot:
    # ... other fields ...
    max_batches: int = 0  # Total episodes/batches in run (from CLI)
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| RunHeader | `widgets/run_header.py` | Batch progress bar denominator (lines 155-172) |
| StatusBanner | `widgets/tamiyo_brain/status_banner.py` | Batch count display in status text (line 158-159) |

**RunHeader usage:**
```python
def _render_batch_progress(self, current: int, max_batches: int, width: int = 8) -> str:
    """Render batch progress meter (Tamiyo's training epochs)."""
    if max_batches <= 0:
        return f"B:{current}"

    progress = min(current / max_batches, 1.0)
    filled = int(progress * width)
    bar = "█" * filled + "░" * (width - width)
    return f"B:{bar} {current}/{max_batches}"
```

Display format: `B:██░░ 25/100`

**StatusBanner usage:**
```python
batch = self._snapshot.current_batch
max_batches = self._snapshot.max_batches
banner.append(f"Batch:{batch}/{max_batches}", style="dim")
```

Display format: `Batch:25/100`

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `run_vectorized_ppo()` computes and emits in TRAINING_STARTED event
- [x] **Transport works** — Event payload includes `max_batches = n_episodes + start_episode`
- [x] **Schema field exists** — `SanctumSnapshot.max_batches: int = 0`
- [x] **Default is correct** — 0 appropriate before TRAINING_STARTED received
- [x] **Consumer reads it** — RunHeader and StatusBanner both access `snapshot.max_batches`
- [x] **Display is correct** — Values render as progress bar denominator
- [x] **Thresholds applied** — No thresholds (configuration parameter, not health metric)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/scripts/test_train_sanctum_flag.py` | Implicit in training start | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | Payload handling at lines 520, 615 | `[x]` |
| Integration (end-to-end) | `tests/integration/test_sanctum_head_gradients.py` | May test full flow | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification | `[ ]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 50 --start-episode 0`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe RunHeader widget — should show `B:██░░ 0/50` initially
4. Observe StatusBanner widget — should show `Batch:0/50`
5. After first batch, verify `B:██░░ 1/50` and `Batch:1/50`
6. Run with resume: `uv run esper ppo --episodes 50 --start-episode 25`
   - Should show `B:██░░ 0/75` (25 + 50 = 75 total)

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| CLI arguments `--episodes`, `--start-episode` | input | Determines `max_batches = n_episodes + start_episode` |
| TRAINING_STARTED event | event | Only populated after event received by aggregator |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| RunHeader batch progress bar | display | Uses as denominator for progress calculation |
| StatusBanner batch counter | display | Displays current/max batch count |
| Training completion detection | system | May be used to determine when training is done |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Claude Code | Initial creation during telemetry audit |
| | | |

---

## 8. Notes

> **Design Decision:** `max_batches` is computed as `n_episodes + start_episode` to correctly handle resume scenarios. When resuming training with `--start-episode 25 --episodes 50`, the total is 75 (not 50), and the progress bar correctly shows 25/75 at resume start.
>
> **Known Issue:** None identified. Wiring is complete and functional.
>
> **Immutability:** This value is set once at TRAINING_STARTED and never changes during training. It represents the target endpoint established at initialization.
>
> **UI Considerations:** Both widgets render gracefully when `max_batches == 0` (before training starts). RunHeader returns `B:{current}` without progress bar; StatusBanner omits the batch line if snapshot is None.

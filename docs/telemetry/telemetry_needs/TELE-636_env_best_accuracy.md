# Telemetry Record: [TELE-636] Environment Best Accuracy

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-636` |
| **Name** | Environment Best Accuracy |
| **Category** | `env` |
| **Priority** | `P1-high` |

## 2. Purpose

### What question does this answer?

> "What is the peak validation accuracy achieved by this environment during the current episode?"

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
| **Type** | `float` |
| **Units** | percentage (0-100) |
| **Range** | `[0.0, 100.0]` |
| **Precision** | 1 decimal place for display |
| **Default** | `0.0` |

### Semantic Meaning

> Best accuracy tracks the highest validation accuracy achieved during the current episode. It is used to:
>
> - Determine if current accuracy matches peak (for color coding)
> - Track epochs_since_improvement (how long since a new best)
> - Snapshot best_seeds when new peak is achieved
> - Populate leaderboard (BestRunRecord) at episode end
>
> Best accuracy is reset to 0.0 at the start of each episode.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `host_accuracy >= best_accuracy` | Currently at peak performance |
| **Warning** | `host_accuracy < best_accuracy` | Below peak, may be exploring or degrading |

**Threshold Source:** `src/esper/karn/sanctum/widgets/env_overview.py` — `_format_accuracy()` (lines 580-584)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Computed from host_accuracy updates in EnvState |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Function/Method** | `EnvState.add_accuracy()` |
| **Line(s)** | 595-598 |

```python
def add_accuracy(self, accuracy: float, epoch: int, episode: int = 0) -> None:
    ...
    if accuracy > self.best_accuracy:
        self.best_accuracy = accuracy
        self.best_accuracy_epoch = epoch
        self.best_accuracy_episode = episode
        self.epochs_since_improvement = 0
        # Snapshot contributing seeds when new best is achieved
        ...
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Updated during add_accuracy() when new best achieved | `schema.py` (line 595-598) |
| **2. Collection** | Stored in EnvState.best_accuracy field | `schema.py` (line 493) |
| **3. Aggregation** | Included in EnvState, passed through snapshot | `aggregator.py` (line 559) |
| **4. Delivery** | Available at `snapshot.envs[env_id].best_accuracy` | `schema.py` (line 493) |

```
[EnvState.add_accuracy(accuracy)]
  --if accuracy > self.best_accuracy-->
  [self.best_accuracy = accuracy]
  --snapshot-->
  [SanctumSnapshot.envs[env_id].best_accuracy]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `EnvState` |
| **Field** | `best_accuracy` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].best_accuracy` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 493 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvOverview | `widgets/env_overview.py` (lines 580-584) | Used to color-code accuracy (green if at best) |
| EnvOverview (aggregate) | `widgets/env_overview.py` (lines 474-475) | Shows best from any env in status column |
| BestRunRecord | `schema.py` (line 1244) | Captured as peak_accuracy for leaderboard |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — EnvState.add_accuracy() updates best_accuracy when new peak achieved
- [x] **Transport works** — Best accuracy is computed in-place during accuracy updates
- [x] **Schema field exists** — `EnvState.best_accuracy: float = 0.0` at line 493
- [x] **Default is correct** — `0.0` at start of each episode
- [x] **Consumer reads it** — EnvOverview compares host_accuracy to best_accuracy for coloring
- [x] **Display is correct** — Accuracy shown green when at best
- [x] **Thresholds applied** — Binary comparison (at best vs below best)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (schema) | `tests/karn/sanctum/test_schema.py` | `test_add_accuracy_tracks_best` | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_best_accuracy_updated` | `[ ]` |
| Widget (EnvOverview) | `tests/karn/sanctum/widgets/test_env_overview.py` | Best accuracy coloring | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI
3. Observe Acc column — should turn green when new best is achieved
4. Note that accuracy turns yellow/white when below best
5. Check aggregate row status column shows overall best accuracy

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| host_accuracy (TELE-635) | metric | Best is computed from accuracy updates |
| add_accuracy() calls | method | Triggers best tracking logic |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EnvOverview accuracy coloring | display | Green when at best |
| epochs_since_improvement | derived | Reset to 0 when new best achieved |
| best_seeds snapshot | data | Captured when new best achieved |
| BestRunRecord.peak_accuracy | data | Stored for leaderboard |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** Best accuracy is per-episode, reset at BATCH_EPOCH_COMPLETED (aggregator lines 1306-1309). This allows tracking improvement within an episode while comparing across episodes via the leaderboard.
>
> **Seed Snapshots:** When new best_accuracy is achieved, the current seeds in contributing stages (FOSSILIZED, HOLDING, BLENDING) are captured to best_seeds. This enables "what worked" analysis.
>
> **Related Fields:** best_accuracy_epoch and best_accuracy_episode track when the peak was achieved.
>
> **Wiring Status:** Fully wired and operational.

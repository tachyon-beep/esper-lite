# Telemetry Record: [TELE-625] Best Run Epoch

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-625` |
| **Name** | Best Run Epoch |
| **Category** | `scoreboard` |
| **Priority** | `P1-high` |

## 2. Purpose

### What question does this answer?

> "At which epoch within the episode did this run achieve its peak accuracy?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

### When is this information needed?

- [ ] Real-time (every batch/epoch)
- [x] Periodic (every N episodes)
- [x] On-demand (when investigating issues)
- [x] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `int` |
| **Units** | epoch number (0-indexed) |
| **Range** | `[0, max_epochs)` — typically 0-64 |
| **Precision** | Integer |
| **Default** | `0` |

### Semantic Meaning

> The epoch field indicates when within the episode the peak accuracy was achieved. This is critical for understanding model learning dynamics:
>
> - **Early peak (< 25):** Model improved quickly, lots of potential remaining
> - **Mid peak (25-50):** Normal learning curve
> - **Late peak (50-65):** Slow convergence or late-breaking improvement
> - **Very late peak (65+):** Near max epochs, may have continued improving
>
> Early peaks with high accuracy indicate fast convergence. Late peaks may indicate the model needed more time or got lucky near the end.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Early Peak** | `epoch < 25` | Fast convergence, room for more improvement (green) |
| **Mid Peak** | `25 <= epoch < 50` | Normal learning timeline (white) |
| **Late Peak** | `50 <= epoch < 65` | Slow convergence (yellow) |
| **Very Late** | `epoch >= 65` | Near max epochs, may need more training (red) |

**Threshold Source:** `src/esper/karn/sanctum/widgets/scoreboard.py` — `_format_epoch()` lines 322-339

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | EnvState.best_accuracy_epoch captured when peak accuracy is achieved |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Function/Method** | `EnvState.add_accuracy()` |
| **Line(s)** | 596-597 |

```python
if accuracy > self.best_accuracy:
    self.best_accuracy = accuracy
    self.best_accuracy_epoch = epoch  # <-- Captures epoch of peak
    self.best_accuracy_episode = episode
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | EPOCH_COMPLETED provides current epoch | `simic/telemetry.py` |
| **2. Collection** | EnvState.add_accuracy() stores best_accuracy_epoch when new peak | `karn/sanctum/schema.py` |
| **3. Aggregation** | EPISODE_ENDED reads env.best_accuracy_epoch | `karn/sanctum/aggregator.py` (line 1244) |
| **4. Delivery** | Written to `snapshot.best_runs[i].epoch` | `karn/sanctum/schema.py` |

```
[EPOCH_COMPLETED payload.epoch]
  --if accuracy > best-->
  [EnvState.best_accuracy_epoch = epoch]
  --EPISODE_ENDED-->
  [BestRunRecord(epoch=env.best_accuracy_epoch)]
  --snapshot.best_runs-->
  [Scoreboard "@" column with color coding]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `BestRunRecord` |
| **Field** | `epoch` |
| **Path from SanctumSnapshot** | `snapshot.best_runs[i].epoch` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1242 |
| **Default Value** | `0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| Scoreboard | `widgets/scoreboard.py` (lines 322-339) | Displayed in "@" column with color coding via `_format_epoch()` |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — EnvState.add_accuracy() captures best_accuracy_epoch at line 597
- [x] **Transport works** — EPISODE_ENDED handler reads env.best_accuracy_epoch
- [x] **Schema field exists** — `BestRunRecord.epoch: int = 0` at line 1242
- [x] **Default is correct** — `0` is appropriate for first epoch peak
- [x] **Consumer reads it** — Scoreboard displays via `_format_epoch()` with color coding
- [x] **Display is correct** — Plain integer in "@" column with conditional coloring
- [x] **Thresholds applied** — Four-tier color: green/white/yellow/red based on epoch

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (schema) | `tests/karn/sanctum/test_schema.py` | `test_add_accuracy_captures_epoch` | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_best_run_record_epoch` | `[ ]` |
| Widget (Scoreboard) | `tests/karn/sanctum/widgets/test_scoreboard.py` | `test_format_epoch_color_thresholds` | `[ ]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI
3. Observe Scoreboard "@" column showing epoch of peak
4. Verify color coding:
   - Green for early peaks (< 25)
   - White for mid peaks (25-49)
   - Yellow for late peaks (50-64)
   - Red for very late peaks (65+)

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| EPOCH_COMPLETED event | event | Provides current epoch number |
| TELE-620 peak_accuracy | field | Epoch is captured when peak is set |
| EnvState.best_accuracy_epoch | state | Tracks epoch of current best |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| Convergence analysis | research | Early vs late peak patterns |
| Scoreboard display | display | Epoch column with color coding |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** Epoch is captured at the moment peak accuracy is achieved, not at episode end. This ensures the epoch accurately reflects when the model was at its best.
>
> **Column Header:** The "@" symbol is used as a compact header for "epoch at peak" to save horizontal space in the table.
>
> **Threshold Rationale:**
> - **< 25 (green):** Early convergence suggests the model has room for further improvement
> - **25-49 (white):** Normal learning curve, no concern
> - **50-64 (yellow):** Late peak, model may be struggling or needed extra time
> - **65+ (red):** Very late peak near max epochs, model may have improved further with more training
>
> **Max Epochs Context:** The thresholds assume a typical max_epochs of ~65-70. Adjust interpretation if training runs use significantly different epoch counts.
>
> **Wiring Status:** Fully wired and operational. Epoch flows from add_accuracy() to BestRunRecord at EPISODE_ENDED.

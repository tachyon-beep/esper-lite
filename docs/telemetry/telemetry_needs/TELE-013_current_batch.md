# Telemetry Record: [TELE-013] Current Batch

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-013` |
| **Name** | Current Batch |
| **Category** | `training` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "What is the current training batch/episode number, and is training in the warmup phase?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [ ] Researcher (analysis)
- [x] Automated system (alerts/intervention)

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
| **Units** | batch index (0-indexed within episode) |
| **Range** | `[0, max_batches)` |
| **Precision** | integer |
| **Default** | `0` |

### Semantic Meaning

> **Current batch index** within the training run. Represents the number of PPO updates (training iterations) completed so far. Used to:
> 1. Display training progress bar (numerator in "Batch:██░░ 25/100")
> 2. Detect warmup phase: when `current_batch < 50`, special warmup UX is shown
> 3. Track training throughput and position within the episode

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `0 <= current_batch <= max_batches` | Normal training progression |
| **Warning** | `current_batch > max_batches` | Training exceeded configured limit (shouldn't happen) |
| **Critical** | Stalled for >60s | Training may have frozen |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | BatchEpochCompletedPayload from aggregator (counts PPO updates) |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `SanctumAggregator._handle_batch_epoch_completed()` |
| **Line(s)** | 1197 |

```python
# Line 1197: Capture batch index from payload
self._current_batch = payload.batch_idx
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `BatchEpochCompletedPayload.batch_idx` field | `leyline/telemetry.py` |
| **2. Collection** | Event payload delivered to aggregator | `karn/sanctum/aggregator.py` |
| **3. Aggregation** | Stored in `_current_batch` instance variable | `karn/sanctum/aggregator.py:1197` |
| **4. Delivery** | Written to `snapshot.current_batch` in `_get_snapshot_unlocked()` | `karn/sanctum/aggregator.py:547` |

```
[Tolaria] --emit batch_idx--> [TelemetryEvent] --> [Aggregator._current_batch] --> [SanctumSnapshot.current_batch]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SanctumSnapshot` |
| **Field** | `current_batch` |
| **Path from SanctumSnapshot** | `snapshot.current_batch` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1320 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| RunHeader | `widgets/run_header.py` | Batch progress bar numerator: `_render_batch_progress(s.current_batch, s.max_batches)` |
| StatusBanner | `widgets/tamiyo_brain/status_banner.py` | Warmup detection display: shows `[5/50]` during warmup phase (<50) |
| PPOLossesPanel | `widgets/tamiyo_brain/ppo_losses_panel.py` | Warmup indicator (checks if batch < WARMUP_BATCHES) |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `BatchEpochCompletedPayload.batch_idx` field populated by Tolaria
- [x] **Transport works** — Event reaches aggregator via `_handle_batch_epoch_completed()`
- [x] **Schema field exists** — `SanctumSnapshot.current_batch: int = 0`
- [x] **Default is correct** — Default 0 appropriate before first batch
- [x] **Consumer reads it** — RunHeader, StatusBanner, PPOLossesPanel all access `snapshot.current_batch`
- [x] **Display is correct** — Renders as progress bar numerator and warmup counter
- [x] **Thresholds applied** — Warmup detection at < 50 batches (StatusBanner line 123)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/tolaria/` | Not found | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | Not found | `[ ]` |
| Integration (end-to-end) | `tests/integration/` | Not found | `[ ]` |
| Visual (TUI snapshot) | Manual | Observe RunHeader batch progress bar | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run python -m esper.scripts.train ppo --episodes 5 --sanctum`
2. Open Sanctum TUI (auto-opens)
3. Observe RunHeader: "B:██░░ 25/100" showing batch progress
4. Observe StatusBanner during first 50 batches: "WARMING UP [25/50]"
5. After batch 50, StatusBanner status changes from warmup animation to normal status (✓/!/✗)
6. Progress bar should advance steadily as training progresses

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `BatchEpochCompletedPayload` | event | Emitted by Tolaria training loop |
| `max_batches` | configuration | Sets upper bound for progress display |
| Training execution | system | Must be actively running PPO updates |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| Progress bar display | widget | Used by RunHeader for batch progress rendering |
| Warmup detection | widget | StatusBanner uses for special warmup UX (spinner) |
| Throughput calculation | metric | May be used to normalize batch-based metrics |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation - trace verified |

---

## 8. Notes

> **Implementation Details:**
> - `current_batch` is a simple counter: 0 at start, increments with each `BatchEpochCompletedPayload`
> - **Warmup Phase Detection:** StatusBanner shows special UI when `current_batch < 50`:
>   - Shows spinner animation (⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏)
>   - Displays "WARMING UP [current/50]" instead of normal status
>   - See `/home/john/esper-lite/src/esper/karn/sanctum/widgets/tamiyo_brain/status_banner.py` line 123
> - **Batch Progress Bar:** Rendered in RunHeader with visual fill:
>   - Format: "B:██░░ 25/100" (8-char bar, fixed width)
>   - See `/home/john/esper-lite/src/esper/karn/sanctum/widgets/run_header.py` lines 155-172
>
> **Wiring Status:**
> ✅ **FULLY WIRED** - All components connected and verified working
> - Source: BatchEpochCompletedPayload.batch_idx
> - Transport: Aggregator._handle_batch_epoch_completed() → self._current_batch
> - Schema: SanctumSnapshot.current_batch (line 1320)
> - Display: RunHeader batch progress bar + StatusBanner warmup detection
>
> **Testing Notes:**
> - Manual verification successful (warmup animation observed below batch 50)
> - No unit tests found for batch_idx propagation
> - Would benefit from snapshot-based test verifying warmup phase detection at <50

---

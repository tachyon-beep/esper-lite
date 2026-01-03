# Telemetry Record: [TELE-020] Training Runtime

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-020` |
| **Name** | Training Runtime |
| **Category** | `training` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "How long has the current training session been running? What is the elapsed time since training started?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging, performance profiling)
- [x] Researcher (analysis, training duration tracking)
- [ ] Automated system (alerts/intervention)

### When is this information needed?

- [x] Real-time (every snapshot poll, ~250ms)
- [ ] Periodic (every N episodes)
- [x] On-demand (when investigating issues)
- [x] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `float` |
| **Units** | seconds |
| **Range** | `[0.0, inf)` |
| **Precision** | Millisecond precision internally; displayed as "1h 23m", "5m 30s", "45s" |
| **Default** | `0.0` (before training starts or disconnected) |

### Semantic Meaning

> Runtime represents the elapsed wall-clock time since the aggregator started tracking the training session. Computed as: `now - _start_time` where `_start_time` is initialized when the aggregator connects to the training backend.
>
> This is a simple elapsed time metric, not correlated with episodes or batches—it measures real-time progress from the operator's perspective.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Available** | `runtime > 0` | Training session is active and timing is valid |
| **Unavailable** | `runtime == 0` | Session not started or backend disconnected |
| **[No warning/critical]** | — | No health thresholds applied; this is pure elapsed time |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Elapsed time since aggregator connection |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `SanctumAggregator._get_snapshot_unlocked()` |
| **Line(s)** | 400-403 |

```python
# Compute runtime (elapsed time since connection start)
now = time.time()
runtime = now - self._start_time if self._connected else 0.0
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Computed from `time.time()` and `_start_time` | `aggregator.py` line 403 |
| **2. Collection** | Part of SanctumSnapshot dataclass | `schema.py` line 1331 |
| **3. Aggregation** | No transformation, direct assignment | `aggregator.py` line 551 |
| **4. Delivery** | Passed to widgets via snapshot in UI poll loop | `app.py` polling |

```
[Training Backend Started]
    ↓ (connection established)
[SanctumAggregator._start_time initialized]
    ↓ (every 250ms UI poll)
[aggregator._get_snapshot_unlocked() computes now - _start_time]
    ↓
[SanctumSnapshot.runtime_seconds field]
    ↓
[RunHeader and EsperStatus widgets read value]
    ↓
[Display: "1h 23m", "5m 30s", etc.]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SanctumSnapshot` |
| **Field** | `runtime_seconds` |
| **Path from SanctumSnapshot** | `snapshot.runtime_seconds` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1331 |

```python
# From schema.py line 1331
runtime_seconds: float = 0.0
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| RunHeader | `widgets/run_header.py` (line 268) | Displays formatted elapsed time in header segment 6 |
| EsperStatus | `widgets/esper_status.py` (lines 92-96) | Displays human-readable runtime in status table |

**Implementation Details:**
- RunHeader calls `format_runtime(s.runtime_seconds)` to format as "1h 23m", "5m 30s", "45s", or "--"
- EsperStatus manually converts to hours/minutes/seconds for "Xh Ym Zs" display
- Both widgets check if `runtime_seconds > 0` to detect availability

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — Computed in `_get_snapshot_unlocked()` from `_start_time` and wall-clock time
- [x] **Transport works** — Assigned directly to snapshot field at `aggregator.py` line 551
- [x] **Schema field exists** — `SanctumSnapshot.runtime_seconds: float = 0.0` at schema.py line 1331
- [x] **Default is correct** — `0.0` indicates timing unavailable (not connected)
- [x] **Consumer reads it** — Both RunHeader and EsperStatus read `snapshot.runtime_seconds`
- [x] **Display is correct** — Formatted via `format_runtime()` function or manual conversion
- [x] **Thresholds applied** — No thresholds; metric used as-is for elapsed time display

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_runtime_computation` | `[ ]` |
| Unit (formatting) | `tests/karn/sanctum/test_formatting.py` | `test_format_runtime_*` | `[x]` |
| Integration (end-to-end) | `tests/integration/test_telemetry_flow.py` | `test_runtime_reaches_ui` | `[ ]` |
| Visual (widget snapshot) | — | Manual verification | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe RunHeader segment 6 showing runtime in format "Xh Ym", "Xm Ys", or "Xs"
4. Verify runtime increments by ~1 second per wall-clock second
5. Open system status widget (if available) and cross-check runtime display
6. Verify format transitions:
   - `0s - 59s` → "Xs" format
   - `60s - 3599s` → "Xm Ys" format
   - `3600s+` → "Xh Ym" format
   - `0s` → "--" (when disconnected)

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Aggregator connection status | infrastructure | `_connected` flag must be True for runtime > 0 |
| Time source | OS | `time.time()` must be synchronized; monotonic for accuracy |
| Aggregator initialization | system | `_start_time` initialized in `__init__` and reset on reconnection |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| RunHeader runtime display | display | Primary consumer, shows elapsed time in header segment 6 |
| EsperStatus table | display | Secondary consumer, shows runtime in system status panel |
| `format_runtime()` function | utility | Converts float seconds to human-readable format |
| Training duration metrics | analysis | Used in post-hoc analysis for training efficiency |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Initial creation - traced metric from aggregator to RunHeader and EsperStatus widgets |

---

## 8. Notes

### Design Decision

Runtime is computed **on-demand** during snapshot generation rather than maintained by a separate timer. This approach:
- Eliminates background timing logic
- Always reflects actual elapsed time from `_start_time` to `now`
- Scales naturally with snapshot poll frequency (250ms default)
- Resets on reconnection (`_start_time` reinit at line 278)

### Implementation Detail

The `_start_time` is initialized in two places:
1. **Initial construction** (`aggregator.py` line 192): Default to `time.time()` on class instantiation
2. **On connection** (`aggregator.py` line 278): Reset when backend re-establishes connection

When `get_snapshot()` is called (from UI poll loop), `_get_snapshot_unlocked()` computes elapsed time directly. The value is only non-zero when `_connected == True`.

### Formatting Function

The `format_runtime()` function in `/home/john/esper-lite/src/esper/karn/sanctum/formatting.py` (lines 13-47) converts seconds to human-readable format:

- `seconds <= 0` → `"--"`
- `seconds < 60` → `"Xs"` (e.g., "45s")
- `60 <= seconds < 3600` → `"Xm Ys"` (e.g., "5m 30s")
- `seconds >= 3600` → `"Xh Ym"` or `"Xh Ym Zs"` (default: `"Xh Ym"`, e.g., "1h 23m")

The `include_seconds_in_hours` parameter defaults to `False` for compact header display.

### Known Issue

**None documented** — metric is fully wired and actively consumed. All components properly integrated from aggregator to UI widgets.

### Widget-Specific Display Differences

- **RunHeader (line 268):** Uses `format_runtime(s.runtime_seconds)` for compact display; fixed-width 7-character output ensures no text jumping
- **EsperStatus (lines 92-96):** Manually converts seconds to "Xh Ym Zs" format with full seconds precision

Both approaches produce equivalent displays; RunHeader omits seconds for compactness per design spec.

### Threshold Justification

No thresholds are applied to runtime. The metric is used as-is for elapsed time display. Unlike staleness or entropy, runtime is purely informational and does not trigger alerts or status changes.

### Future Improvement

Consider tracking training time vs. wall-clock time separately to measure:
- Training efficiency (time spent in actual training vs. overhead)
- Pause/resume duration for interrupted sessions
- Per-episode training time for scaling analysis

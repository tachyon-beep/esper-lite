# Telemetry Record: [TELE-701] Connected

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-701` |
| **Name** | Connected |
| **Category** | `infrastructure` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Is the training backend connected and actively streaming telemetry to the TUI?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging backend connection issues)
- [ ] Researcher (not analytical)
- [x] Automated system (determines if TUI is in stale state)

### When is this information needed?

- [x] Real-time (every snapshot poll)
- [ ] Periodic (every N episodes)
- [ ] On-demand (when investigating issues)
- [ ] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `bool` |
| **Units** | Binary state (True/False) |
| **Range** | `[False, True]` |
| **Precision** | N/A (boolean) |
| **Default** | `False` |

### Semantic Meaning

> Boolean flag indicating whether the training backend (Tolaria/Simic) has successfully initiated communication with the Sanctum aggregator. Set to `True` at `TRAINING_STARTED` event, indicating the first valid telemetry has been received. Used in conjunction with `staleness_seconds` to determine overall connection health:
>
> - `connected=True, staleness<2s` → LIVE (fresh telemetry)
> - `connected=True, staleness<5s` → SLOW (delayed but present)
> - `connected=True, staleness≥5s` → STALE (data not updating)
> - `connected=False` → DISCONNECTED (no training started)

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `connected=True` | Backend actively transmitting |
| **Warning** | `connected=True, staleness>2s` | Backend connected but slow |
| **Critical** | `connected=False` | No backend connection |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Training start event from Tolaria |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `SanctumAggregator._handle_training_started()` |
| **Line(s)** | 616 |

```python
# Event handler sets connected=True when TRAINING_STARTED payload arrives
self._connected = True
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `TRAINING_STARTED` event payload (no explicit field) | `simic/tolaria/` (external) |
| **2. Collection** | Aggregator receives event | `aggregator.py` |
| **3. Aggregation** | Sets internal state `_connected` | `aggregator.py` line 616 |
| **4. Delivery** | Snapshot includes `connected` field | `schema.py` line 1330 |

```
[Tolaria] --TRAINING_STARTED--> [SanctumAggregator] --_connected=True--> [SanctumSnapshot.connected]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SanctumSnapshot` |
| **Field** | `connected` |
| **Path from SanctumSnapshot** | `snapshot.connected` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1330 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| RunHeader | `widgets/run_header.py` | Determines connection status indicator (● LIVE / ◐ SLOW / ○ STALE) |

**Usage Details:**
- RunHeader calls `_get_connection_status()` (line 65-85)
- Checks `snapshot.connected` at line 74
- Returns icon and color based on `connected` + `staleness_seconds`
- Displays as "● LIVE", "◐ SLOW", or "○ STALE/Disconnected"

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `_handle_training_started()` sets `_connected = True`
- [x] **Transport works** — Field reaches aggregator state during event processing
- [x] **Schema field exists** — `SanctumSnapshot.connected: bool = False`
- [x] **Default is correct** — `False` (not connected until TRAINING_STARTED arrives)
- [x] **Consumer reads it** — RunHeader checks `snapshot.connected` directly
- [x] **Display is correct** — Icon/color updates correctly based on connected state
- [ ] **Thresholds applied** — N/A (boolean, no numerical thresholds)

**WIRING STATUS: COMPLETE** ✓

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/karn/sanctum/test_aggregator.py` | `test_training_started_sets_connected` | `[ ]` |
| Unit (aggregator) | Same | `test_connected_flag_in_snapshot` | `[ ]` |
| Integration (end-to-end) | `tests/integration/test_backend_connection.py` | `test_ui_shows_live_after_training_starts` | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification | `[x]` |

### Manual Verification Steps

1. Start training: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (should auto-open or `uv run sanctum`)
3. Observe RunHeader left segment showing "● LIVE" (green)
4. Kill training process (`Ctrl+C`)
5. Wait 5 seconds
6. Verify RunHeader shows "○ Disconnected" (red)
7. Restart training
8. Verify immediately returns to "● LIVE"

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| TRAINING_STARTED event | event | Must arrive for connected to become True |
| Aggregator event processing | system | Must successfully deserialize event |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| RunHeader connection status | widget | Uses to display connection icon |
| Staleness computation | system | Used with `staleness_seconds` to determine overall health |
| UI initialization | system | TUI waits for `connected=True` before showing full UI |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Initial creation - infrastructure telemetry audit |

---

## 8. Notes

> **Design Decision:** Connection status is determined by the arrival of the TRAINING_STARTED event, not by continuous heartbeat. This simplifies implementation and matches user expectations: "are we getting data?"
>
> **Implementation Note:** `connected` is never explicitly set to False after becoming True. Staleness is tracked separately via `staleness_seconds` (computed from last event timestamp). This separation allows for better distinction between "backend crashed" vs. "backend is slow."
>
> **Staleness Coupling:** The true connection health is determined by BOTH `connected` AND `staleness_seconds`. RunHeader combines these:
> - `connected=False` → "○ Disconnected"
> - `connected=True, staleness<2s` → "● LIVE"
> - `connected=True, 2s≤staleness<5s` → "◐ SLOW"
> - `connected=True, staleness≥5s` → "○ STALE"
>
> **Potential Issue:** Once `connected=True`, it never resets. If training backend crashes and restarts, the TUI may show stale connection state until staleness threshold triggers visual change. This is acceptable behavior per design review.
>
> **TODO: [FUTURE FUNCTIONALITY]** - Add explicit heartbeat or connection timeout mechanism to explicitly reset `connected=False` if no events received for >10 seconds. Currently relies on staleness color coding alone.

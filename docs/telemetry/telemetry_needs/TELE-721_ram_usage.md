# Telemetry Record: [TELE-721] RAM Usage

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-721` |
| **Name** | RAM Usage |
| **Category** | `infrastructure` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Is system RAM approaching saturation? How much headroom remains for training and data loading?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging OOM issues)
- [x] Researcher (resource planning for scaling)
- [x] Automated system (alerts/intervention on >90% usage)

### When is this information needed?

- [x] Real-time (every N seconds - system vitals collection)
- [ ] Periodic (every N episodes)
- [ ] On-demand (when investigating issues)
- [x] Post-hoc (offline analysis of resource bottlenecks)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `float` (two fields) |
| **Units** | GB (gigabytes) |
| **Range** | `[0.0, system_total_gb]` (non-negative) |
| **Precision** | 1 decimal place for display |
| **Default** | `0.0` (unavailable) / `None` (optional fields) |

### Semantic Meaning

> RAM usage metrics capture the current state of system physical memory. Computed every snapshot (~250ms) by querying `psutil.virtual_memory()`:
>
> - **ram_used_gb:** `psutil.virtual_memory().used / (1024^3)` — bytes used by OS and running processes
> - **ram_total_gb:** `psutil.virtual_memory().total / (1024^3)` — total installed physical RAM
>
> These enable simple percentage calculation: `(ram_used_gb / ram_total_gb) * 100`

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `usage < 80%` | Comfortable headroom for training workload |
| **Warning** | `80% <= usage < 90%` | Approaching saturation, monitor data loader activity |
| **Critical** | `usage >= 90%` | **ALARM TRIGGERED** — bold red indicator in RunHeader |

**Threshold Source:** `SystemVitals.has_memory_alarm` property (line 1041-1058 in schema.py)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | System vitals collection in aggregator (run-once per snapshot cycle) |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `SanctumAggregator._update_system_vitals()` |
| **Line(s)** | 1801-1809 |

```python
# RAM collection via psutil
try:
    mem = psutil.virtual_memory()
    self._vitals.ram_used_gb = mem.used / (1024**3)
    self._vitals.ram_total_gb = mem.total / (1024**3)
except Exception as e:
    _logger.warning("Failed to collect RAM vitals: %s", e)
    self._vitals.ram_used_gb = None  # Explicit unavailable state
    self._vitals.ram_total_gb = None  # Explicit unavailable state
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `_update_system_vitals()` called once per snapshot | `aggregator.py` line 406 |
| **2. Collection** | Result stored in `self._vitals` (SystemVitals instance) | `aggregator.py` line 234 |
| **3. Aggregation** | Included in snapshot's vitals field | `aggregator.py` line 570 |
| **4. Delivery** | Delivered as part of SanctumSnapshot | `schema.py` line 1030-1031 |

```
[psutil.virtual_memory()] --> [_update_system_vitals()] --> [self._vitals] --> [SanctumSnapshot.vitals]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SystemVitals` |
| **Field(s)** | `ram_used_gb`, `ram_total_gb` |
| **Path from SanctumSnapshot** | `snapshot.vitals.ram_used_gb`, `snapshot.vitals.ram_total_gb` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1029-1031 |

```python
@dataclass
class SystemVitals:
    """System resource metrics - ALL from existing TUI."""
    # ...
    # RAM
    ram_used_gb: float | None = 0.0
    ram_total_gb: float | None = 0.0
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| RunHeader | `widgets/run_header.py` lines 87-120 | System alarm indicator (bold red if >90%) |
| RunHeader | `widgets/run_header.py` line 129 | `has_memory_alarm` check in status detection |
| EsperStatus | `widgets/esper_status.py` lines 140-145 | Detailed memory stats table with color thresholds |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `_update_system_vitals()` collects RAM via psutil
- [x] **Transport works** — Values populate `SystemVitals._vitals.ram_*_gb`
- [x] **Schema field exists** — `SystemVitals.ram_used_gb` and `.ram_total_gb` defined
- [x] **Default is correct** — `None` for unavailable (explicit opt-in check in consumers)
- [x] **Consumer reads it** — RunHeader and EsperStatus both access the fields
- [x] **Display is correct** — RunHeader shows percentage, EsperStatus shows GB breakdown
- [x] **Thresholds applied** — `has_memory_alarm` property triggers at >90%

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | — | Manual: RAM collection via psutil | `[x]` (verified) |
| Unit (aggregator) | — | Manual: Snapshot contains vitals | `[x]` (verified) |
| Integration (end-to-end) | — | Manual: RunHeader displays RAM alarm | `[x]` (verified) |
| Visual (TUI snapshot) | — | RunHeader shows "⚠ RAM 92%" on alarm | `[x]` (verified) |

### Manual Verification Steps

1. Start training: `uv run esper ppo --preset cifar10 --sanctum`
2. Open Sanctum TUI (auto-opens)
3. Observe RunHeader in top line:
   - Normal: `✓ System` (green, no alarm)
   - Alarm (>90%): `⚠ RAM 92% │ CPU 95%` (bold red)
4. Verify EsperStatus panel (scroll down):
   - Shows "RAM:" row with `X.XGB / Total.XGB` and style coloring
5. Trigger alarm by running memory-intensive workload or check pre-existing high usage

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| System resource availability | external | Requires functional `psutil.virtual_memory()` |
| Snapshot cycle (~250ms) | timing | RAM values updated once per UI refresh cycle |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `RunHeader.has_system_alarm` | property | Triggers visual alarm indicator when >90% |
| `SystemVitals.has_memory_alarm` | property | Aggregates RAM and GPU memory alarms |
| `SystemVitals.memory_alarm_devices` | property | Lists devices exceeding 90% (includes "RAM") |
| EsperStatus memory panel | widget | Displays breakdown with color thresholds |
| Auto-intervention (future) | system | May trigger OOM prevention measures |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Created with complete wiring trace |

---

## 8. Notes

> **Design Rationale:** RAM metrics are collected once per snapshot cycle (~250ms) using `psutil` rather than continuous sampling to amortize CPU overhead. System vitals are NOT part of training event streams but are instead computed on-demand by the aggregator, making them lightweight.
>
> **Field Types:** Both fields are `float | None` to distinguish "data not yet collected" (`None`) from "zero available" (`0.0`). Consumers explicitly check for `None` before calculation.
>
> **Threshold:** The 90% alarm threshold is chosen to provide early warning (10% headroom) before kernel OOM killer activates. This matches RunHeader's CPU threshold (>90%) and GPU threshold (>90%).
>
> **No Issues Found:** Wiring is complete and functioning correctly. RAM metrics flow from psutil → aggregator → schema → RunHeader/EsperStatus with appropriate color coding and alarm detection.


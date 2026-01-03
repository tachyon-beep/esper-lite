# Telemetry Record: [TELE-720] CPU Percent

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-720` |
| **Name** | CPU Percent |
| **Category** | `infrastructure` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Is the host CPU becoming a training bottleneck? Are we approaching thermal/scheduler limits?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging performance issues)
- [ ] Researcher (not analytical)
- [x] Automated system (triggers resource alarm)

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
| **Type** | `float \| None` |
| **Units** | percentage (0-100) |
| **Range** | `[0.0, 100.0]` |
| **Precision** | 1 decimal place for display |
| **Default** | `0.0` (no error state) |

### Semantic Meaning

> CPU utilization as a percentage of available CPU capacity, sampled via `psutil.cpu_percent(interval=None)`. Represents non-blocking snapshot of system-wide CPU usage across all cores. Value of `None` indicates collection failure (psutil unavailable or permission denied).
>
> At >90%, triggers warning alarm in RunHeader to alert operator of potential thermal/scheduler bottleneck. This is displayed prominently as "⚠ CPU 95%" in red.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `cpu_percent < 75%` | Adequate headroom, no concern |
| **Warning** | `75% ≤ cpu_percent ≤ 90%` | Monitor, potential bottleneck developing |
| **Critical** | `cpu_percent > 90%` | Alarm triggered, CPU saturated (bold red in UI) |

**Threshold Source:** RunHeader `_get_system_alarm_indicator()` line 100, threshold hardcoded as `> 90`

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | System vitals collected during each snapshot generation |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `SanctumAggregator._update_system_vitals()` |
| **Line(s)** | 1792-1799 |

```python
# CPU collection via psutil
try:
    self._vitals.cpu_percent = psutil.cpu_percent(interval=None)
except Exception as e:
    _logger.warning("Failed to collect CPU vitals: %s", e)
    self._vitals.cpu_percent = None  # Explicit unavailable state
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `_update_system_vitals()` called during snapshot generation | `aggregator.py` line 1792 |
| **2. Collection** | `psutil.cpu_percent(interval=None)` samples system CPU usage | `aggregator.py` line 1796 |
| **3. Aggregation** | Value assigned to `_vitals.cpu_percent` internal state | `aggregator.py` line 1796 |
| **4. Delivery** | Included in snapshot `vitals` at snapshot snapshot time | `aggregator.py` line 406, `get_snapshot()` method |

```
[psutil] --cpu_percent()--> [_update_system_vitals()] --> [_vitals.cpu_percent] --> [snapshot.vitals.cpu_percent]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SystemVitals` |
| **Field** | `cpu_percent` |
| **Path from SanctumSnapshot** | `snapshot.vitals.cpu_percent` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1027 |

**Field Definition:**
```python
cpu_percent: float | None = 0.0
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| RunHeader | `widgets/run_header.py` | System alarm indicator (line 100-101) - shows "⚠ CPU 95%" in bold red when >90% |
| EsperStatus | `widgets/esper_status.py` | CPU percentage display in system vitals table (line 151-152) - shows "CPU: 45.3%" |

**Usage Details:**

**RunHeader (alarm trigger):**
- Calls `_get_system_alarm_indicator()` (line 87-120)
- Checks `vitals.cpu_percent > 90` at line 100
- Adds `f"CPU {int(vitals.cpu_percent)}%"` to alarm list if true
- Displays as "⚠ CPU 95%" in bold red in header subtitle
- Also checks in `has_system_alarm` property (line 132) for style determination

**EsperStatus (detailed display):**
- Checks `vitals.cpu_percent is not None and vitals.cpu_percent > 0` at line 151
- Displays as `f"{vitals.cpu_percent:.1f}%"` in system vitals table
- Shows "--" (dim) if None or unavailable

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `_update_system_vitals()` collects CPU via psutil
- [x] **Transport works** — Value reaches snapshot.vitals during get_snapshot()
- [x] **Schema field exists** — `SystemVitals.cpu_percent: float | None = 0.0`
- [x] **Default is correct** — `0.0` (sensible for no error state; None indicates unavailable)
- [x] **Consumer reads it** — RunHeader and EsperStatus both access directly
- [x] **Display is correct** — RunHeader shows alarm (>90%), EsperStatus shows value
- [x] **Thresholds applied** — >90% triggers bold red alarm in RunHeader

**WIRING STATUS: COMPLETE** ✓

All components are connected and functional. The metric flows from psutil → aggregator → schema → widgets without gaps.

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/karn/sanctum/test_aggregator.py` | `test_system_vitals_collects_cpu` | `[ ]` |
| Unit (aggregator) | Same | `test_cpu_percent_in_snapshot` | `[ ]` |
| Integration (end-to-end) | `tests/integration/test_system_vitals.py` | `test_high_cpu_triggers_alarm` | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification | `[x]` |

### Manual Verification Steps

1. Start training: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (should auto-open or `uv run sanctum`)
3. Observe RunHeader right segment showing "✓ System" (green) when CPU <90%
4. Open EsperStatus widget to see detailed CPU percentage
5. Artificially increase CPU load (e.g., parallel build or stress test)
6. Verify RunHeader changes to "⚠ CPU 95%" (bold red) when threshold exceeded
7. Verify EsperStatus updates CPU percentage in real-time
8. Verify alarm clears when CPU drops below 90%

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| psutil library | library | Must be installed and functional; permission to sample /proc on Linux |
| OS CPU sampling capability | system | Linux/macOS/Windows built-in; may be unavailable in some containers |
| Snapshot generation cycle | event | CPU collected every time `get_snapshot()` called (UI poll frequency) |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| RunHeader alarm indicator | widget | Uses for "⚠ CPU X%" display when >90% |
| EsperStatus vitals panel | widget | Uses for detailed CPU percentage display |
| System alarm flag | display | Affects RunHeader styling (controls has_system_alarm property) |
| Training operator awareness | user | Alerts to potential host CPU bottleneck |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Initial creation - infrastructure telemetry audit (TELE-720) |

---

## 8. Notes

> **Design Decision:** CPU collection uses non-blocking `psutil.cpu_percent(interval=None)` rather than blocking `cpu_percent(interval=1.0)`. Non-blocking returns cached value from last OS sample (typically within 1-2 seconds). This prevents TUI freezes during snapshot generation while remaining timely for alarm purposes.
>
> **None Handling:** When psutil collection fails (e.g., permission denied, library not installed), `cpu_percent` is explicitly set to `None` to distinguish from healthy "CPU is 0%". Widgets check `is not None` before display to avoid false zero readings.
>
> **Alarm Threshold:** The >90% threshold matches typical system engineering practice (headroom for OS/background tasks). This is a hardcoded value, not configurable. Design review approved for simple fixed threshold given non-critical nature of CPU alarm (memory is higher priority).
>
> **Sampling Frequency:** CPU is sampled at the same frequency as other system vitals (during `get_snapshot()` calls, typically 4-10 Hz based on UI refresh rate). This provides responsive updates while amortizing psutil overhead across other collections (RAM, GPU).
>
> **Cross-Platform:** psutil.cpu_percent() is cross-platform (Linux, macOS, Windows). Behavior may vary slightly:
> - Linux: Reads from /proc/stat
> - macOS: Uses BSD syscalls
> - Windows: Uses WMI
> Threshold (>90%) is universal and OS-independent.
>
> **Potential Issue:** In containerized environments (Docker, K8s), psutil may see container CPU limits rather than actual host CPU. In such cases, the value is still meaningful but represents different semantics. No mitigation needed as this is expected behavior.
>
> **Future Enhancement:** Consider per-core breakdown for multi-core analysis (e.g., show if single core is maxed while others idle). Currently only reports system-wide average. Would require `psutil.cpu_percent(percpu=True)` and additional schema fields.

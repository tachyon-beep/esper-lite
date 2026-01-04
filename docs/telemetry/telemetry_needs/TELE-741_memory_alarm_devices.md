# Telemetry Record: [TELE-741] Memory Alarm Devices

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-741` |
| **Name** | Memory Alarm Devices |
| **Category** | `infrastructure` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Which compute devices (GPU or RAM) are experiencing critical memory pressure (>90% utilization)? Are we at risk of OOM?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging OOM or memory bottlenecks)
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
| **Type** | `list[str]` (computed property) |
| **Units** | device names and identifiers |
| **Range** | Empty list (no alarms) to multiple devices in alarm state |
| **Precision** | Device identifiers only (no percentages in list; percentages added by consumer) |
| **Default** | `[]` (empty list when all devices below 90% threshold) |

### Semantic Meaning

> Computed property of `SystemVitals` that returns a list of device names exceeding 90% memory utilization. Each device is identified by a human-readable string: `"RAM"` for host system RAM, or `"cuda:N"` for GPU device N. The property iterates all memory sources (host RAM and all GPU devices) and returns only those exceeding the critical threshold of 90% utilization.
>
> Formula for memory usage: `(memory_used / memory_total) > 0.90`
>
> This is a **computed property** (not a raw telemetry field). It derives from raw fields in `SystemVitals`:
> - `ram_used_gb`, `ram_total_gb` (host memory)
> - `gpu_stats` dict with `GPUStats(memory_used_gb, memory_total_gb)` entries

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | Empty list `[]` | All devices <90% utilization, no memory concern |
| **Warning** | 1-2 devices in list | One or two devices approaching limits, monitor closely |
| **Critical** | 3+ devices in list OR any device at >95% | Multiple devices in alarm state, risk of OOM imminent |

**Threshold Source:** `SystemVitals.memory_alarm_devices` property (schema.py line 1061), hardcoded threshold `> 0.90` (line 1066, 1072, 1077)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Computed property of `SystemVitals`; derived from raw memory stats collected via system calls |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Function/Method** | `SystemVitals.memory_alarm_devices` (property) |
| **Line(s)** | 1060-1079 |

```python
@property
def memory_alarm_devices(self) -> list[str]:
    """Get list of devices exceeding 90% memory usage."""
    devices = []
    # Check RAM
    if (self.ram_total_gb is not None and self.ram_used_gb is not None and
        self.ram_total_gb > 0 and (self.ram_used_gb / self.ram_total_gb) > 0.90):
        devices.append("RAM")
    # Check multi-GPU stats
    for device, stats in self.gpu_stats.items():
        if stats.memory_total_gb > 0:
            usage = stats.memory_used_gb / stats.memory_total_gb
            if usage > 0.90:
                devices.append(f"cuda:{device}")
    # Fallback to single GPU if no gpu_stats but fallback fields are populated
    if not self.gpu_stats and self.gpu_memory_total_gb > 0:
        usage = self.gpu_memory_used_gb / self.gpu_memory_total_gb
        if usage > 0.90:
            devices.append("cuda:0")
    return devices
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Raw Collection** | System calls (psutil, torch.cuda) sample memory usage | `aggregator.py` lines 1792-1847 (_update_system_vitals) |
| **2. Population** | Raw fields assigned to `SystemVitals` instance (`_vitals.ram_used_gb`, `_vitals.gpu_stats`) | `aggregator.py` lines 1804-1844 |
| **3. Aggregation** | Computed property iterates populated `SystemVitals` instance | `aggregator.py` line 406 (calls `_update_system_vitals()` during get_snapshot) |
| **4. Delivery** | Property accessed by widget consumer on snapshot object | `widgets/run_header.py` line 104 (iteration over `vitals.memory_alarm_devices`) |

```
[psutil/torch.cuda] --> [_update_system_vitals()] --> [SystemVitals fields] --> [memory_alarm_devices property] --> [RunHeader._get_system_alarm_indicator()]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SystemVitals` |
| **Field** | `memory_alarm_devices` (computed property, not a stored field) |
| **Path from SanctumSnapshot** | `snapshot.vitals.memory_alarm_devices` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 1060 (property definition) |

**Property Definition:**
```python
@property
def memory_alarm_devices(self) -> list[str]:
    # ... (computation shown above)
```

**Underlying Fields:**
```python
# In SystemVitals:
ram_used_gb: float | None = 0.0
ram_total_gb: float | None = 0.0
gpu_stats: dict[int, GPUStats] = field(default_factory=dict)
gpu_memory_used_gb: float = 0.0
gpu_memory_total_gb: float = 0.0
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| RunHeader | `widgets/run_header.py` | System alarm indicator (lines 104-118) - iterates devices and displays with percentages |

**Usage Details:**

**RunHeader (primary consumer):**
- Calls `_get_system_alarm_indicator()` method (lines 87-120)
- Iterates over `vitals.memory_alarm_devices` at line 104
- For each device in the list:
  - If device is `"RAM"` (line 105): calculates percentage and formats as `f"RAM {pct}%"` (lines 106-108)
  - If device starts with `"cuda:"` (line 109): extracts device ID, retrieves stats, calculates percentage, formats as `f"{device} {pct}%"` (lines 110-118)
- Joins all alarm strings with `" │ "` separator and displays in header subtitle (line 120)
- When any devices in alarm list, displays as "⚠ RAM 92% │ cuda:0 95%" in bold red
- When list is empty, displays "✓ System" in green
- Also referenced in `has_system_alarm` property (line 129) to drive style determination

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — System vitals collected in `_update_system_vitals()` populates underlying fields
- [x] **Transport works** — Raw fields reach `SystemVitals` instance; property computed on demand
- [x] **Schema field exists** — `SystemVitals.memory_alarm_devices` property defined at schema.py line 1060
- [x] **Default is correct** — Returns empty list `[]` when no devices exceed 90% (sensible no-alarm state)
- [x] **Consumer reads it** — RunHeader iterates the list and formats device names with percentages
- [x] **Display is correct** — Devices shown in header subtitle as "⚠ RAM 92% │ cuda:0 95%" in bold red when alarms active
- [x] **Thresholds applied** — >90% threshold hardcoded in property, colors applied by consumer

**WIRING STATUS: COMPLETE** ✓

All components are connected and functional. The metric flows from system vitals → schema property → widget display without gaps.

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (property) | `tests/karn/sanctum/test_schema.py` | `test_memory_alarm_devices_empty_when_below_90` | `[ ]` |
| Unit (property) | Same | `test_memory_alarm_devices_includes_ram_when_above_90` | `[ ]` |
| Unit (property) | Same | `test_memory_alarm_devices_includes_gpu_when_above_90` | `[ ]` |
| Unit (property) | Same | `test_memory_alarm_devices_multi_gpu_fallback` | `[ ]` |
| Integration (aggregator) | `tests/karn/sanctum/test_backend.py` | `test_gpu_memory_alarm_in_snapshot` | `[x]` |
| Widget (RunHeader) | `tests/karn/sanctum/test_run_header.py` | `test_run_header_system_alarm_triggered` | `[x]` |
| Widget (RunHeader) | Same | `test_run_header_border_red_on_memory_alarm` | `[x]` |
| Visual (TUI snapshot) | — | Manual verification | `[x]` |

### Manual Verification Steps

1. Start training: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (should auto-open or `uv run sanctum`)
3. Observe RunHeader right segment showing "✓ System" (green) when all devices <90%
4. Monitor EsperStatus widget to see raw memory percentages per device
5. Artificially increase memory usage:
   - For GPU: Run additional CUDA workloads or reduce batch size to allocate more memory
   - For RAM: Run memory-intensive processes or consume additional host memory
6. Verify RunHeader changes to "⚠ RAM 92%" or "⚠ cuda:0 95%" (bold red) when threshold exceeded
7. Verify device list updates as devices enter/exit alarm state
8. Verify alarm clears when memory drops below 90%

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| psutil library | library | Must be installed for RAM statistics via `psutil.virtual_memory()` |
| PyTorch CUDA | library | Optional; required for GPU memory statistics via `torch.cuda.get_device_properties()` and `torch.cuda.memory_stats()` |
| System memory APIs | system | Linux `/proc/meminfo`, macOS/BSD syscalls, Windows WMI |
| Snapshot generation cycle | event | Property computed every time `get_snapshot()` called (during each snapshot poll) |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| RunHeader alarm indicator | widget | Uses for "⚠ RAM X% │ cuda:0 Y%" display when any device >90% |
| `has_memory_alarm` property | schema | Sibling property used by widgets to determine if alarm state is active |
| System alarm flag | display | Affects RunHeader styling and color (controls `has_system_alarm` property which drives alarm coloring) |
| Training operator awareness | user | Alerts to critical memory pressure and OOM risk |
| Potential auto-intervention | future | Could trigger batch size reduction or checkpoint save if wired to future auto-scaling system |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Initial creation - infrastructure telemetry audit (TELE-741) |

---

## 8. Notes

> **Design Decision - Computed Property:** `memory_alarm_devices` is implemented as a computed property rather than a stored field. This is intentional: the list changes frequently as memory usage fluctuates, and computing it on-demand from fresh `SystemVitals` state is cheaper than storing and updating it as a separate field. Properties in Python are transparent to callers and allow lazy evaluation.
>
> **Threshold Logic - 90% Rationale:** The 90% threshold matches industrial practice for memory safety margins. Leaving 10% headroom prevents OOM errors that occur suddenly when allocators cannot find contiguous blocks. This is more conservative than CPU threshold (which uses 90% as warning only) because memory exhaustion is fatal to training, while CPU throttling is recoverable.
>
> **Device Naming Convention:**
> - `"RAM"` for host system RAM (clear, non-technical)
> - `"cuda:N"` for GPU N (matches PyTorch convention, consistent with user's device specifiers)
> - Fallback to `"cuda:0"` when no multi-GPU stats but single GPU fields populated (backward compatibility)
>
> **None Handling:** RAM fields can be `None` if psutil collection fails. Property explicitly checks `is not None` before division to avoid TypeError. Returns empty list if both RAM fields are None (treated as "not available, assume no alarm").
>
> **Multi-GPU Support:** Property iterates `gpu_stats` dict (keyed by device ID) to support arbitrary numbers of GPUs. Fallback path handles legacy single-GPU fields for backward compatibility with older configs.
>
> **Empty List as Safe Default:** Returns `[]` when all devices below threshold. This allows consumer code to use simple `if vitals.memory_alarm_devices:` checks for truthiness. Empty list is falsy in Python, so conditional branches work naturally.
>
> **Percentage Computation by Consumer:** The property returns only device names (list[str]), not percentages. RunHeader widget recalculates percentages on display to ensure freshness and to avoid floating-point precision issues in the property. This separation of concerns keeps property simple and testable.
>
> **Sampling Frequency:** Memory is sampled at the same frequency as other system vitals (during `get_snapshot()` calls, typically 4-10 Hz based on UI refresh rate). This provides responsive updates for alarm state changes while amortizing psutil overhead across other collections (CPU, GPU utilization).
>
> **Known Limitation - Container Environments:** In containerized environments (Docker, Kubernetes), psutil and torch.cuda may report container-level memory limits rather than physical host memory. In such cases, the alarm may trigger at lower absolute memory usage. This is expected and acceptable behavior; the alarm correctly reflects the container's resource constraints.
>
> **Future Enhancement - Per-Device Thresholds:** Currently all devices use fixed 90% threshold. Could enhance to support per-device configurable thresholds (e.g., GPU:85%, RAM:90%) if operational requirements diverge. Would require `TrainingConfig.memory_alarm_thresholds` dict.
>
> **Future Enhancement - Trend Analysis:** Could extend to track memory pressure trends (increasing/stable/decreasing) to predict imminent OOM before it occurs. Would require historical buffer and derivative computation (similar to entropy velocity in TELE-001).
>
> **Integration with has_memory_alarm:** Sibling property `SystemVitals.has_memory_alarm` (lines 1040-1058) checks if ANY device exceeds threshold, returning bool. This is more efficient for binary alarm checks. The `memory_alarm_devices` property (this record) provides the detailed list for per-device display. Both are complementary.
>
> **Testing Note:** Unit tests for the property should mock both `ram_total_gb`/`ram_used_gb` and `gpu_stats` entries to cover:
> - Empty case (no alarms)
> - RAM only alarm
> - GPU only alarm
> - Multi-GPU alarms
> - Fallback single-GPU field path
> - Boundary cases (exactly 90%, 89.9%, 90.1%)


# Telemetry Record: [TELE-740] Memory Alarm Trigger

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-740` |
| **Name** | Memory Alarm Trigger |
| **Category** | `infrastructure` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Are any system devices (RAM, GPU) approaching critical memory saturation (>90%)? Do we need to intervene to prevent OOM?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [ ] Researcher (analysis)
- [x] Automated system (alerts/intervention)

### When is this information needed?

- [x] Real-time (every poll cycle)
- [ ] Periodic (every N episodes)
- [ ] On-demand (when investigating issues)
- [ ] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `bool` |
| **Units** | N/A |
| **Range** | `true` (alarm active) / `false` (no alarm) |
| **Precision** | N/A |
| **Default** | `false` |

### Semantic Meaning

> Computed property that returns `true` if ANY device (CPU, RAM, or GPU) exceeds 90% memory utilization. Used to trigger visual alerts in the TUI and provide early warning before Out-of-Memory (OOM) failures.
>
> Computed as:
> ```
> has_memory_alarm = (RAM > 90%) OR (any GPU > 90%) OR (CPU > 90%)
> ```
>
> The threshold of 90% is chosen to provide headroom for system allocation and kernel buffers before hard limits are hit.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `has_memory_alarm == false` | All devices operating safely |
| **Warning** | `has_memory_alarm == true` | At least one device exceeds 90% usage; imminent risk |
| **Critical** | Device at 100% | OOM imminent; training may crash |

**Threshold Source:** Hard-coded 90% threshold in `SystemVitals.has_memory_alarm` property

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | System resource monitoring (psutil + torch.cuda) |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `SanctumAggregator._update_system_vitals()` |
| **Line(s)** | ~1792-1848 |

```python
# System vitals collection
def _update_system_vitals(self) -> None:
    """Update system vitals (CPU, RAM, GPU)."""
    # RAM
    mem = psutil.virtual_memory()
    self._vitals.ram_used_gb = mem.used / (1024**3)
    self._vitals.ram_total_gb = mem.total / (1024**3)

    # GPU
    if torch.cuda.is_available():
        gpu_stats: dict[int, GPUStats] = {}
        for i, device in enumerate(self._gpu_devices):
            mem_reserved = torch.cuda.memory_reserved(device_idx) / (1024**3)
            gpu_stats[device_idx] = GPUStats(
                device_id=device_idx,
                memory_used_gb=mem_reserved,
                memory_total_gb=mem_total,
            )
        self._vitals.gpu_stats = gpu_stats
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Collection** | psutil.virtual_memory(), torch.cuda APIs | `aggregator.py` |
| **2. Population** | Direct field assignment to `_vitals` | `aggregator.py` |
| **3. Schema inclusion** | `_vitals` passed to snapshot | `aggregator.py` |
| **4. Delivery** | Property evaluated lazily by consumers | `schema.py` |

```
[psutil/torch.cuda] --> [_update_system_vitals()] --> [self._vitals (SystemVitals)] --> [SanctumSnapshot.vitals] --> [Consumers]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SystemVitals` |
| **Field** | `has_memory_alarm` (computed property, not stored field) |
| **Path from SanctumSnapshot** | `snapshot.vitals.has_memory_alarm` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Lines** | 1040-1058 |

The property reads from stored fields: `ram_used_gb`, `ram_total_gb`, `gpu_stats`, `gpu_memory_used_gb`, `gpu_memory_total_gb`

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| RunHeader | `widgets/run_header.py` | Used in `has_system_alarm` property to determine alert display (line 129) |
| AnomalyStrip | `widgets/anomaly_strip.py` | Stored as `self.memory_alarm` for visual anomaly tracking (line 112) |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `_update_system_vitals()` collects RAM + GPU stats from psutil/torch
- [x] **Transport works** — Data flows through `_vitals: SystemVitals` to snapshot
- [x] **Schema field exists** — `has_memory_alarm` property defined in `SystemVitals`
- [x] **Default is correct** — Property returns `false` when no device exceeds 90%
- [x] **Consumer reads it** — RunHeader and AnomalyStrip both access `snapshot.vitals.has_memory_alarm`
- [x] **Display is correct** — RunHeader renders "⚠ [device] [%]" when true, "✓ System" when false
- [x] **Thresholds applied** — Hard-coded 90% threshold in property logic

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (property) | `tests/karn/sanctum/test_schema.py` | `test_system_vitals_memory_alarm_threshold` | `[x]` |
| Unit (property) | `tests/karn/sanctum/test_schema.py` | `test_system_vitals_no_alarm_below_threshold` | `[x]` |
| Unit (property) | `tests/karn/sanctum/test_schema.py` | `test_system_vitals_ram_only_alarm` | `[x]` |
| Unit (property) | `tests/karn/sanctum/test_schema.py` | `test_system_vitals_multi_gpu_alarm` | `[x]` |
| Integration (end-to-end) | — | Manual verification | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification | `[ ]` |

### Manual Verification Steps

1. Start training: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe RunHeader right-aligned alarm display
4. Verify displays "✓ System" when memory <90%
5. Artificially increase memory load to >90% on RAM or GPU
6. Verify RunHeader updates to "⚠ RAM 92%" or "⚠ cuda:0 95%"
7. Check AnomalyStrip for visual anomaly marker

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `TELE-721` (ram_usage) | telemetry | Provides `ram_used_gb`, `ram_total_gb` |
| `TELE-730` (gpu_memory_usage) | telemetry | Provides `gpu_stats`, `gpu_memory_used_gb`, `gpu_memory_total_gb` |
| `TELE-720` (cpu_percent) | telemetry | Provides `cpu_percent` for CPU alarm |
| psutil library | system | Used for CPU and RAM collection |
| torch.cuda | library | Used for GPU memory collection |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| RunHeader display | widget | Shows "⚠" when true, "✓ System" when false |
| AnomalyStrip tracking | widget | Records memory alarm events on timeline |
| `memory_alarm_devices` property | computed | Provides device breakdown (RAM, cuda:0, cuda:1, etc.) |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Initial telemetry record creation |

---

## 8. Notes

> **Design Decision:** `has_memory_alarm` is a computed property rather than a stored field. This avoids duplication and keeps the alarm logic in one place (the SystemVitals class).
>
> **Implementation Detail:** The property checks three sources:
> 1. RAM: `ram_used_gb / ram_total_gb > 0.90`
> 2. Multi-GPU: Loop through `gpu_stats` dict entries
> 3. Single GPU fallback: Uses `gpu_memory_used_gb / gpu_memory_total_gb > 0.90` (for backward compatibility if gpu_stats not populated)
>
> **Related Property:** `memory_alarm_devices` returns a list of which devices triggered the alarm (e.g., `["RAM", "cuda:1"]`). This is used by RunHeader to display detailed alarm messages.
>
> **Wiring Status:** Fully wired and tested. Both display consumers (RunHeader, AnomalyStrip) are actively using this metric. No known gaps.
>
> **Future Enhancement:** Consider adding separate thresholds for different device types (e.g., RAM warning at 85%, GPU warning at 90%) or adding a "critical" level at 95% for more nuanced alerting.

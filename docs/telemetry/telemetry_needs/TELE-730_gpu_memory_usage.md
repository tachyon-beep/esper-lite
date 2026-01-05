# Telemetry Record: [TELE-730] GPU Memory Usage

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-730` |
| **Name** | GPU Memory Usage |
| **Category** | `infrastructure` |
| **Priority** | `P0-critical` |

## 2. Purpose

### What question does this answer?

> "Is the GPU memory usage approaching dangerous levels (>90% of total), and could an out-of-memory error occur on the next allocation?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging OOM issues)
- [x] Researcher (understanding memory scaling)
- [x] Automated system (alerts/intervention)

### When is this information needed?

- [x] Real-time (every snapshot update)
- [ ] Periodic (every N episodes)
- [ ] On-demand (when investigating issues)
- [ ] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `float` (per-GPU) |
| **Units** | GB used / GB total |
| **Range** | `[0.0, total_memory_gb]` |
| **Precision** | 1 decimal place for display (e.g., "5.2GB") |
| **Default** | `0.0` when CUDA unavailable |

### Semantic Meaning

> GPU memory usage represents the amount of CUDA memory reserved (allocated footprint) by PyTorch. Uses `torch.cuda.memory_reserved()` (not allocated) to surface OOM risk earlier.
>
> The percentage is computed as: `(memory_used_gb / memory_total_gb) * 100`
>
> High percentage = imminent OOM risk. This metric is per-GPU for multi-GPU support, with convenience fields for single-GPU access.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `< 80%` | Normal operating range, safe headroom |
| **Warning** | `80-90%` | Approaching capacity, monitor closely |
| **Critical** | `>90%` | Alarm state, OOM likely on next allocation |

**Threshold Source:** `SystemVitals.has_memory_alarm` property (>90%), displayed in RunHeader with bold red styling

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PyTorch CUDA memory tracking (no explicit emission) |
| **File** | `src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `SanctumAggregator._update_system_stats()` (lines 1811-1848) |
| **Line(s)** | ~1821 |

```python
# GPU memory reserved is queried directly from PyTorch
mem_reserved = torch.cuda.memory_reserved(device_idx) / (1024**3)
gpu_stats[device_idx] = GPUStats(
    device_id=device_idx,
    memory_used_gb=mem_reserved,
    memory_total_gb=mem_total,
)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Collection** | Direct torch.cuda query (no event emission) | `aggregator.py` |
| **2. Storage** | `SystemVitals.gpu_stats[device_id]` dict | `schema.py` |
| **3. Convenience Fields** | `SystemVitals.gpu_memory_used_gb` (GPU 0 fallback) | `schema.py` |
| **4. Delivery** | `snapshot.vitals` passed to all widgets | `aggregator.get_snapshot()` |

```
[torch.cuda.memory_reserved()]
  --> [aggregator._update_system_stats()]
  --> [SystemVitals.gpu_stats, gpu_memory_used_gb]
  --> [snapshot.vitals]
  --> [RunHeader widget]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `GPUStats` (per-GPU) and `SystemVitals` (aggregate) |
| **Fields** | `GPUStats.memory_used_gb`, `GPUStats.memory_total_gb` |
| **Path from SanctumSnapshot** | `snapshot.vitals.gpu_stats[device_id].memory_used_gb` |
| **Fallback Path** | `snapshot.vitals.gpu_memory_used_gb` (single GPU) |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Lines** | GPUStats (425-436), SystemVitals (1012-1080) |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| RunHeader | `widgets/run_header.py` (lines 103-120) | Alarm indicator with percentage, bold red when >90% |
| EsperStatus | `widgets/esper_status.py` | GPU memory detail display with "X.XGB / X.XGB" format |
| SystemVitals | (via property `has_memory_alarm`) | Triggers alarm state, marks devices exceeding 90% |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — PyTorch directly queries memory via torch.cuda.memory_reserved()
- [x] **Transport works** — Aggregator collects into SystemVitals.gpu_stats on each snapshot
- [x] **Schema field exists** — GPUStats.memory_used_gb and memory_total_gb defined
- [x] **Default is correct** — 0.0 when CUDA unavailable (explicit state)
- [x] **Consumer reads it** — RunHeader accesses snapshot.vitals.gpu_stats and calculates percentage
- [x] **Display is correct** — Percentage formatted as "cuda:0 95%" with bold red style
- [x] **Thresholds applied** — >90% triggers memory_alarm_devices property, displayed as alarm

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (aggregator) | `tests/karn/sanctum/test_backend.py` | GPU stats collection | `[x]` |
| Integration (snapshot) | `tests/karn/sanctum/test_schema.py` | SystemVitals.has_memory_alarm | `[x]` |
| Widget (display) | `tests/karn/sanctum/test_run_header.py` | test_run_header_system_alarm_triggered | `[x]` |
| Visual (TUI snapshot) | — | Manual verification | `[x]` |

### Manual Verification Steps

1. Start training: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe RunHeader at top of screen
4. Verify GPU memory shows as "OK" when <90%
5. Monitor GPU memory during long training run
6. Verify "⚠ cuda:0 95%" appears in bold red when >90%
7. Check EsperStatus widget for detailed "X.XGB / Y.YGB" display

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PyTorch CUDA availability | external | Requires torch.cuda.is_available() |
| GPU device list | configuration | From aggregator initialization (_gpu_devices) |
| Total memory query | PyTorch API | torch.cuda.get_device_properties(device_idx).total_memory |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| RunHeader alarm indicator | widget | Triggers bold red "⚠" when memory_alarm_devices non-empty |
| SystemVitals.has_memory_alarm | property | Used by RunHeader._get_system_alarm_indicator() |
| EsperStatus detail panel | widget | Displays full memory statistics for all GPUs |
| Governor OOM detection | system | May trigger emergency actions on memory pressure |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Audit | Initial creation - TELE-730 audit |

---

## 8. Notes

> **Design Decision:** Uses `torch.cuda.memory_reserved()` rather than `memory_allocated()` because reserved memory (allocator footprint) gives earlier warning of OOM risk. Allocated memory can suddenly jump to reserved on next allocation attempt.
>
> **Multi-GPU Support:** Stored in `SystemVitals.gpu_stats: dict[int, GPUStats]` for per-GPU tracking. Convenience fields `gpu_memory_used_gb` and `gpu_memory_total_gb` provide single-GPU fallback for display.
>
> **Alarm Threshold:** Hardcoded at 90% per infrastructure best practices (OOM typically occurs at 95%+, so 90% gives 5% safety margin).
>
> **Zero-Default Behavior:** When CUDA unavailable, gpu_stats set to empty dict {} explicitly. Widgets check dict contents before accessing.
>
> **Wiring Status:** FULLY WIRED. GPU memory collection -> aggregator -> schema -> widgets all connected and tested. Threshold alarm mechanism working (>90% triggers bold red display).


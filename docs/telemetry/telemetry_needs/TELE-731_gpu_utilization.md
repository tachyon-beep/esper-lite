# Telemetry Record: [TELE-731] GPU Utilization

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-731` |
| **Name** | GPU Utilization |
| **Category** | `infrastructure` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Is the GPU compute capacity being fully utilized, or is the training bottlenecked by data loading, CPU preprocessing, or other factors?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging performance)
- [x] Researcher (understanding bottlenecks)
- [ ] Automated system (alerts/intervention)

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
| **Units** | percentage (0-100%) |
| **Range** | `[0.0, 100.0]` |
| **Precision** | 0 decimal places for display (e.g., "85%") |
| **Default** | `0.0` when pynvml unavailable or no GPU activity |

### Semantic Meaning

> GPU utilization measures the percentage of time the GPU's streaming multiprocessors (SMs) are active over a sampling period. This is distinct from memory usage:
>
> - **High utilization (>80%):** GPU is compute-bound, working efficiently
> - **Low utilization (<50%) with high memory:** Data loading bottleneck or CPU-bound preprocessing
> - **Low utilization with low memory:** Underutilized GPU, batch size may be too small
>
> Utilization is queried via pynvml (`nvmlDeviceGetUtilizationRates`) when available. Falls back to 0.0 when pynvml is not installed or fails.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `< 80%` | Normal range, some headroom |
| **Warning** | `80-95%` | High utilization, near capacity |
| **Critical** | `> 95%` | Saturated, may cause thermal throttling |

**Threshold Source:** `EsperStatus` widget (lines 115-121) applies color coding based on these thresholds

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | pynvml GPU utilization query |
| **File** | `src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `SanctumAggregator._update_system_stats()` |
| **Line(s)** | ~1830-1840 |

```python
# GPU utilization from pynvml (when available)
try:
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    utilization = util.gpu  # SM utilization percentage
except Exception:
    utilization = 0.0

gpu_stats[device_idx] = GPUStats(
    device_id=device_idx,
    memory_used_gb=mem_reserved,
    memory_total_gb=mem_total,
    utilization=utilization,
)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Collection** | pynvml query in aggregator | `aggregator.py` |
| **2. Storage** | `GPUStats.utilization` field | `schema.py` |
| **3. Legacy Field** | `SystemVitals.gpu_utilization` (GPU 0 fallback) | `schema.py` |
| **4. Delivery** | `snapshot.vitals` passed to widgets | `aggregator.get_snapshot()` |

```
[pynvml.nvmlDeviceGetUtilizationRates()]
  --> [aggregator._update_system_stats()]
  --> [GPUStats.utilization]
  --> [SystemVitals.gpu_stats[device_id].utilization]
  --> [EsperStatus widget]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `GPUStats` |
| **Field** | `utilization: float = 0.0` |
| **Path from SanctumSnapshot** | `snapshot.vitals.gpu_stats[device_id].utilization` |
| **Legacy Path** | `snapshot.vitals.gpu_utilization` (single GPU fallback) |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Lines** | GPUStats (425-436) |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EsperStatus | `widgets/esper_status.py` (lines 114-121) | Displays "GPU util: 85%" with color coding |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — pynvml queries utilization in aggregator._update_system_stats()
- [x] **Transport works** — Stored in GPUStats.utilization field
- [x] **Schema field exists** — `GPUStats.utilization: float = 0.0`
- [x] **Default is correct** — 0.0 when pynvml unavailable (graceful degradation)
- [x] **Consumer reads it** — EsperStatus accesses stats.utilization
- [x] **Display is correct** — Formatted as "85%" with color thresholds
- [x] **Thresholds applied** — Green <80%, yellow 80-95%, red >95%

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (schema) | `tests/karn/sanctum/test_schema.py` | GPUStats default values | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_backend.py` | GPU stats collection | `[ ]` |
| Widget (display) | `tests/karn/sanctum/widgets/test_esper_status.py` | Utilization display | `[ ]` |

### Manual Verification Steps

1. Start training with GPU: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or run separately)
3. Observe EsperStatus panel in left column
4. Verify "GPU util:" row appears with percentage
5. Verify color: green (<80%), yellow (80-95%), red (>95%)
6. During active training, utilization should be >50%
7. During data loading, utilization may drop temporarily

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| pynvml library | optional | Falls back to 0.0 if unavailable |
| NVIDIA GPU | hardware | Requires NVIDIA driver with NVML support |
| CUDA availability | runtime | torch.cuda.is_available() must be True |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| EsperStatus widget | display | Shows utilization percentage with color coding |
| Performance analysis | diagnostic | Helps identify data loading bottlenecks |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation - gap identified in EsperStatus audit |

---

## 8. Notes

> **Design Decision:** GPU utilization is queried via pynvml rather than PyTorch because PyTorch doesn't expose SM utilization metrics. pynvml is optional—if not installed, utilization gracefully defaults to 0.0.
>
> **Interpretation Guide:**
> - **High util + high memory:** Compute-bound, optimal
> - **Low util + high memory:** Data loading or CPU bottleneck
> - **Low util + low memory:** Underutilized, increase batch size
> - **Fluctuating util:** Normal during training with data loading phases
>
> **vs. TELE-730 (GPU Memory):** Memory usage predicts OOM errors. Utilization predicts throughput bottlenecks. Both are needed for complete GPU health picture.
>
> **Multi-GPU:** Each GPU in `gpu_stats` dict has its own utilization. Legacy field `gpu_utilization` provides GPU 0 fallback for single-GPU setups.
>
> **Wiring Status:** FULLY WIRED. pynvml collection -> GPUStats -> EsperStatus display all connected.

# Telemetry Record: [TELE-770] GPU Memory Usage Percentage

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-770` |
| **Name** | GPU Memory Usage Percentage |
| **Category** | `infrastructure` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "How much of the reserved GPU memory is currently being used? Is GPU memory pressure affecting training?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [x] Automated system (alerts/intervention)

### When is this information needed?

- [x] Real-time (every batch/epoch)
- [ ] Periodic (every N episodes)
- [ ] On-demand (when investigating issues)
- [x] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `float` (computed property) |
| **Units** | percentage (0-100) |
| **Range** | `[0.0, 100.0]` |
| **Precision** | 1 decimal place for display (`:.0f%`) |
| **Default** | `0.0` |

### Semantic Meaning

> GPU memory usage as a percentage of reserved VRAM:
>
> `memory_usage_percent = (cuda_memory_allocated_gb / cuda_memory_reserved_gb) * 100`
>
> Shows the effective utilization of reserved GPU memory. PyTorch allocates memory in blocks; this metric reveals fragmentation and actual vs. reserved overhead.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `memory_usage_percent < 75%` | Safe operating range, GPU memory not constrained |
| **Warning** | `75% <= memory_usage_percent <= 90%` | Approaching saturation, monitor closely |
| **Critical** | `memory_usage_percent > 90%` | Critical memory pressure, OOM risk imminent |

**Threshold Source:** `StatusBanner._append_metrics()` lines 164-170

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | CUDA memory metrics collected during PPO update |
| **File** | `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Function/Method** | `PPOAgent._collect_infrastructure_metrics()` |
| **Line(s)** | 336-348 |

```python
# Infrastructure metrics collection (per PyTorch expert review)
allocated = torch.cuda.memory_allocated(self.device) / (1024**3)  # GB
reserved = torch.cuda.memory_reserved(self.device) / (1024**3)  # GB
# ... peak and fragmentation computation ...
return {
    "cuda_memory_allocated_gb": allocated,
    "cuda_memory_reserved_gb": reserved,
    "cuda_memory_peak_gb": peak,
    "cuda_memory_fragmentation": fragmentation,
}
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Collected in `_collect_infrastructure_metrics()` | `simic/agent/ppo.py` |
| **2. Collection** | Passed to `emit_ppo_update_event()` via metrics dict | `simic/telemetry/emitters.py` |
| **3. Aggregation** | Embedded in `PPOUpdatePayload` | `leyline/telemetry.py` |
| **4. Delivery** | Handler unpacks to `tamiyo.infrastructure` fields | `karn/sanctum/aggregator.py` |

```
[PPOAgent._collect_infrastructure_metrics()]
    ↓ (metrics dict)
[emit_ppo_update_event(metrics={...})]
    ↓ (PPOUpdatePayload)
[PPOUpdatePayload(cuda_memory_allocated_gb=..., cuda_memory_reserved_gb=...)]
    ↓ (TelemetryEvent)
[Aggregator._handle_ppo_update()]
    ↓ (direct field assignment)
[TamiyoState.infrastructure.cuda_memory_allocated_gb/reserved_gb]
    ↓ (computed property)
[TamiyoState.infrastructure.memory_usage_percent]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `InfrastructureMetrics` (nested in `TamiyoState`) |
| **Property** | `memory_usage_percent` (computed property, not stored field) |
| **Constituent Fields** | `cuda_memory_allocated_gb`, `cuda_memory_reserved_gb` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.infrastructure.memory_usage_percent` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Lines** | 796-821 (InfrastructureMetrics dataclass, property at 815-820) |

```python
@dataclass
class InfrastructureMetrics:
    """PyTorch infrastructure health metrics."""
    cuda_memory_allocated_gb: float = 0.0
    cuda_memory_reserved_gb: float = 0.0
    # ... other metrics ...

    @property
    def memory_usage_percent(self) -> float:
        """Memory usage as percentage for compact display."""
        if self.cuda_memory_reserved_gb <= 0:
            return 0.0
        return (self.cuda_memory_allocated_gb / self.cuda_memory_reserved_gb) * 100
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| StatusBanner | `widgets/tamiyo_brain/status_banner.py` | Real-time display `[Mem:45%]` with color thresholds |

**Consumer Code Reference:** `status_banner.py` lines 161-170

```python
# Memory as percentage (per UX review - more scannable than absolute)
mem_pct = self._snapshot.tamiyo.infrastructure.memory_usage_percent
if mem_pct > 0:
    if mem_pct > 90:
        mem_style = "red bold"
    elif mem_pct > 75:
        mem_style = "yellow"
    else:
        mem_style = "dim"
    banner.append(f"  [Mem:{mem_pct:.0f}%]", style=mem_style)
```

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `PPOAgent._collect_infrastructure_metrics()` computes CUDA memory metrics
- [x] **Transport works** — Metrics passed through `emit_ppo_update_event()` → `PPOUpdatePayload`
- [x] **Schema field exists** — `InfrastructureMetrics.cuda_memory_allocated_gb/reserved_gb` fields present
- [x] **Property is correct** — `memory_usage_percent` property correctly computes percentage
- [x] **Consumer reads it** — `StatusBanner._append_metrics()` accesses and displays it
- [x] **Display is correct** — Value renders with `:.0f%` formatting, positioned as `[Mem:XX%]`
- [x] **Thresholds applied** — Color coding: <75% dim, 75-90% yellow, >90% red bold

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/agent/test_ppo.py` | Infrastructure metrics collection tests | `[x]` |
| Unit (schema) | `tests/karn/sanctum/test_schema.py` | InfrastructureMetrics property tests | `[x]` |
| Integration (end-to-end) | `tests/integration/test_telemetry_flow.py` | Memory usage reaches widget | `[x]` |
| Visual (TUI snapshot) | — | Manual verification in StatusBanner | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe StatusBanner header line
4. Verify `[Mem:XX%]` appears after batch counter
5. Monitor GPU load to trigger threshold coloring:
   - Normal load: `[Mem:45%]` in dim style
   - Moderate load: `[Mem:82%]` in yellow style
   - High load: `[Mem:95%]` in red bold style

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| CUDA device | hardware | Only populated on GPU; returns 0.0 on CPU |
| PPO update cycle | event | Emitted during `update()` phase every batch |
| `torch.cuda.memory_allocated()` | library | PyTorch CUDA API function |
| `torch.cuda.memory_reserved()` | library | PyTorch CUDA API function |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| StatusBanner status color | display | Memory >90% contributes to alert coloring |
| Memory alarm system | system | `SystemVitals.has_memory_alarm` checks this via composite logic |
| Training operator awareness | human | Operator sees warning/critical indicators in real-time |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Audit | Initial creation with complete wiring verification |

---

## 8. Notes

> **Design Decision:** Percentage display chosen over absolute GB values for superior scanability. A user can quickly gauge "80% full" vs. "92% full" at a glance, whereas "6.2 GB of 7.1 GB" requires mental arithmetic (per UX specialist review).

> **CPU Behavior:** The metric safely returns 0.0 on CPU devices since `torch.cuda.memory_allocated()` would fail. StatusBanner ignores zero values, so the memory display is omitted on CPU-only training.

> **Memory Fragmentation Context:** This percentage reflects allocator overhead—PyTorch reserves blocks even when partially used. The difference between allocated and reserved is `cuda_memory_fragmentation = 1 - (allocated/reserved)`. When fragmentation >0.3, users should consider model checkpointing or batch size reduction.

> **Threshold Rationale:**
> - **75% warning:** PyTorch CUDA allocation can spike unexpectedly during parameter updates; 75% leaves 25% headroom
> - **90% critical:** Less than 1 GB headroom on typical GPUs (8-10 GB); OOM likely on next batch

> **Integration Status:** Fully wired and operational. The metric appears in every StatusBanner update and is color-coded for quick operator scanning.

# Telemetry Record: [TELE-557] Seed Grad Ratio

> **Status:** `[x] Planned` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-557` |
| **Name** | Seed Grad Ratio |
| **Category** | `seed` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Is this seed's gradient flow healthy?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
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
| **Type** | `float` |
| **Units** | ratio (dimensionless) |
| **Range** | `[0.0, ∞)` |
| **Precision** | 2 decimal places for display |
| **Default** | `0.0` (no gradient data) |

### Semantic Meaning

> Gradient ratio measures the health of gradient flow through the seed module. It's computed as the ratio of output gradient norm to input gradient norm:
>
> - **grad_ratio ≈ 1.0:** Healthy gradient flow
> - **grad_ratio << 1.0:** Potential vanishing gradients (triggers has_vanishing flag)
> - **grad_ratio >> 1.0:** Potential exploding gradients (triggers has_exploding flag)
> - **grad_ratio = 0.0:** No gradient data available
>
> The ratio is used alongside the has_vanishing and has_exploding boolean flags for health assessment.

### Health Thresholds

| Context | Condition | Meaning |
|---------|-----------|---------|
| Any stage | 0.5 ≤ grad_ratio ≤ 2.0 | Healthy gradient flow |
| Any stage | grad_ratio < 0.5 | Vanishing risk |
| Any stage | grad_ratio > 2.0 | Exploding risk |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Gradient hook computation, computed per backward pass |
| **File** | `/home/john/esper-lite/src/esper/kasmina/slot.py` |
| **Class** | `SeedState` |
| **Field** | `grad_ratio: float` (schema line 398) |

```python
@dataclass
class SeedState:
    """State of a single seed slot."""
    grad_ratio: float = 0.0
    has_vanishing: bool = False
    has_exploding: bool = False
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Computation** | Gradient hooks compute ratio during backward | `kasmina/gradient_monitor.py` |
| **2. Storage** | SeedState.grad_ratio updated | `kasmina/slot.py` |
| **3. Telemetry** | SeedTelemetry.grad_ratio field | `leyline/telemetry.py` |
| **4. Schema** | SeedState.grad_ratio in Sanctum | `karn/sanctum/schema.py` (line 398) |
| **5. Display** | EnvDetailScreen SeedCard | `karn/sanctum/widgets/env_detail_screen.py` (lines 158-159) |

```
[Gradient hooks]
  --computes-->
  [SeedState.grad_ratio]
  --EPOCH_COMPLETED event-->
  [SeedTelemetry.grad_ratio]
  --SanctumAggregator-->
  [schema.SeedState.grad_ratio]
  --SeedCard-->
  ["ratio=0.85"]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SeedState` (Sanctum schema) |
| **Field** | `grad_ratio` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].seeds[slot_id].grad_ratio` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 398 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvDetailScreen SeedCard | `widgets/env_detail_screen.py` (lines 158-159) | "ratio=0.85" when healthy |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — Gradient hooks compute ratio per backward pass
- [x] **Transport works** — SeedState.grad_ratio updated; telemetry carries value
- [x] **Schema field exists** — `SeedState.grad_ratio: float = 0.0` at line 398
- [x] **Default is correct** — `0.0` is correct for no gradient data
- [x] **Consumer reads it** — SeedCard._render_active() displays `seed.grad_ratio`
- [x] **Display is correct** — Format: "ratio=0.85" green, "OK" if no ratio but healthy

### Display Code

```python
# From env_detail_screen.py lines 152-162
# Gradient health
grad_text = Text("Grad: ")
if seed.has_exploding:
    grad_text.append("▲ EXPLODING", style="bold red")
elif seed.has_vanishing:
    grad_text.append("▼ VANISHING", style="bold yellow")
elif seed.grad_ratio is not None and seed.grad_ratio > 0:
    grad_text.append(f"ratio={seed.grad_ratio:.2f}", style="green")
else:
    grad_text.append("OK", style="green")
lines.append(grad_text)
```

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Gradient hooks | computation | Captures input/output gradients |
| Backward pass | execution | Gradients computed during backprop |
| has_vanishing/has_exploding | flags | Binary health indicators |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| SeedCard display | display | Shows gradient health status |
| Prune decisions | training | Unhealthy gradients may trigger prune |
| Training stability alerts | monitoring | Flags gradient issues early |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation for EnvDetailScreen SeedCard |

---

## 8. Notes

> **Related Fields:** grad_ratio works in conjunction with has_vanishing and has_exploding boolean flags:
> - **has_exploding=True:** Displayed as "▲ EXPLODING" (bold red)
> - **has_vanishing=True:** Displayed as "▼ VANISHING" (bold yellow)
> - **Neither flag + grad_ratio > 0:** Displayed as "ratio=0.85" (green)
> - **Neither flag + grad_ratio = 0:** Displayed as "OK" (green)
>
> **Display Priority:** Exploding takes precedence over vanishing, which takes precedence over ratio display.
>
> **Update Frequency:** Gradient ratio is computed during each backward pass and reflects the current epoch's gradient health.

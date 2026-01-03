# Telemetry Record: [TELE-556] Seed Accuracy Delta

> **Status:** `[x] Planned` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-556` |
| **Name** | Seed Accuracy Delta |
| **Category** | `seed` |
| **Priority** | `P0-critical` |

## 2. Purpose

### What question does this answer?

> "How much is this seed improving or hurting the model's accuracy?"

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
| **Units** | percentage points (%) |
| **Range** | `(-∞, +∞)` |
| **Precision** | 2 decimal places for display |
| **Default** | `0.0` (no contribution) |

### Semantic Meaning

> Accuracy delta measures the per-seed accuracy contribution computed via counterfactual analysis. It represents the difference in accuracy when this seed is included versus excluded:
>
> - **accuracy_delta > 0:** Seed is helping (positive contribution)
> - **accuracy_delta = 0:** Seed has no measurable effect
> - **accuracy_delta < 0:** Seed is hurting (negative contribution)
>
> Stage-aware semantics:
> - **TRAINING/GERMINATED:** Always 0.0 — seed has alpha=0 and cannot affect output
> - **BLENDING/HOLDING:** Active contribution measurement
> - **FOSSILIZED:** Final contribution snapshot at integration time

### Health Thresholds

| Context | Condition | Meaning |
|---------|-----------|---------|
| BLENDING | accuracy_delta > 0 | Healthy — seed improving accuracy |
| BLENDING | accuracy_delta ≈ 0 | Neutral — seed not contributing |
| BLENDING | accuracy_delta < 0 | Warning — seed hurting accuracy |
| TRAINING/GERMINATED | N/A | Always 0.0 (learning phase) |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Counterfactual engine, computed per epoch |
| **File** | `/home/john/esper-lite/src/esper/kasmina/slot.py` |
| **Class** | `SeedState` |
| **Field** | `accuracy_delta: float` (schema line 396) |

```python
@dataclass
class SeedState:
    """State of a single seed slot."""
    accuracy_delta: float = 0.0
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Computation** | Counterfactual engine computes contribution | `tamiyo/counterfactual.py` |
| **2. Storage** | SeedState.accuracy_delta updated | `kasmina/slot.py` |
| **3. Telemetry** | SeedTelemetry.accuracy_delta field | `leyline/telemetry.py` |
| **4. Schema** | SeedState.accuracy_delta in Sanctum | `karn/sanctum/schema.py` (line 396) |
| **5. Display** | EnvDetailScreen SeedCard | `karn/sanctum/widgets/env_detail_screen.py` (lines 142-150) |

```
[Counterfactual engine]
  --computes-->
  [SeedState.accuracy_delta]
  --EPOCH_COMPLETED event-->
  [SeedTelemetry.accuracy_delta]
  --SanctumAggregator-->
  [schema.SeedState.accuracy_delta]
  --SeedCard-->
  ["Acc Δ: +0.25%"]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SeedState` (Sanctum schema) |
| **Field** | `accuracy_delta` |
| **Path from SanctumSnapshot** | `snapshot.envs[env_id].seeds[slot_id].accuracy_delta` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 396 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| EnvDetailScreen SeedCard | `widgets/env_detail_screen.py` (lines 142-150) | "Acc Δ: +0.25%" with color coding |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — Counterfactual engine computes accuracy_delta per epoch
- [x] **Transport works** — SeedState.accuracy_delta updated; telemetry carries value
- [x] **Schema field exists** — `SeedState.accuracy_delta: float = 0.0` at line 396
- [x] **Default is correct** — `0.0` is correct for non-contributing seeds
- [x] **Consumer reads it** — SeedCard._render_active() displays `seed.accuracy_delta`
- [x] **Display is correct** — Format: "+0.25%" green, "-0.15%" red, "0.0 (learning)" for TRAINING

### Display Code

```python
# From env_detail_screen.py lines 142-150
# Accuracy delta (stage-aware display)
# TRAINING/GERMINATED seeds have alpha=0 and cannot affect output
if seed.stage in ("TRAINING", "GERMINATED"):
    lines.append(Text("Acc Δ: 0.0 (learning)", style="dim italic"))
elif seed.accuracy_delta is not None and seed.accuracy_delta != 0:
    delta_style = "green" if seed.accuracy_delta > 0 else "red"
    lines.append(Text(f"Acc Δ: {seed.accuracy_delta:+.2f}%", style=delta_style))
else:
    lines.append(Text("Acc Δ: --", style="dim"))
```

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Counterfactual engine | computation | Computes contribution via leave-one-out |
| Seed alpha | parameter | Seeds with alpha=0 have no contribution |
| Model accuracy | metric | Baseline for delta calculation |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| SeedCard display | display | Shows contribution with color coding |
| Tamiyo decisions | training | Negative delta may trigger prune |
| Fossilization gate | lifecycle | Must show positive contribution |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation for EnvDetailScreen SeedCard |

---

## 8. Notes

> **Stage-Aware Display:** TRAINING and GERMINATED seeds always show "0.0 (learning)" because they have alpha=0 and cannot affect the host model's output. This is not an error — it's the expected behavior during the learning phase.
>
> **Color Coding:**
> - **Green:** Positive delta — seed is helping
> - **Red:** Negative delta — seed is hurting (may trigger prune consideration)
> - **Dim:** Zero or null — no measurable contribution
>
> **Update Frequency:** Accuracy delta is computed once per epoch by the counterfactual engine. It reflects the instantaneous contribution, not a cumulative total.

# Telemetry Record: [TELE-321] Per-Head Previous Gradient Norms

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-321` |
| **Name** | Per-Head Previous Gradient Norms |
| **Category** | `gradient` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Are gradient norms trending upward (toward explosion) or downward (toward vanishing) for each action head?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [ ] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

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
| **Type** | `float` (8 fields) |
| **Units** | L2 norm (unitless) |
| **Range** | `[0.0, inf)` — non-negative |
| **Precision** | 3 decimal places for display |
| **Default** | `0.0` each |

### Fields

| Field | Description |
|-------|-------------|
| `head_op_grad_norm_prev` | Previous gradient norm for op (action type) head |
| `head_slot_grad_norm_prev` | Previous gradient norm for slot head |
| `head_blueprint_grad_norm_prev` | Previous gradient norm for blueprint head |
| `head_style_grad_norm_prev` | Previous gradient norm for style head |
| `head_tempo_grad_norm_prev` | Previous gradient norm for tempo head |
| `head_alpha_target_grad_norm_prev` | Previous gradient norm for alpha target head |
| `head_alpha_speed_grad_norm_prev` | Previous gradient norm for alpha speed head |
| `head_alpha_curve_grad_norm_prev` | Previous gradient norm for alpha curve head |

### Semantic Meaning

> These fields store the gradient norm from the previous PPO update for each action head. They are not emitted directly by the training loop; instead, the aggregator maintains them by copying the current gradient norm to the `_prev` field before updating with the new value.
>
> The primary use is computing trend arrows in the TUI:
> - `delta = current_grad_norm - prev_grad_norm`
> - `abs(delta) < 0.01` → stable (→)
> - `delta > 0` → increasing (↗)
> - `delta < 0` → decreasing (↘)
>
> Trend context determines arrow color:
> - Low gradients (`< 0.1`): increasing is good (green), decreasing is bad (red)
> - High gradients (`> 2.0`): increasing is bad (red), decreasing is good (green)
> - Normal range: neutral (dim)

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `abs(delta) < 0.01` | Gradient stable between updates |
| **Context-dependent** | `delta != 0` | Trend interpreted based on current gradient magnitude |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Not directly emitted — computed in aggregator |
| **File** | N/A (derived field) |
| **Function/Method** | N/A |
| **Line(s)** | N/A |

> **Note:** The `_prev` fields are not present in `PPOUpdatePayload`. The aggregator computes them by saving the current `head_*_grad_norm` value to `head_*_grad_norm_prev` before overwriting with the new value from the payload.

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `head_*_grad_norm` emitted via PPO update | `simic/agent/ppo.py` |
| **2. Collection** | Event includes per-head grad norms | `leyline/telemetry.py` |
| **3. Aggregation** | Aggregator saves current to `_prev`, then updates current | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Written to `snapshot.tamiyo.head_*_grad_norm_prev` | `karn/sanctum/schema.py` |

```
[PPOAgent] --emit_ppo_update(head_*_grad_norm)--> [TelemetryEmitter]
    |
    v
[Aggregator] --before update: _prev = current, current = new-->
    |
    v
[TamiyoState.head_*_grad_norm_prev]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Fields** | `head_op_grad_norm_prev`, `head_slot_grad_norm_prev`, `head_blueprint_grad_norm_prev`, `head_style_grad_norm_prev`, `head_tempo_grad_norm_prev`, `head_alpha_target_grad_norm_prev`, `head_alpha_speed_grad_norm_prev`, `head_alpha_curve_grad_norm_prev` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.head_*_grad_norm_prev` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Lines** | 916-923 |

### Aggregator Logic

```python
# From aggregator.py lines 885-910
if payload.head_slot_grad_norm is not None:
    self._tamiyo.head_slot_grad_norm_prev = self._tamiyo.head_slot_grad_norm
    self._tamiyo.head_slot_grad_norm = payload.head_slot_grad_norm
# ... repeated for all 8 heads
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| ActionHeadsPanel | `widgets/tamiyo_brain/action_heads_panel.py` | Trend arrow computation (lines 354-364) |

```python
# From action_heads_panel.py lines 353-364
grad: float = getattr(tamiyo, grad_field)
grad_prev: float = getattr(tamiyo, f"{grad_field}_prev")
trend = self._gradient_trend(grad, grad_prev)
# ...
result.append(trend, style=self._gradient_trend_style(grad, grad_prev))
```

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — Current grad norms emitted, prev computed in aggregator
- [x] **Transport works** — Aggregator correctly saves current to prev before update
- [x] **Schema field exists** — All 8 `head_*_grad_norm_prev` fields defined in TamiyoState
- [x] **Default is correct** — 0.0 appropriate (first update shows no trend)
- [x] **Consumer reads it** — ActionHeadsPanel uses `{grad_field}_prev` pattern
- [x] **Display is correct** — Trend arrows rendered with context-sensitive coloring
- [x] **Thresholds applied** — `_gradient_trend_style()` uses 0.1/2.0 thresholds

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_aggregator_tracks_previous_gradient_norms` | `[x]` |
| Integration (end-to-end) | — | Manual verification | `[x]` |
| Visual (TUI snapshot) | — | Manual verification | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens)
3. Observe ActionHeadsPanel "Grad" row
4. Verify trend arrows appear after second PPO update
5. Watch for arrow direction changes as training progresses
6. Verify arrow colors match gradient health context (green/red when recovering/exploding)

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `TELE-320` (head_*_grad_norm) | telemetry | Current values must be populated first |
| PPO update cycle | event | Only updated after each PPO batch |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| ActionHeadsPanel trend arrows | display | Visual indicator of gradient trajectory |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Initial creation during telemetry audit |

---

## 8. Notes

> **Design Decision:** These fields are computed in the aggregator rather than emitted by the training loop. This keeps the telemetry payload simpler and ensures the "previous" value is always exactly one update behind, regardless of any emission timing issues.
>
> **First Update Behavior:** On the first PPO update, all `_prev` fields remain at 0.0 (default), so the trend will show "increasing" if the first gradient norm is positive. This is expected and not a bug.
>
> **Relationship to TELE-320:** The current gradient norm fields (TELE-320) and previous gradient norm fields (TELE-321) work together. The current values are the source of truth; the previous values enable temporal analysis.
